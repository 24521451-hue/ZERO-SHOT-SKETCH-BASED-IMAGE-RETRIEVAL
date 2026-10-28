#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import math
from typing import List, Dict

from config import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LowRankLinearAdapter(nn.Module):
        def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
            super().__init__()
            self.base_linear = base_linear
            for p in self.base_linear.parameters():
                p.requires_grad = False

            in_features = base_linear.in_features
            out_features = base_linear.out_features
            self.rank = rank
            self.scaling = alpha / float(rank)

            self.lora_A = nn.Parameter(
                torch.randn(rank, in_features, dtype=base_linear.weight.dtype) * (1.0 / math.sqrt(rank))
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, rank, dtype=base_linear.weight.dtype)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base_linear(x)
            lora_out = (x.to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return base_out + lora_out.to(base_out.dtype)

    class BottleneckResidualAdapter(nn.Module):
        def __init__(self, embed_dim: int = 512, bottleneck_dim: int = 256, res_scale: float = 0.2):
            super().__init__()
            self.fc1 = nn.Linear(embed_dim, bottleneck_dim)
            self.act = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(bottleneck_dim, embed_dim)
            self.res_scale = res_scale

            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_fp32 = x.float()
            residual = self.fc2(self.act(self.fc1(x_fp32)))
            out = x_fp32 + self.res_scale * residual
            return F.normalize(out, dim=-1)

    class AsymmetricHybridLoRACLIP(nn.Module):
        def __init__(
            self,
            clip_model: nn.Module,
            lora_rank: int = 8,
            lora_alpha: float = 16.0,
            embed_dim: int = 512,
            bottleneck_dim: int = 256,
            res_scale: float = 0.2,
            unfreeze_sketch_ln: bool = True,
        ):
            super().__init__()

            self.frozen_visual = copy.deepcopy(clip_model.visual).eval()
            for p in self.frozen_visual.parameters():
                p.requires_grad = False

            self.sketch_visual = clip_model.visual
            for p in self.sketch_visual.parameters():
                p.requires_grad = False

            if hasattr(self.sketch_visual, "transformer"):
                for block in self.sketch_visual.transformer.resblocks:
                    if unfreeze_sketch_ln:
                        for m in block.modules():
                            if isinstance(m, nn.LayerNorm):
                                for p in m.parameters():
                                    p.requires_grad = True
                    if hasattr(block, "attn") and hasattr(block.attn, "out_proj"):
                        orig_linear = block.attn.out_proj
                        lora_mod = LowRankLinearAdapter(orig_linear, rank=lora_rank, alpha=lora_alpha)
                        block.add_module("lora_attn_out", lora_mod)

                        def _make_hook(l_mod):
                            def _hook(module, args, output):
                                x_in = args[0]
                                delta = (
                                    x_in.to(l_mod.lora_A.dtype) @ l_mod.lora_A.t() @ l_mod.lora_B.t()
                                ) * l_mod.scaling
                                return output + delta.to(output.dtype)
                            return _hook

                        orig_linear.register_forward_hook(_make_hook(lora_mod))

            self.sketch_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)
            self.photo_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)
            self.text_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)

        @torch.no_grad()
        def encode_frozen_visual(self, img_tensor: torch.Tensor) -> torch.Tensor:
            dtype = next(self.frozen_visual.parameters()).dtype
            feat = self.frozen_visual(img_tensor.to(dtype)).float()
            return F.normalize(feat, dim=-1)

        def encode_sketch(self, sketch_tensor: torch.Tensor) -> torch.Tensor:
            dtype = next(self.sketch_visual.parameters()).dtype
            raw_feat = self.sketch_visual(sketch_tensor.to(dtype)).float()
            return self.sketch_adapter(F.normalize(raw_feat, dim=-1))

        def encode_photo(self, photo_tensor: torch.Tensor) -> torch.Tensor:
            raw_feat = self.encode_frozen_visual(photo_tensor)
            return self.photo_adapter(raw_feat)

        def encode_photo_from_cached(self, cached_photo_feat: torch.Tensor) -> torch.Tensor:
            return self.photo_adapter(F.normalize(cached_photo_feat.float(), dim=-1))

        def adapt_text_prototypes(self, raw_text_prototypes: torch.Tensor) -> torch.Tensor:
            return self.text_adapter(F.normalize(raw_text_prototypes.float(), dim=-1))

        def count_parameters(self) -> Dict[str, int]:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return {
                "total_params": total,
                "trainable_params": trainable,
                "trainable_ratio_pct": round(100.0 * trainable / max(total, 1), 4),
            }

    @torch.no_grad()
    def build_clip_text_prototypes(
        clip_model: nn.Module,
        tokenize_fn,
        class_names: List[str],
        device: torch.device,
        template: str = "a photo of a {}",
    ) -> torch.Tensor:
        prompts = [
            template.format(c.replace("_", " ").replace("(", "").replace(")", "").strip())
            for c in class_names
        ]
        tokens = tokenize_fn(prompts).to(device)
        text_feats = clip_model.encode_text(tokens).float()
        return F.normalize(text_feats, dim=-1)
