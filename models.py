#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: models.py
ROLE: Asymmetric Hybrid LoRA-CLIP Architecture, Bottleneck Residual Adapters &
      Multi-Template Semantic Prototype Generator
PROJECT: SeCo-LoRA-CLIP — Zero-Shot & Generalized Sketch-Based Image Retrieval
================================================================================
"""

import math
from typing import List, Dict, Optional, Tuple

from config import TORCH_AVAILABLE, CLIP_PROMPT_TEMPLATES

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRALinearLayer(nn.Module):
        """
        Low-Rank Adaptation (LoRA) module for a linear projection W_0 * x + (alpha/r) * B * A * x.
        Initialized with Kaiming-uniform/normal on A and strictly zeros on B so that delta W = 0
        at Epoch 0, preserving the exact pre-trained CLIP manifold at initialization.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.05,
        ):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank
            self.scaling = alpha / float(max(rank, 1))
            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

            self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))

            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_fp32 = self.dropout(x.float())
            delta = (x_fp32 @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return delta.to(x.dtype)

    class LoRAMultiheadAttention(nn.Module):
        """
        Custom Multi-Head Self-Attention wrapper that explicitly computes Q, K, V, and Out
        projections in PyTorch Python space (avoiding C++ F.multi_head_attention_forward
        bypass of out_proj module hooks) and supports toggling LoRA on/off dynamically.
        """

        def __init__(
            self,
            orig_attn: nn.MultiheadAttention,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.05,
            target_qkv: bool = True,
        ):
            super().__init__()
            self.embed_dim = orig_attn.embed_dim
            self.num_heads = orig_attn.num_heads
            self.head_dim = self.embed_dim // self.num_heads
            self.target_qkv = target_qkv
            self.lora_enabled: bool = True

            # Freeze original attention weights & biases
            self.in_proj_weight = nn.Parameter(orig_attn.in_proj_weight.data.clone(), requires_grad=False)
            if orig_attn.in_proj_bias is not None:
                self.in_proj_bias = nn.Parameter(orig_attn.in_proj_bias.data.clone(), requires_grad=False)
            else:
                self.register_parameter("in_proj_bias", None)

            self.out_proj_weight = nn.Parameter(orig_attn.out_proj.weight.data.clone(), requires_grad=False)
            if orig_attn.out_proj.bias is not None:
                self.out_proj_bias = nn.Parameter(orig_attn.out_proj.bias.data.clone(), requires_grad=False)
            else:
                self.register_parameter("out_proj_bias", None)

            if self.target_qkv and rank > 0:
                self.lora_q = LoRALinearLayer(self.embed_dim, self.embed_dim, rank, alpha, dropout)
                self.lora_k = LoRALinearLayer(self.embed_dim, self.embed_dim, rank, alpha, dropout)
                self.lora_v = LoRALinearLayer(self.embed_dim, self.embed_dim, rank, alpha, dropout)
            else:
                self.lora_q = None
                self.lora_k = None
                self.lora_v = None

            if rank > 0:
                self.lora_out = LoRALinearLayer(self.embed_dim, self.embed_dim, rank, alpha, dropout)
            else:
                self.lora_out = None

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            need_weights: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # CLIP ViT uses sequence-first format: (L, B, D)
            seq_len, bsz, embed_dim = query.shape
            w_dtype = self.in_proj_weight.dtype
            q_in = query.to(w_dtype)

            # Base QKV linear projection
            qkv = F.linear(q_in, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            # Inject LoRA into Q, K, V if enabled
            if self.lora_enabled and self.target_qkv and self.lora_q is not None:
                q = q + self.lora_q(q_in)
                k = k + self.lora_k(q_in)
                v = v + self.lora_v(q_in)

            # Reshape for multi-head scaled dot-product attention: (B, num_heads, L, head_dim)
            q = q.contiguous().view(seq_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            k = k.contiguous().view(seq_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            v = v.contiguous().view(seq_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

            # Restore shape to (L, B, D)
            attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(seq_len, bsz, embed_dim)

            # Out projection + LoRA Out
            out = F.linear(attn_out, self.out_proj_weight, self.out_proj_bias)
            if self.lora_enabled and self.lora_out is not None:
                out = out + self.lora_out(attn_out)

            return out.to(query.dtype), None

    class BottleneckResidualAdapter(nn.Module):
        """
        Lightweight Bottleneck Residual Adapter (SeCo-SBIR Sec. 4.4) with zero-initialized
        second layer so the residual path starts as an exact identity mapping on the unit
        hypersphere.
        """

        def __init__(self, embed_dim: int = 512, bottleneck_dim: int = 256, res_scale: float = 0.2):
            super().__init__()
            self.fc1 = nn.Linear(embed_dim, bottleneck_dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(bottleneck_dim, embed_dim)
            self.res_scale = res_scale

            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_fp32 = F.normalize(x.float(), dim=-1)
            residual = self.fc2(self.act(self.fc1(x_fp32)))
            out = x_fp32 + self.res_scale * residual
            return F.normalize(out, dim=-1)

    class AsymmetricHybridLoRACLIP(nn.Module):
        """
        SeCo-LoRA-CLIP Architecture:
          - Single Visual Backbone (Zero VRAM Duplication):
              * When encoding Frozen Reference or Photos (`lora_enabled=False` and Frozen LayerNorm
                stats), acts as the 100% frozen pre-trained CLIP ViT.
              * When encoding Sketches (`lora_enabled=True` and adapted LayerNorm), activates
                LoRA adapters in Q, K, V, and Out projections across all 12 ViT blocks.
          - Modality-Specific Bottleneck Residual Adapters for Sketch, Photo, and Text.
        """

        def __init__(
            self,
            clip_model: nn.Module,
            lora_rank: int = 8,
            lora_alpha: float = 16.0,
            lora_dropout: float = 0.05,
            lora_target_qkv: bool = True,
            embed_dim: int = 512,
            bottleneck_dim: int = 256,
            res_scale: float = 0.2,
            unfreeze_sketch_ln: bool = True,
            freeze_photo_adapter: bool = False,
        ):
            super().__init__()
            self.unfreeze_sketch_ln = unfreeze_sketch_ln
            self.freeze_photo_adapter = freeze_photo_adapter

            self.visual = clip_model.visual
            for p in self.visual.parameters():
                p.requires_grad = False

            # Store frozen LayerNorm weights/biases so `encode_frozen_visual` is 100% exact
            self._frozen_ln_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for name, m in self.visual.named_modules():
                if isinstance(m, nn.LayerNorm):
                    self._frozen_ln_params[name] = (
                        m.weight.data.clone().detach(),
                        m.bias.data.clone().detach(),
                    )
                    if unfreeze_sketch_ln:
                        m.weight.requires_grad = True
                        m.bias.requires_grad = True

            self.lora_attn_modules: nn.ModuleList = nn.ModuleList()
            if hasattr(self.visual, "transformer"):
                for block in self.visual.transformer.resblocks:
                    if hasattr(block, "attn") and isinstance(block.attn, nn.MultiheadAttention):
                        lora_attn = LoRAMultiheadAttention(
                            orig_attn=block.attn,
                            rank=lora_rank,
                            alpha=lora_alpha,
                            dropout=lora_dropout,
                            target_qkv=lora_target_qkv,
                        ).to(next(self.visual.parameters()).device)
                        block.attn = lora_attn
                        self.lora_attn_modules.append(lora_attn)

            self.sketch_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)
            self.photo_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)
            self.text_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)

            if self.freeze_photo_adapter:
                for p in self.photo_adapter.parameters():
                    p.requires_grad = False

        def _set_lora_mode(self, enabled: bool) -> None:
            for mod in self.lora_attn_modules:
                mod.lora_enabled = enabled

        @torch.no_grad()
        def encode_frozen_visual(self, img_tensor: torch.Tensor) -> torch.Tensor:
            """
            Passes images through the 100% original frozen CLIP ViT (LoRA disabled, original
            LayerNorm parameters restored during forward pass).
            """
            self._set_lora_mode(False)

            saved_ln: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            if self.unfreeze_sketch_ln:
                for name, m in self.visual.named_modules():
                    if isinstance(m, nn.LayerNorm) and name in self._frozen_ln_params:
                        saved_ln[name] = (m.weight.data.clone(), m.bias.data.clone())
                        frz_w, frz_b = self._frozen_ln_params[name]
                        m.weight.data.copy_(frz_w.to(m.weight.device, dtype=m.weight.dtype))
                        m.bias.data.copy_(frz_b.to(m.bias.device, dtype=m.bias.dtype))

            try:
                dtype = next(self.visual.parameters()).dtype
                feat = self.visual(img_tensor.to(dtype)).float()
                out = F.normalize(feat, dim=-1)
            finally:
                if self.unfreeze_sketch_ln:
                    for name, m in self.visual.named_modules():
                        if isinstance(m, nn.LayerNorm) and name in saved_ln:
                            cur_w, cur_b = saved_ln[name]
                            m.weight.data.copy_(cur_w)
                            m.bias.data.copy_(cur_b)
                self._set_lora_mode(True)

            return out

        def encode_sketch(self, sketch_tensor: torch.Tensor) -> torch.Tensor:
            """Encodes sketches through the LoRA-adapted ViT + Sketch Bottleneck Residual Adapter."""
            self._set_lora_mode(True)
            dtype = next(self.visual.parameters()).dtype
            raw_feat = self.visual(sketch_tensor.to(dtype)).float()
            return self.sketch_adapter(F.normalize(raw_feat, dim=-1))

        def encode_photo(self, photo_tensor: torch.Tensor) -> torch.Tensor:
            """Encodes natural photos through the frozen CLIP ViT + Photo Residual Adapter."""
            raw_feat = self.encode_frozen_visual(photo_tensor)
            if self.freeze_photo_adapter:
                return raw_feat
            return self.photo_adapter(raw_feat)

        def encode_photo_from_cached(self, cached_photo_feat: torch.Tensor) -> torch.Tensor:
            """Fast path: projects pre-cached frozen CLIP photo embeddings via Photo Residual Adapter."""
            norm_feat = F.normalize(cached_photo_feat.float(), dim=-1)
            if self.freeze_photo_adapter:
                return norm_feat
            return self.photo_adapter(norm_feat)

        def adapt_text_prototypes(self, raw_text_prototypes: torch.Tensor) -> torch.Tensor:
            """Projects pre-extracted CLIP text class prototypes into the adapted joint space."""
            return self.text_adapter(F.normalize(raw_text_prototypes.float(), dim=-1))

        def get_optimizer_param_groups(self, lr: float = 1e-4, ln_lr: float = 2e-5, weight_decay: float = 1e-4):
            """Separates LayerNorm parameters (lower LR, zero weight decay) from LoRA & Adapters."""
            lora_and_adapter_params = []
            ln_params = []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if "ln_" in name or "LayerNorm" in name:
                    ln_params.append(p)
                else:
                    lora_and_adapter_params.append(p)

            groups = [
                {"params": lora_and_adapter_params, "lr": lr, "weight_decay": weight_decay},
            ]
            if ln_params:
                groups.append({"params": ln_params, "lr": ln_lr, "weight_decay": 0.0})
            return groups

        def get_trainable_state_dict(self) -> Dict[str, torch.Tensor]:
            """Returns a compact state_dict containing ONLY trainable parameters (~1.8 MB)."""
            return {
                name: p.detach().cpu()
                for name, p in self.named_parameters()
                if p.requires_grad
            }

        def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
            """Loads compact trainable parameters into the model."""
            own_state = dict(self.named_parameters())
            for name, param in state_dict.items():
                if name in own_state:
                    own_state[name].data.copy_(param.to(own_state[name].device, dtype=own_state[name].dtype))

        def count_parameters(self) -> Dict[str, float]:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            lora_params = sum(p.numel() for m in self.lora_attn_modules for p in m.parameters() if p.requires_grad)
            adapter_params = sum(
                p.numel()
                for mod in [self.sketch_adapter, self.photo_adapter, self.text_adapter]
                for p in mod.parameters()
                if p.requires_grad
            )
            ln_params = trainable - lora_params - adapter_params
            return {
                "total_params": int(total),
                "trainable_params": int(trainable),
                "lora_params": int(lora_params),
                "adapter_params": int(adapter_params),
                "layernorm_params": int(ln_params),
                "trainable_ratio_pct": round(100.0 * trainable / max(total, 1), 4),
            }

    @torch.no_grad()
    def build_clip_text_prototypes(
        clip_model: nn.Module,
        tokenize_fn,
        class_names: List[str],
        device: torch.device,
        use_ensemble: bool = True,
    ) -> torch.Tensor:
        """
        Builds normalized category-level semantic prototypes from CLIP's text encoder.
        Supports multi-template prompt ensembling across 7 canonical photo/sketch templates.
        """
        templates = CLIP_PROMPT_TEMPLATES if use_ensemble else ("a photo of a {}.",)
        cleaned_names = [
            c.replace("_", " ").replace("(", "").replace(")", "").strip()
            for c in class_names
        ]

        all_template_feats = []
        for tmpl in templates:
            prompts = [tmpl.format(cname) for cname in cleaned_names]
            tokens = tokenize_fn(prompts).to(device)
            text_feats = clip_model.encode_text(tokens).float()
            text_feats = F.normalize(text_feats, dim=-1)
            all_template_feats.append(text_feats)

        mean_feats = torch.stack(all_template_feats, dim=0).mean(dim=0)
        return F.normalize(mean_feats, dim=-1)