#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: losses.py
ROLE: Multi-Objective Loss Function Suite (Triplet + NT-Xent + Semantic Class
      Prototype Alignment + Perturbation-Based Asymmetric InfoNCE Consistency)
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

from typing import Dict, Tuple, Optional
from config import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SeCoLoRAMultiObjectiveLoss(nn.Module):
        def __init__(
            self,
            margin: float = 0.3,
            temp_ntxent: float = 0.07,
            temp_cls: float = 0.07,
            temp_consist: float = 0.07,
            lambda_cls: float = 1.0,
            lambda_consist: float = 1.0,
        ):
            super().__init__()
            self.margin = margin
            self.temp_ntxent = temp_ntxent
            self.temp_cls = temp_cls
            self.temp_consist = temp_consist
            self.lambda_cls = lambda_cls
            self.lambda_consist = lambda_consist

        @staticmethod
        def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return 1.0 - (a * b).sum(dim=-1)

        def compute_triplet_loss(
            self, f_s: torch.Tensor, f_p_pos: torch.Tensor, f_p_neg: torch.Tensor
        ) -> torch.Tensor:
            d_pos = self.cosine_distance(f_s, f_p_pos)
            d_neg = self.cosine_distance(f_s, f_p_neg)
            return F.relu(self.margin + d_pos - d_neg).mean()

        def compute_nt_xent_loss(
            self, f_s: torch.Tensor, f_p_pos: torch.Tensor
        ) -> torch.Tensor:
            logits_s2p = (f_s @ f_p_pos.t()) / self.temp_ntxent
            targets = torch.arange(f_s.size(0), device=f_s.device)
            return 0.5 * (
                F.cross_entropy(logits_s2p, targets)
                + F.cross_entropy(logits_s2p.t(), targets)
            )

        def compute_classification_loss(
            self,
            f_s: torch.Tensor,
            f_p_pos: torch.Tensor,
            text_prototypes: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
            logits_s = (f_s @ text_prototypes.t()) / self.temp_cls
            logits_p = (f_p_pos @ text_prototypes.t()) / self.temp_cls
            return F.cross_entropy(logits_s, labels) + F.cross_entropy(logits_p, labels)

        def compute_consistency_loss(
            self,
            f_s_clean: torch.Tensor,
            f_s_frz_pert: torch.Tensor,
            f_p_clean: torch.Tensor,
            f_p_frz_pert: torch.Tensor,
        ) -> torch.Tensor:
            targets = torch.arange(f_s_clean.size(0), device=f_s_clean.device)
            logits_p = (f_p_frz_pert @ f_p_clean.t()) / self.temp_consist
            logits_s = (f_s_frz_pert @ f_s_clean.t()) / self.temp_consist
            return F.cross_entropy(logits_p, targets) + F.cross_entropy(logits_s, targets)

        def forward(
            self,
            f_s_clean: torch.Tensor,
            f_p_pos_clean: torch.Tensor,
            f_p_neg_clean: torch.Tensor,
            f_s_frz_pert: Optional[torch.Tensor] = None,
            f_p_frz_pert: Optional[torch.Tensor] = None,
            text_prototypes: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            l_tri = self.compute_triplet_loss(f_s_clean, f_p_pos_clean, f_p_neg_clean)
            l_ntx = self.compute_nt_xent_loss(f_s_clean, f_p_pos_clean)
            total = l_tri + l_ntx

            metrics = {
                "loss_triplet": l_tri.item(),
                "loss_ntxent": l_ntx.item(),
            }

            if text_prototypes is not None and labels is not None and self.lambda_cls > 0:
                l_cls = self.compute_classification_loss(
                    f_s_clean, f_p_pos_clean, text_prototypes, labels
                )
                total = total + self.lambda_cls * l_cls
                metrics["loss_class"] = l_cls.item()

            if (
                f_s_frz_pert is not None
                and f_p_frz_pert is not None
                and self.lambda_consist > 0
            ):
                l_con = self.compute_consistency_loss(
                    f_s_clean, f_s_frz_pert, f_p_pos_clean, f_p_frz_pert
                )
                total = total + self.lambda_consist * l_con
                metrics["loss_consist"] = l_con.item()

            metrics["loss_total"] = total.item()
            return total, metrics