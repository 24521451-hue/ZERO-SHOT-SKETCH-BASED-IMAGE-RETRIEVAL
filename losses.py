#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: losses.py
ROLE: Multi-Objective Loss Function Suite:
      1. Cross-Modal Margin Ranking Triplet Loss (Category + Hard Intra-Class)
      2. Class-Aware Bidirectional NT-Xent Contrastive Loss (False-Negative Free)
      3. Semantic Class Prototype Alignment Loss (Co-Adapted CLIP Text Space)
      4. Perturbation-Based Asymmetric InfoNCE Consistency Loss (SeCo-SBIR MM'26)
      5. Relative Distance Uniformity Regularizer via Symmetric KL / f-Divergence
         (CLIP-AT CVPR'23 Eq. 4 for stabilizing cross-category FG-ZS-SBIR margin)
PROJECT: SeCo-LoRA-CLIP — Zero-Shot & Generalized Sketch-Based Image Retrieval
================================================================================
"""

from typing import Dict, Tuple, Optional
from config import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SeCoLoRAMultiObjectiveLoss(nn.Module):
        """
        Unified Multi-Objective Loss for SeCo-LoRA-CLIP:
          L_total = L_triplet + lambda_ntxent * L_NT-Xent
                    + lambda_cls * L_class + lambda_consist * L_consist
                    + lambda_fg_kl * L_delta
        """

        def __init__(
            self,
            margin: float = 0.3,
            use_ntxent: bool = True,
            class_aware_ntxent: bool = True,
            temp_ntxent: float = 0.07,
            temp_cls: float = 0.07,
            temp_consist: float = 0.07,
            lambda_ntxent: float = 1.0,
            lambda_cls: float = 1.0,
            lambda_consist: float = 1.0,
            lambda_fg_kl: float = 0.1,
        ):
            super().__init__()
            self.margin = margin
            self.use_ntxent = use_ntxent
            self.class_aware_ntxent = class_aware_ntxent
            self.temp_ntxent = temp_ntxent
            self.temp_cls = temp_cls
            self.temp_consist = temp_consist
            self.lambda_ntxent = lambda_ntxent
            self.lambda_cls = lambda_cls
            self.lambda_consist = lambda_consist
            self.lambda_fg_kl = lambda_fg_kl

        @staticmethod
        def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Computes cosine distance d(a, b) = 1 - cos(a, b) for L2-normalized vectors."""
            return 1.0 - (a * b).sum(dim=-1)

        def compute_triplet_loss(
            self, f_s: torch.Tensor, f_p_pos: torch.Tensor, f_p_neg: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Computes margin triplet loss and returns both the scalar loss and the
            per-sample relative distance vector delta(s, p+, p-) = d(s, p+) - d(s, p-).
            """
            d_pos = self.cosine_distance(f_s, f_p_pos)
            d_neg = self.cosine_distance(f_s, f_p_neg)
            delta = d_pos - d_neg
            loss = F.relu(self.margin + delta).mean()
            return loss, delta

        def compute_nt_xent_loss(
            self,
            f_s: torch.Tensor,
            f_p_pos: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Bidirectional Normalized Temperature-Scaled Cross-Entropy (NT-Xent) with
            Class-Aware False-Negative Masking:
              When multiple samples in a mini-batch belong to the same category (y_i == y_j, i != j),
              standard InfoNCE erroneously treats them as negative pairs and pushes same-class
              embeddings apart. We mask out off-diagonal same-class pairs from the denominator
              so only true cross-class negatives are repelled.
            """
            bsz = f_s.size(0)
            logits_s2p = (f_s @ f_p_pos.t()) / self.temp_ntxent
            logits_p2s = logits_s2p.t()
            targets = torch.arange(bsz, device=f_s.device)

            if self.class_aware_ntxent and labels is not None:
                same_class = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
                diag_mask = torch.eye(bsz, dtype=torch.bool, device=f_s.device)
                false_neg_mask = same_class & (~diag_mask)

                logits_s2p = logits_s2p.masked_fill(false_neg_mask, -1e4)
                logits_p2s = logits_p2s.masked_fill(false_neg_mask, -1e4)

            loss_s2p = F.cross_entropy(logits_s2p, targets)
            loss_p2s = F.cross_entropy(logits_p2s, targets)
            return 0.5 * (loss_s2p + loss_p2s)

        def compute_classification_loss(
            self,
            f_s: torch.Tensor,
            f_p_pos: torch.Tensor,
            text_prototypes: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
            """
            Cross-modal semantic prototype classification loss (CLIP-AT Eq. 3 & SeCo-SBIR Eq. 16).
            Aligns both sketch and photo embeddings with category-level text prototypes.
            """
            logits_s = (f_s @ text_prototypes.t()) / self.temp_cls
            logits_p = (f_p_pos @ text_prototypes.t()) / self.temp_cls
            return 0.5 * (F.cross_entropy(logits_s, labels) + F.cross_entropy(logits_p, labels))

        def compute_consistency_loss(
            self,
            f_s_clean: torch.Tensor,
            f_s_frz_pert: torch.Tensor,
            f_p_clean: torch.Tensor,
            f_p_frz_pert: torch.Tensor,
        ) -> torch.Tensor:
            """
            Perturbation-Based Asymmetric InfoNCE Consistency Constraint (SeCo-SBIR Eq. 10-11).
            Anchors clean adapted representations (f_s_clean, f_p_clean) to the structural
            relational geometry of perturbed frozen CLIP representations (f_s_frz_pert, f_p_frz_pert).
            """
            bsz = f_s_clean.size(0)
            targets = torch.arange(bsz, device=f_s_clean.device)

            logits_p = (f_p_frz_pert @ f_p_clean.t()) / self.temp_consist
            logits_s = (f_s_frz_pert @ f_s_clean.t()) / self.temp_consist

            loss_p = F.cross_entropy(logits_p, targets)
            loss_s = F.cross_entropy(logits_s, targets)
            return 0.5 * (loss_p + loss_s)

        def compute_relative_distance_f_divergence(
            self, delta: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            """
            Stabilizes the Triplet Margin across diverse categories via Sgarro's average
            KL-divergence (f-Divergence, CLIP-AT CVPR'23 Eq. 4):
              For each category c in the mini-batch with K >= 2 triplets, forms the softmax
              distribution D_c = softmax(delta_c) of length K and minimizes pairwise KL(D_i || D_j).
            """
            unique_classes, counts = torch.unique(labels, return_counts=True)
            min_k = int(counts.min().item()) if len(counts) > 0 else 0
            if len(unique_classes) < 2 or min_k < 2:
                return delta.new_tensor(0.0)

            dist_list = []
            for c in unique_classes:
                c_deltas = delta[labels == c][:min_k]
                dist_c = F.softmax(c_deltas / 0.1, dim=0)
                dist_list.append(dist_c)

            dists = torch.stack(dist_list, dim=0)  # (C_b, K)
            log_dists = torch.log(dists + 1e-8)

            # Pairwise KL(D_i || D_j) = sum_k D_i(k) * (log D_i(k) - log D_j(k))
            # Using broadcasting over (C_b, 1, K) and (1, C_b, K)
            kl_matrix = (dists.unsqueeze(1) * (log_dists.unsqueeze(1) - log_dists.unsqueeze(0))).sum(dim=-1)
            num_c = len(unique_classes)
            off_diag = ~torch.eye(num_c, dtype=torch.bool, device=delta.device)
            return kl_matrix[off_diag].mean()

        def forward(
            self,
            f_s_clean: torch.Tensor,
            f_p_pos_clean: torch.Tensor,
            f_p_neg_clean: torch.Tensor,
            f_s_frz_pert: Optional[torch.Tensor] = None,
            f_p_frz_pert: Optional[torch.Tensor] = None,
            text_prototypes: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            f_p_hard_neg: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            l_tri, delta = self.compute_triplet_loss(f_s_clean, f_p_pos_clean, f_p_neg_clean)
            total = l_tri

            metrics: Dict[str, float] = {
                "loss_triplet": float(l_tri.item()),
                "loss_ntxent": 0.0,
                "loss_class": 0.0,
                "loss_consist": 0.0,
                "loss_fg_kl": 0.0,
            }

            # Optional intra-category hard-triplet loss for fine-grained structural discrimination
            if f_p_hard_neg is not None:
                l_hard_tri, delta_hard = self.compute_triplet_loss(f_s_clean, f_p_pos_clean, f_p_hard_neg)
                total = total + 0.5 * l_hard_tri
                delta = 0.5 * (delta + delta_hard)
                metrics["loss_triplet"] = float((l_tri + 0.5 * l_hard_tri).item())

            if self.use_ntxent and self.lambda_ntxent > 0.0:
                l_ntx = self.compute_nt_xent_loss(f_s_clean, f_p_pos_clean, labels=labels)
                total = total + self.lambda_ntxent * l_ntx
                metrics["loss_ntxent"] = float(l_ntx.item())

            if text_prototypes is not None and labels is not None and self.lambda_cls > 0.0:
                l_cls = self.compute_classification_loss(
                    f_s_clean, f_p_pos_clean, text_prototypes, labels
                )
                total = total + self.lambda_cls * l_cls
                metrics["loss_class"] = float(l_cls.item())

            if (
                f_s_frz_pert is not None
                and f_p_frz_pert is not None
                and self.lambda_consist > 0.0
            ):
                l_con = self.compute_consistency_loss(
                    f_s_clean, f_s_frz_pert, f_p_pos_clean, f_p_frz_pert
                )
                total = total + self.lambda_consist * l_con
                metrics["loss_consist"] = float(l_con.item())

            if labels is not None and self.lambda_fg_kl > 0.0:
                l_kl = self.compute_relative_distance_f_divergence(delta, labels)
                total = total + self.lambda_fg_kl * l_kl
                metrics["loss_fg_kl"] = float(l_kl.item())

            metrics["loss_total"] = float(total.item())
            return total, metrics