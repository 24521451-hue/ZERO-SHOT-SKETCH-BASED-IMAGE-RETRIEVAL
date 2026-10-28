#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: evaluate.py
ROLE: Academic Retrieval Metrics (ZS-SBIR, GZS-SBIR, FG-ZS-SBIR) &
      Adaptive Confidence Threshold Out-Of-Distribution (OOD) Detector
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

from typing import List, Dict, Optional
import numpy as np


def compute_category_sbir_metrics_np(
    query_feats: np.ndarray,
    query_labels: np.ndarray,
    gallery_feats: np.ndarray,
    gallery_labels: np.ndarray,
    chunk_size: int = 512,
) -> Dict[str, float]:
    q_norm = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)
    g_norm = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)

    num_q = q_norm.shape[0]
    num_g = g_norm.shape[0]
    k_100 = min(100, num_g)
    k_200 = min(200, num_g)

    ap_all_list, ap_200_list, p100_list, p200_list = [], [], [], []
    ranks = np.arange(1, num_g + 1, dtype=np.float32)[None, :]

    for start in range(0, num_q, chunk_size):
        end = min(start + chunk_size, num_q)
        sim_matrix = q_norm[start:end] @ g_norm.T
        sorted_indices = np.argsort(-sim_matrix, axis=1)
        sorted_g_labels = gallery_labels[sorted_indices]

        q_lbl = query_labels[start:end, None]
        matches = (sorted_g_labels == q_lbl).astype(np.float32)
        total_rel = matches.sum(axis=1)
        valid_mask = total_rel > 0

        cum_matches = np.cumsum(matches, axis=1)
        prec_at_k = cum_matches / ranks

        p100_batch = prec_at_k[:, k_100 - 1]
        p200_batch = prec_at_k[:, k_200 - 1]

        ap_all = (prec_at_k * matches).sum(axis=1) / np.maximum(total_rel, 1.0)
        denom_200 = np.maximum(np.minimum(total_rel, float(k_200)), 1.0)
        ap_200 = (prec_at_k[:, :k_200] * matches[:, :k_200]).sum(axis=1) / denom_200

        ap_all_list.append(ap_all[valid_mask])
        ap_200_list.append(ap_200[valid_mask])
        p100_list.append(p100_batch[valid_mask])
        p200_list.append(p200_batch[valid_mask])

    return {
        "mAP@all": float(np.concatenate(ap_all_list).mean() * 100.0),
        "mAP@200": float(np.concatenate(ap_200_list).mean() * 100.0),
        "P@100": float(np.concatenate(p100_list).mean() * 100.0),
        "P@200": float(np.concatenate(p200_list).mean() * 100.0),
    }


def compute_fine_grained_sbir_metrics_np(
    query_feats: np.ndarray,
    query_inst_ids: List[str],
    gallery_feats: np.ndarray,
    gallery_inst_ids: List[str],
) -> Dict[str, float]:
    g_id_to_idx = {gid: i for i, gid in enumerate(gallery_inst_ids)}
    valid_q_indices = [i for i, qid in enumerate(query_inst_ids) if qid in g_id_to_idx]

    if not valid_q_indices:
        return {"Acc@1": 0.0, "Acc@5": 0.0, "Acc@10": 0.0}

    q_f = query_feats[valid_q_indices]
    q_f = q_f / (np.linalg.norm(q_f, axis=1, keepdims=True) + 1e-8)
    g_f = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)

    target_g_idx = np.array([g_id_to_idx[query_inst_ids[i]] for i in valid_q_indices])[:, None]
    sim = q_f @ g_f.T
    topk_indices = np.argsort(-sim, axis=1)[:, : min(10, g_f.shape[0])]

    acc1 = float((topk_indices[:, :1] == target_g_idx).any(axis=1).mean() * 100.0)
    acc5 = float((topk_indices[:, :5] == target_g_idx).any(axis=1).mean() * 100.0)
    acc10 = float((topk_indices[:, :10] == target_g_idx).any(axis=1).mean() * 100.0)

    return {"Acc@1": acc1, "Acc@5": acc5, "Acc@10": acc10}


class AdaptiveOODDetector:
    def __init__(self, alpha: float = 0.7, std_multiplier: float = 1.5):
        self.alpha = alpha
        self.std_multiplier = std_multiplier
        self.threshold: float = 0.0
        self.id_mean: float = 0.0
        self.id_std: float = 1.0

    def compute_scores(
        self,
        query_feats: np.ndarray,
        gallery_feats: np.ndarray,
        text_prototypes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q_f = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)
        g_f = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)
        sim = q_f @ g_f.T
        top_k = min(5, g_f.shape[0])
        vis_sim = -np.sort(-sim, axis=1)[:, :top_k].mean(axis=1)

        if text_prototypes is not None:
            t_f = text_prototypes / (np.linalg.norm(text_prototypes, axis=1, keepdims=True) + 1e-8)
            txt_sim = (q_f @ t_f.T).max(axis=1)
            return self.alpha * vis_sim + (1.0 - self.alpha) * txt_sim
        return vis_sim

    def fit_threshold(self, in_dist_scores: np.ndarray) -> float:
        self.id_mean = float(in_dist_scores.mean())
        self.id_std = float(in_dist_scores.std() + 1e-6)
        self.threshold = self.id_mean - self.std_multiplier * self.id_std
        return self.threshold

    def is_out_of_distribution(self, query_scores: np.ndarray) -> np.ndarray:
        return query_scores < self.threshold

    @staticmethod
    def evaluate_ood_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
        y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        y_score = np.concatenate([id_scores, ood_scores])

        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]

        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1.0 - y_true_sorted)
        tpr = tps / max(float(tps[-1]), 1.0)
        fpr = fps / max(float(fps[-1]), 1.0)

        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auroc = float(trapz_fn(tpr, fpr))
        idx_95 = np.searchsorted(tpr, 0.95)
        fpr_at_95 = float(fpr[min(idx_95, len(fpr) - 1)])

        return {"AUROC": auroc * 100.0, "FPR@95TPR": fpr_at_95 * 100.0}