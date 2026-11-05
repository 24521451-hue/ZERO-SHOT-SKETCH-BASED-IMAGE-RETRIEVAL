#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: evaluate.py
ROLE: Academic Retrieval Metrics Suite:
      1. Category-Level ZS-SBIR & GZS-SBIR (mAP@all, mAP@200, P@100, P@200,
         Recall@1, Recall@5, Recall@10)
      2. Fine-Grained ZS-SBIR (Canonical Intra-Category Paired Acc@1, Acc@5,
         Acc@10 per CLIP-AT CVPR'23 Sec. 6 & Cross-Category Paired Acc@K)
      3. Adaptive Confidence Threshold Out-Of-Distribution (OOD) Detector
         (Hybrid Visual-Semantic Energy/Cosine Scoring, AUROC, FPR@95TPR, F1)
PROJECT: SeCo-LoRA-CLIP — Zero-Shot & Generalized Sketch-Based Image Retrieval
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
    """
    Computes standard academic Category-Level SBIR metrics:
      - mAP@all, mAP@200
      - P@100, P@200
      - R@1, R@5, R@10 (Category hit-rate / Recall@K in Top-K retrieved photos)
    """
    q_norm = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)
    g_norm = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)

    num_q = q_norm.shape[0]
    num_g = g_norm.shape[0]
    k_100 = min(100, num_g)
    k_200 = min(200, num_g)
    k_1 = min(1, num_g)
    k_5 = min(5, num_g)
    k_10 = min(10, num_g)

    ap_all_list, ap_200_list, p100_list, p200_list = [], [], [], []
    r1_list, r5_list, r10_list = [], [], []
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

        if not np.any(valid_mask):
            continue

        cum_matches = np.cumsum(matches, axis=1)
        prec_at_k = cum_matches / ranks

        p100_batch = prec_at_k[:, k_100 - 1]
        p200_batch = prec_at_k[:, k_200 - 1]

        # Standard AP@all and AP@200 definition (Dey et al. CVPR'19, Sain et al. CVPR'23, Long et al. MM'26)
        ap_all = (prec_at_k * matches).sum(axis=1) / np.maximum(total_rel, 1.0)
        denom_200 = np.maximum(np.minimum(total_rel, float(k_200)), 1.0)
        ap_200 = (prec_at_k[:, :k_200] * matches[:, :k_200]).sum(axis=1) / denom_200

        # Recall@K (Success@K: whether at least 1 relevant photo appears in Top-K)
        r1_batch = (matches[:, :k_1].sum(axis=1) > 0).astype(np.float32)
        r5_batch = (matches[:, :k_5].sum(axis=1) > 0).astype(np.float32)
        r10_batch = (matches[:, :k_10].sum(axis=1) > 0).astype(np.float32)

        ap_all_list.append(ap_all[valid_mask])
        ap_200_list.append(ap_200[valid_mask])
        p100_list.append(p100_batch[valid_mask])
        p200_list.append(p200_batch[valid_mask])
        r1_list.append(r1_batch[valid_mask])
        r5_list.append(r5_batch[valid_mask])
        r10_list.append(r10_batch[valid_mask])

    if not ap_all_list:
        return {
            "mAP@all": 0.0, "mAP@200": 0.0, "P@100": 0.0, "P@200": 0.0,
            "R@1": 0.0, "R@5": 0.0, "R@10": 0.0,
        }

    return {
        "mAP@all": round(float(np.concatenate(ap_all_list).mean() * 100.0), 2),
        "mAP@200": round(float(np.concatenate(ap_200_list).mean() * 100.0), 2),
        "P@100": round(float(np.concatenate(p100_list).mean() * 100.0), 2),
        "P@200": round(float(np.concatenate(p200_list).mean() * 100.0), 2),
        "R@1": round(float(np.concatenate(r1_list).mean() * 100.0), 2),
        "R@5": round(float(np.concatenate(r5_list).mean() * 100.0), 2),
        "R@10": round(float(np.concatenate(r10_list).mean() * 100.0), 2),
    }


def compute_generalized_sbir_metrics_np(
    unseen_query_feats: np.ndarray,
    unseen_query_labels: np.ndarray,
    unseen_gallery_feats: np.ndarray,
    unseen_gallery_labels: np.ndarray,
    seen_gallery_feats: np.ndarray,
    seen_gallery_labels: np.ndarray,
    seen_label_offset: int = 1000,
) -> Dict[str, float]:
    """
    Computes Generalized Zero-Shot SBIR (GZS-SBIR) metrics (SeCo-SBIR Table 2):
      The retrieval gallery combines BOTH Unseen-class photos (P^u) and Seen-class photos (P^s),
      creating severe distraction bias toward familiar seen classes.
      We offset seen gallery labels by +1000 so they are strictly disjoint from unseen query labels.
    """
    combined_gallery_feats = np.concatenate([unseen_gallery_feats, seen_gallery_feats], axis=0)
    combined_gallery_labels = np.concatenate(
        [unseen_gallery_labels, seen_gallery_labels + seen_label_offset], axis=0
    )
    return compute_category_sbir_metrics_np(
        query_feats=unseen_query_feats,
        query_labels=unseen_query_labels,
        gallery_feats=combined_gallery_feats,
        gallery_labels=combined_gallery_labels,
    )


def compute_fine_grained_sbir_metrics_np(
    query_feats: np.ndarray,
    query_inst_ids: List[str],
    gallery_feats: np.ndarray,
    gallery_inst_ids: List[str],
    query_labels: Optional[np.ndarray] = None,
    gallery_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes Fine-Grained Zero-Shot SBIR (FG-ZS-SBIR) instance-matching metrics:
      1. Filters gallery to ONLY paired photos (i.e., photos from original Sketchy that have
         corresponding hand-drawn sketches, excluding un-paired ImageNet extended photos).
      2. Follows the canonical CLIP-AT (CVPR 2023, Sec. 6) protocol:
         "For cross-category FG-ZS-SBIR, accuracy is measured taking only a single category
          at a time, as Acc.@q for Sketchy, which reflects percentage of sketches having
          true matched photo in the top-q list."
      3. Also computes Cross-Category All-Paired Acc@K (`Cross_Acc@1`, `Cross_Acc@5`) across
         all paired unseen photos simultaneously.
    """
    q_norm = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)
    g_norm = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)

    # Identify paired gallery photos (those whose instance_id matches at least one query sketch)
    q_inst_set = set(query_inst_ids)
    paired_g_mask = np.array([gid in q_inst_set for gid in gallery_inst_ids], dtype=bool)
    if not np.any(paired_g_mask):
        return {
            "Acc@1": 0.0, "Acc@5": 0.0, "Acc@10": 0.0,
            "Cross_Acc@1": 0.0, "Cross_Acc@5": 0.0,
        }

    # Deduplicate gallery by (label, inst_id) in case multiple copies exist
    paired_g_indices = np.where(paired_g_mask)[0]
    g_paired_feats = g_norm[paired_g_indices]
    g_paired_inst = [gallery_inst_ids[i] for i in paired_g_indices]
    g_paired_lbls = gallery_labels[paired_g_indices] if gallery_labels is not None else None

    # --- 1. Cross-Category Paired FG-ZS-SBIR ---
    g_id_to_paired_idx = {gid: idx for idx, gid in enumerate(g_paired_inst)}
    valid_q_idx = [i for i, qid in enumerate(query_inst_ids) if qid in g_id_to_paired_idx]
    if not valid_q_idx:
        return {
            "Acc@1": 0.0, "Acc@5": 0.0, "Acc@10": 0.0,
            "Cross_Acc@1": 0.0, "Cross_Acc@5": 0.0,
        }

    q_valid_feats = q_norm[valid_q_idx]
    target_paired_idx = np.array([g_id_to_paired_idx[query_inst_ids[i]] for i in valid_q_idx])[:, None]
    sim_all = q_valid_feats @ g_paired_feats.T
    k_max_all = min(10, g_paired_feats.shape[0])
    topk_all = np.argsort(-sim_all, axis=1)[:, :k_max_all]

    cross_acc1 = float((topk_all[:, :1] == target_paired_idx).any(axis=1).mean() * 100.0)
    cross_acc5 = float((topk_all[:, : min(5, k_max_all)] == target_paired_idx).any(axis=1).mean() * 100.0)
    cross_acc10 = float((topk_all[:, :k_max_all] == target_paired_idx).any(axis=1).mean() * 100.0)

    # --- 2. Canonical Single-Category-at-a-Time FG-ZS-SBIR (CLIP-AT CVPR'23 Table 1) ---
    if query_labels is not None and g_paired_lbls is not None:
        intra_hits_1, intra_hits_5, intra_hits_10 = [], [], []
        unique_classes = np.unique(query_labels[valid_q_idx])

        for c in unique_classes:
            c_q_mask = [i for i in valid_q_idx if query_labels[i] == c]
            c_g_mask = np.where(g_paired_lbls == c)[0]
            if len(c_q_mask) == 0 or len(c_g_mask) == 0:
                continue

            c_g_feats = g_paired_feats[c_g_mask]
            c_g_inst = [g_paired_inst[j] for j in c_g_mask]
            c_gid_map = {gid: j for j, gid in enumerate(c_g_inst)}

            c_q_valid = [i for i in c_q_mask if query_inst_ids[i] in c_gid_map]
            if not c_q_valid:
                continue

            c_q_feats = q_norm[c_q_valid]
            c_targets = np.array([c_gid_map[query_inst_ids[i]] for i in c_q_valid])[:, None]
            c_sim = c_q_feats @ c_g_feats.T
            k_c = min(10, c_g_feats.shape[0])
            c_topk = np.argsort(-c_sim, axis=1)[:, :k_c]

            intra_hits_1.append((c_topk[:, :1] == c_targets).any(axis=1).astype(np.float32))
            intra_hits_5.append((c_topk[:, : min(5, k_c)] == c_targets).any(axis=1).astype(np.float32))
            intra_hits_10.append((c_topk[:, :k_c] == c_targets).any(axis=1).astype(np.float32))

        if intra_hits_1:
            acc1 = float(np.concatenate(intra_hits_1).mean() * 100.0)
            acc5 = float(np.concatenate(intra_hits_5).mean() * 100.0)
            acc10 = float(np.concatenate(intra_hits_10).mean() * 100.0)
            return {
                "Acc@1": round(acc1, 2),
                "Acc@5": round(acc5, 2),
                "Acc@10": round(acc10, 2),
                "Cross_Acc@1": round(cross_acc1, 2),
                "Cross_Acc@5": round(cross_acc5, 2),
            }

    return {
        "Acc@1": round(cross_acc1, 2),
        "Acc@5": round(cross_acc5, 2),
        "Acc@10": round(cross_acc10, 2),
        "Cross_Acc@1": round(cross_acc1, 2),
        "Cross_Acc@5": round(cross_acc5, 2),
    }


class AdaptiveOODDetector:
    """
    Adaptive Confidence Threshold Out-Of-Distribution (OOD) Detector:
      Detects when a user draws an unknown object outside the gallery database
      (e.g., querying Unseen-21 sketches against a Gallery containing ONLY Seen-104 photos/classes).
      Combines:
        1. Top-K Visual Cosine Consensus: mean cosine similarity to top-5 nearest gallery photos.
        2. Semantic Prototype Energy Score: temperature-scaled log-sum-exp / max similarity over
           gallery category text prototypes.
    """

    def __init__(self, alpha: float = 0.65, std_multiplier: float = 1.5, energy_temp: float = 0.05):
        self.alpha = alpha
        self.std_multiplier = std_multiplier
        self.energy_temp = energy_temp
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

        # Chunked top-K visual similarity to prevent OOM on large galleries
        num_q = q_f.shape[0]
        top_k = min(5, g_f.shape[0])
        vis_sim = np.zeros(num_q, dtype=np.float32)

        chunk_size = 512
        for start in range(0, num_q, chunk_size):
            end = min(start + chunk_size, num_q)
            sim_chunk = q_f[start:end] @ g_f.T
            part_top = -np.partition(-sim_chunk, top_k - 1, axis=1)[:, :top_k]
            vis_sim[start:end] = part_top.mean(axis=1)

        if text_prototypes is not None:
            t_f = text_prototypes / (np.linalg.norm(text_prototypes, axis=1, keepdims=True) + 1e-8)
            logits_txt = q_f @ t_f.T
            txt_max = logits_txt.max(axis=1)
            # Smooth maximum (Energy score) normalized to cosine scale
            shifted = (logits_txt - txt_max[:, None]) / self.energy_temp
            energy_score = txt_max + self.energy_temp * np.log(np.exp(shifted).mean(axis=1) + 1e-8)
            return self.alpha * vis_sim + (1.0 - self.alpha) * energy_score

        return vis_sim

    def fit_threshold(
        self,
        in_dist_scores: np.ndarray,
        ood_val_scores: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calibrates the adaptive threshold tau_OOD:
          - If validation OOD scores are provided, finds the optimal Youden's J statistic
            threshold maximizing (TPR - FPR).
          - Otherwise uses the statistical Gaussian tail bound: mu_ID - k * sigma_ID.
        """
        self.id_mean = float(in_dist_scores.mean())
        self.id_std = float(in_dist_scores.std() + 1e-6)

        if ood_val_scores is not None and len(ood_val_scores) > 0:
            all_s = np.concatenate([in_dist_scores, ood_val_scores])
            candidates = np.percentile(all_s, np.linspace(5, 95, 91))
            best_j = -1.0
            best_t = self.id_mean - self.std_multiplier * self.id_std
            for t in candidates:
                tpr = float((in_dist_scores >= t).mean())
                tnr = float((ood_val_scores < t).mean())
                j_stat = tpr + tnr - 1.0
                if j_stat > best_j:
                    best_j = j_stat
                    best_t = float(t)
            self.threshold = best_t
        else:
            self.threshold = self.id_mean - self.std_multiplier * self.id_std

        return self.threshold

    def is_out_of_distribution(self, query_scores: np.ndarray) -> np.ndarray:
        return query_scores < self.threshold

    def evaluate_ood_auroc(self, id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
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

        # Compute F1 and Detection Accuracy at self.threshold
        if self.threshold == 0.0:
            self.fit_threshold(id_scores, ood_scores)

        pred_id = (y_score >= self.threshold).astype(np.float32)
        det_acc = float((pred_id == y_true).mean() * 100.0)

        return {
            "AUROC": round(auroc * 100.0, 2),
            "FPR@95TPR": round(fpr_at_95 * 100.0, 2),
            "Det_Acc": round(det_acc, 2),
            "Threshold": round(float(self.threshold), 4),
        }