#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: visualize.py
ROLE: Academic Visualization Suite for Publication-Ready Figures:
      1. Cross-Modal t-SNE Manifold Alignment (Sketch & Photo Domains)
      2. 2D Fourier Magnitude Spectrum Analysis (High vs. Low Frequency Bias)
      3. Out-Of-Distribution (OOD) Confidence Score Distribution & Thresholding
      4. Top-K Retrieval Qualitative Grid with Ground-Truth Borders
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

import os
import argparse
from typing import List

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_tsne_manifold(
    photo_feats: np.ndarray,
    sketch_feats: np.ndarray,
    photo_labels: np.ndarray,
    sketch_labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    max_classes: int = 8,
    max_samples_per_class: int = 80,
    seed: int = 42,
) -> str:
    """
    Generates a side-by-side publication-grade t-SNE plot (Photo Domain vs. Sketch Domain)
    matching Figures 4 & 5 in SeCo-SBIR (ACM MM 2026).
    """
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(photo_labels)[:max_classes]

    p_idx_list, s_idx_list = [], []
    for c in unique_classes:
        c_p = np.where(photo_labels == c)[0]
        c_s = np.where(sketch_labels == c)[0]
        if len(c_p) > max_samples_per_class:
            c_p = rng.choice(c_p, size=max_samples_per_class, replace=False)
        if len(c_s) > max_samples_per_class:
            c_s = rng.choice(c_s, size=max_samples_per_class, replace=False)
        p_idx_list.append(c_p)
        s_idx_list.append(c_s)

    p_idx = np.concatenate(p_idx_list)
    s_idx = np.concatenate(s_idx_list)

    p_sub = photo_feats[p_idx]
    s_sub = sketch_feats[s_idx]
    p_lbl = photo_labels[p_idx]
    s_lbl = sketch_labels[s_idx]

    combined = np.concatenate([p_sub, s_sub], axis=0)
    combined = combined / (np.linalg.norm(combined, axis=1, keepdims=True) + 1e-8)

    try:
        from sklearn.manifold import TSNE
        coords = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(combined) // 6)),
            random_state=seed,
            init="pca",
            learning_rate="auto",
        ).fit_transform(combined)
    except Exception:
        centered = combined - combined.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        coords = u[:, :2] * s[:2]

    coords_p = coords[: len(p_sub)]
    coords_s = coords[len(p_sub) :]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    cmap = plt.get_cmap("tab10")

    for i, c in enumerate(unique_classes):
        cname = class_names[int(c)] if int(c) < len(class_names) else f"Class {int(c)}"
        color = cmap(i % 10)
        mask_p = p_lbl == c
        mask_s = s_lbl == c

        axes[0].scatter(
            coords_p[mask_p, 0], coords_p[mask_p, 1],
            c=[color], label=cname, s=24, alpha=0.82, edgecolors="none"
        )
        axes[1].scatter(
            coords_s[mask_s, 0], coords_s[mask_s, 1],
            c=[color], label=cname, s=24, alpha=0.82, marker="^", edgecolors="none"
        )

    axes[0].set_title("Photo Domain Embedding Space (t-SNE)", fontsize=12, fontweight="bold")
    axes[1].set_title("Sketch Domain Embedding Space (t-SNE)", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[1].legend(loc="best", frameon=True, fontsize=9, title="Unseen Classes")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_2d_fourier_spectrum(
    sketch_img_rgb: np.ndarray,
    photo_img_rgb: np.ndarray,
    save_path: str,
) -> str:
    """
    Computes and visualizes the 2D Log-Magnitude Fourier Spectrum (centered via fftshift)
    of a Sketch and its corresponding Photo, illustrating the High-Frequency (HF) vs.
    Low-Frequency (LF) spectral gap described in SketchFusion (CVPR 2025, Sec. 4).
    """
    def _compute_log_spectrum(img_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        mag = np.log1p(np.abs(f_shift))
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        return mag

    spec_sketch = _compute_log_spectrum(sketch_img_rgb)
    spec_photo = _compute_log_spectrum(photo_img_rgb)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), dpi=200)
    axes[0].imshow(photo_img_rgb)
    axes[0].set_title("Natural Photo (RGB)", fontsize=10, fontweight="bold")
    axes[1].imshow(spec_photo, cmap="viridis")
    axes[1].set_title("Photo 2D Fourier Spectrum", fontsize=10, fontweight="bold")

    axes[2].imshow(sketch_img_rgb)
    axes[2].set_title("Hand-Drawn Sketch", fontsize=10, fontweight="bold")
    axes[3].imshow(spec_sketch, cmap="viridis")
    axes[3].set_title("Sketch 2D Fourier Spectrum", fontsize=10, fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_ood_confidence_histogram(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: float,
    auroc_pct: float,
    save_path: str,
) -> str:
    """
    Plots the density distribution of Adaptive Confidence Scores for In-Distribution
    vs. Out-Of-Distribution queries along with the calibrated decision boundary tau_OOD.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)

    ax.hist(
        id_scores, bins=30, density=True, alpha=0.65,
        color="#1f77b4", label=f"In-Distribution Queries (n={len(id_scores)})"
    )
    ax.hist(
        ood_scores, bins=30, density=True, alpha=0.65,
        color="#d62728", label=f"OOD / Absent-Gallery Queries (n={len(ood_scores)})"
    )
    ax.axvline(
        threshold, color="#111111", linestyle="--", linewidth=2.0,
        label=f"Adaptive Threshold tau_OOD = {threshold:.3f}"
    )

    ax.set_title(
        f"Out-Of-Distribution (OOD) Confidence Separation (AUROC = {auroc_pct:.2f}%)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Hybrid Visual-Semantic Confidence Score S(s)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_topk_retrieval_grid(
    query_sketch_rgb: np.ndarray,
    query_class_name: str,
    retrieved_photos_rgb: List[np.ndarray],
    retrieved_class_names: List[str],
    similarity_scores: List[float],
    save_path: str,
) -> str:
    """
    Renders a qualitative retrieval row: Query Sketch on the left, followed by Top-K
    retrieved gallery photos with green borders for matches and red borders for mismatches.
    """
    k = len(retrieved_photos_rgb)
    fig, axes = plt.subplots(1, k + 1, figsize=(2.6 * (k + 1), 2.8), dpi=200)

    axes[0].imshow(query_sketch_rgb)
    axes[0].set_title(f"Query: {query_class_name}", fontsize=10, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for i in range(k):
        ax = axes[i + 1]
        ax.imshow(retrieved_photos_rgb[i])
        is_match = retrieved_class_names[i] == query_class_name
        border_color = "#2ca02c" if is_match else "#d62728"
        tag = "MATCH" if is_match else "WRONG"
        title_str = f"Rank {i+1}: {retrieved_class_names[i]} ({similarity_scores[i]:.3f} - {tag})"
        ax.set_title(title_str, fontsize=8.0, color=border_color, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_visualization_self_test(out_folder: str = "/tmp/sbir_viz_test") -> None:
    """Generates and verifies all 4 publication-grade plot types deterministically."""
    os.makedirs(out_folder, exist_ok=True)
    rng = np.random.default_rng(42)

    D = 64
    centers = rng.standard_normal((5, D)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    p_lbl = np.repeat(np.arange(5), 25)
    s_lbl = np.repeat(np.arange(5), 25)
    p_feats = centers[p_lbl] + 0.12 * rng.standard_normal((125, D)).astype(np.float32)
    s_feats = centers[s_lbl] + 0.12 * rng.standard_normal((125, D)).astype(np.float32)
    c_names = ["cabin", "cow", "dolphin", "giraffe", "windmill"]
    f1 = plot_tsne_manifold(p_feats, s_feats, p_lbl, s_lbl, c_names, os.path.join(out_folder, "tsne_test.png"))

    sketch_canvas = np.full((224, 224, 3), 255, dtype=np.uint8)
    cv2.circle(sketch_canvas, (112, 112), 60, (0, 0, 0), 3)
    photo_canvas = rng.integers(40, 220, size=(224, 224, 3), dtype=np.uint8)
    cv2.circle(photo_canvas, (112, 112), 60, (20, 80, 180), -1)
    f2 = plot_2d_fourier_spectrum(sketch_canvas, photo_canvas, os.path.join(out_folder, "fourier_test.png"))

    id_s = rng.normal(loc=0.58, scale=0.06, size=200).astype(np.float32)
    ood_s = rng.normal(loc=0.31, scale=0.07, size=200).astype(np.float32)
    f3 = plot_ood_confidence_histogram(id_s, ood_s, threshold=0.49, auroc_pct=99.4, save_path=os.path.join(out_folder, "ood_hist_test.png"))

    photos_list = [photo_canvas for _ in range(5)]
    cls_list = ["windmill", "windmill", "cabin", "windmill", "windmill"]
    sims = [0.81, 0.78, 0.71, 0.69, 0.66]
    f4 = plot_topk_retrieval_grid(sketch_canvas, "windmill", photos_list, cls_list, sims, os.path.join(out_folder, "topk_test.png"))

    for fp in [f1, f2, f3, f4]:
        assert os.path.exists(fp) and os.path.getsize(fp) > 1000, f"Failed to generate {fp}"
    print(f"[PASS] visualize.py: All 4 academic plots verified in {out_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS331 SBIR Academic Visualization Suite")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    run_visualization_self_test()