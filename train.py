#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: train.py
ROLE: CLI Entrypoint for Zero-Shot Baseline, Full Training, GZS-SBIR Evaluation,
      Ablation Studies & Self-Test Verification
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

import os
import argparse
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image

from config import SBIRConfig, SKETCHY_SPLIT2_UNSEEN_21, set_reproducible_seed, TORCH_AVAILABLE
from dataset import SketchCVPreprocessor, SketchyExtendedSplitManager
from evaluate import (
    compute_category_sbir_metrics_np,
    compute_fine_grained_sbir_metrics_np,
    AdaptiveOODDetector,
)

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader
    from dataset import build_transforms, SBIRLinearDataset, SeCoLoRATripletDataset
    from models import AsymmetricHybridLoRACLIP, build_clip_text_prototypes
    from losses import SeCoLoRAMultiObjectiveLoss


def extract_features_dataloader(
    loader, encode_fn, device
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    feats_list, labels_list, inst_list, paths_list = [], [], [], []
    for imgs, labels, inst_ids, paths in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad():
            feats = encode_fn(imgs)
        feats_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())
        inst_list.extend(list(inst_ids))
        paths_list.extend(list(paths))

    return (
        np.concatenate(feats_list, axis=0),
        np.concatenate(labels_list, axis=0),
        inst_list,
        paths_list,
    )


def run_training_and_evaluation(cfg: SBIRConfig, eval_only_baseline: bool = False) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for GPU training. Run on Kaggle/Colab or install torch.")

    import clip

    set_reproducible_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"[INFO] Initializing Split Manager on: {cfg.data_root} ({cfg.split_protocol})")
    manager = SketchyExtendedSplitManager(
        root_dir=cfg.data_root,
        photo_subdir=cfg.photo_subdir,
        sketch_subdir=cfg.sketch_subdir,
        split_protocol=cfg.split_protocol,
    )

    seen_photos, seen_sketches = manager.collect_samples(manager.seen_classes, manager.seen_cls2idx)
    unseen_photos, unseen_sketches = manager.collect_samples(manager.unseen_classes, manager.unseen_cls2idx)

    print(f"[DATA] Seen ({len(manager.seen_classes)} classes): {len(seen_photos)} photos | {len(seen_sketches)} sketches")
    print(f"[DATA] Unseen ({len(manager.unseen_classes)} classes): {len(unseen_photos)} photos | {len(unseen_sketches)} sketches")

    clip_model, _ = clip.load(cfg.clip_backbone, device=device, jit=False)
    model = AsymmetricHybridLoRACLIP(
        clip_model=clip_model,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        embed_dim=cfg.embed_dim,
        bottleneck_dim=cfg.bottleneck_dim,
        res_scale=cfg.res_scale,
        unfreeze_sketch_ln=cfg.unfreeze_sketch_ln,
    ).to(device)

    print(f"[MODEL] {model.count_parameters()}")

    seen_text_proto_raw = build_clip_text_prototypes(
        clip_model, clip.tokenize, manager.seen_classes, device
    )

    tfs = build_transforms(image_size=cfg.image_size, pad_ratio=cfg.pad_ratio)

    unseen_photo_loader = DataLoader(
        SBIRLinearDataset(unseen_photos, tfs["photo_clean"]),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    unseen_sketch_loader = DataLoader(
        SBIRLinearDataset(unseen_sketches, tfs["sketch_clean"]),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model.eval()
    g_feats_0, g_lbls_0, g_inst_0, _ = extract_features_dataloader(
        unseen_photo_loader, model.encode_frozen_visual, device
    )
    q_feats_0, q_lbls_0, q_inst_0, _ = extract_features_dataloader(
        unseen_sketch_loader, model.encode_frozen_visual, device
    )
    base0_cat = compute_category_sbir_metrics_np(q_feats_0, q_lbls_0, g_feats_0, g_lbls_0)
    base0_fg = compute_fine_grained_sbir_metrics_np(q_feats_0, q_inst_0, g_feats_0, g_inst_0)
    print(f"[BASELINE 0 - FROZEN CLIP] ZS-SBIR: {base0_cat} | FG-ZS-SBIR: {base0_fg}")

    if eval_only_baseline:
        return

    train_dataset = SeCoLoRATripletDataset(seen_photos, seen_sketches, tfs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    criterion = SeCoLoRAMultiObjectiveLoss(
        margin=cfg.triplet_margin,
        temp_ntxent=cfg.temp_ntxent,
        temp_cls=cfg.temp_cls,
        temp_consist=cfg.temp_consist,
        lambda_cls=cfg.lambda_cls,
        lambda_consist=cfg.lambda_consist,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_map200 = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            s_clean = batch["sketch_clean"].to(device, non_blocking=True)
            s_pert = batch["sketch_pert"].to(device, non_blocking=True)
            p_pos_clean = batch["photo_pos_clean"].to(device, non_blocking=True)
            p_pos_pert = batch["photo_pos_pert"].to(device, non_blocking=True)
            p_neg_clean = batch["photo_neg_clean"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                f_s_clean = model.encode_sketch(s_clean)
                f_p_pos_clean = model.encode_photo(p_pos_clean)
                f_p_neg_clean = model.encode_photo(p_neg_clean)

                f_s_frz_pert = model.encode_frozen_visual(s_pert)
                f_p_frz_pert = model.encode_frozen_visual(p_pos_pert)
                text_proto = model.adapt_text_prototypes(seen_text_proto_raw)

                loss, loss_dict = criterion(
                    f_s_clean=f_s_clean,
                    f_p_pos_clean=f_p_pos_clean,
                    f_p_neg_clean=f_p_neg_clean,
                    f_s_frz_pert=f_s_frz_pert,
                    f_p_frz_pert=f_p_frz_pert,
                    text_prototypes=text_proto,
                    labels=labels,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss_dict["loss_total"]

        scheduler.step()
        avg_loss = running_loss / max(len(train_loader), 1)

        model.eval()
        g_feats, g_lbls, g_inst, _ = extract_features_dataloader(
            unseen_photo_loader, model.encode_photo, device
        )
        q_feats, q_lbls, q_inst, _ = extract_features_dataloader(
            unseen_sketch_loader, model.encode_sketch, device
        )

        zs_metrics = compute_category_sbir_metrics_np(q_feats, q_lbls, g_feats, g_lbls)
        fg_metrics = compute_fine_grained_sbir_metrics_np(q_feats, q_inst, g_feats, g_inst)

        print(
            f"[Epoch {epoch:02d}/{cfg.num_epochs:02d}] Loss: {avg_loss:.4f} | "
            f"ZS-SBIR mAP@200: {zs_metrics['mAP@200']:.2f}%, P@200: {zs_metrics['P@200']:.2f}%, "
            f"mAP@all: {zs_metrics['mAP@all']:.2f}% | FG Acc@1: {fg_metrics['Acc@1']:.2f}%"
        )

        if zs_metrics["mAP@200"] > best_map200:
            best_map200 = zs_metrics["mAP@200"]
            ckpt_path = os.path.join(cfg.save_dir, "best_seco_lora_clip.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "sketch_adapter": model.sketch_adapter.state_dict(),
                    "photo_adapter": model.photo_adapter.state_dict(),
                    "text_adapter": model.text_adapter.state_dict(),
                    "sketch_visual": model.sketch_visual.state_dict(),
                    "zs_metrics": zs_metrics,
                    "fg_metrics": fg_metrics,
                },
                ckpt_path,
            )


def run_self_verification() -> None:
    print("=" * 76)
    print("[SELF-TEST] Verifying Modular Repository (config, dataset, models, losses, evaluate)...")
    set_reproducible_seed(42)

    assert len(SKETCHY_SPLIT2_UNSEEN_21) == 21
    print("[PASS] 1. config.py: 21 Unseen Classes verified.")

    canvas = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (420, 50), (550, 180), (0, 0, 0), thickness=3)
    pil_sketch = Image.fromarray(canvas)
    pre = SketchCVPreprocessor(
        target_size=224, enable_morph=True, morph_prob=1.0, enable_random_erase=True, erase_prob=1.0
    )
    out_pil = pre(pil_sketch)
    assert out_pil.size == (224, 224)
    print(f"[PASS] 2. dataset.py: SketchCVPreprocessor verified ({out_pil.size}).")

    D = 512
    num_q, num_g = 40, 200
    q_labels = np.repeat(np.arange(4), 10)
    g_labels = np.repeat(np.arange(4), 50)
    centers = np.random.randn(4, D).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    q_feats = centers[q_labels] + 0.15 * np.random.randn(num_q, D).astype(np.float32)
    g_feats = centers[g_labels] + 0.15 * np.random.randn(num_g, D).astype(np.float32)

    cat_res = compute_category_sbir_metrics_np(q_feats, q_labels, g_feats, g_labels)
    q_inst = [f"id_{i}" for i in range(num_q)]
    g_inst = [f"id_{i}" for i in range(num_g)]
    g_feats[:num_q] = q_feats + 0.02 * np.random.randn(*q_feats.shape).astype(np.float32)
    fg_res = compute_fine_grained_sbir_metrics_np(q_feats, q_inst, g_feats, g_inst)
    print(f"[PASS] 3. evaluate.py (Retrieval): ZS={cat_res} | FG={fg_res}")

    ood_det = AdaptiveOODDetector(alpha=0.7, std_multiplier=1.5)
    id_s = ood_det.compute_scores(q_feats, g_feats, centers)
    ood_s = ood_det.compute_scores(np.random.randn(25, D).astype(np.float32), g_feats, centers)
    tau = ood_det.fit_threshold(id_s)
    ood_res = ood_det.evaluate_ood_auroc(id_s, ood_s)
    print(f"[PASS] 4. evaluate.py (OOD Detector): tau={tau:.4f}, {ood_res}")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS331 Modular SBIR Training & Evaluation CLI")
    parser.add_argument("--data-root", type=str, default="/kaggle/input/sketchy-extended")
    parser.add_argument("--photo-subdir", type=str, default="photo")
    parser.add_argument("--sketch-subdir", type=str, default="sketch")
    parser.add_argument("--split", type=str, default="split2", choices=["split1", "split2"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-consist", type=float, default=1.0)
    parser.add_argument("--eval-baseline-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test or not os.path.exists(args.data_root):
        run_self_verification()
    else:
        config = SBIRConfig(
            data_root=args.data_root,
            photo_subdir=args.photo_subdir,
            sketch_subdir=args.sketch_subdir,
            split_protocol=args.split,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lora_rank=args.lora_rank,
            lambda_cls=args.lambda_cls,
            lambda_consist=args.lambda_consist,
        )
        run_training_and_evaluation(config, eval_only_baseline=args.eval_baseline_only)