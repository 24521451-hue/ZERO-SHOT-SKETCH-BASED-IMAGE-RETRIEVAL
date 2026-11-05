#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: config.py
ROLE: Central Configuration, Zero-Shot Split Protocols & Reproducibility Utils
PROJECT: SeCo-LoRA-CLIP — Zero-Shot & Generalized Sketch-Based Image Retrieval
================================================================================
"""

import os
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Canonical 21 Unseen Categories absent from ImageNet-1K (Yelamarthi et al., ECCV 2018;
# standard benchmark in CLIP-AT CVPR'23, SketchFusion CVPR'25, and SeCo-SBIR MM'26)
SKETCHY_SPLIT2_UNSEEN_21: List[str] = [
    "cabin", "cow", "dolphin", "door", "giraffe",
    "helicopter", "mouse", "pear", "raccoon", "rhinoceros",
    "saw", "scissors", "seagull", "skyscraper", "songbird",
    "sword", "tree", "wading_bird", "wheelchair", "windmill",
    "window"
]

# Secondary reference: Split 1 (Shen et al., CVPR 2018 - 100 Seen / 25 Unseen)
SKETCHY_SPLIT1_UNSEEN_25: List[str] = [
    "airplane", "alarm_clock", "ant", "ape", "apple",
    "armor", "axe", "banana", "bat", "bear",
    "bee", "beetle", "bell", "bench", "bicycle",
    "blimp", "bread", "butterfly", "cabin", "camel",
    "candle", "cannon", "car_(sedan)", "castle", "cat"
]

# Standard OpenAI CLIP Normalization Constants
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

# Multi-template semantic prompt ensemble for robust class prototype extraction
CLIP_PROMPT_TEMPLATES: Tuple[str, ...] = (
    "a photo of a {}.",
    "a natural photograph of a {}.",
    "a close-up photo of a {}.",
    "a clear photo of the {}.",
    "a rendition of a {}.",
    "a sketch of a {}.",
    "a line drawing of a {}.",
)


@dataclass
class SBIRConfig:
    """Hyperparameter & System Configuration Dataclass for SeCo-LoRA-CLIP."""
    data_root: str = "/kaggle/input/sketchy-extended"
    photo_subdir: str = "photo"
    sketch_subdir: str = "sketch"
    split_protocol: str = "split2"  # 'split2' (104/21) or 'split1' (100/25)
    image_size: int = 224

    # Sketch CV Pipeline
    use_tight_crop: bool = True
    pad_ratio: float = 0.08
    enable_morph: bool = True
    morph_prob: float = 0.5
    enable_random_erase: bool = True
    erase_prob: float = 0.5

    # Backbone & Asymmetric LoRA-CLIP Architecture
    clip_backbone: str = "ViT-B/32"
    embed_dim: int = 512
    bottleneck_dim: int = 256
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_qkv: bool = True       # Inject LoRA into Q/K/V projections as well as Out-Proj
    res_scale: float = 0.2
    unfreeze_sketch_ln: bool = True    # Fine-tune LayerNorm in Sketch ViT tower (CLIP-AT CVPR'23)
    freeze_photo_adapter: bool = False # If True, strictly 100% frozen photo tower (no adapter)
    use_prompt_ensemble: bool = True   # Average multi-template CLIP text embeddings

    # Multi-Objective Loss Hyperparameters (SeCo-SBIR MM'26 + Class-Aware Masking)
    triplet_margin: float = 0.3
    use_ntxent: bool = True
    class_aware_ntxent: bool = True    # Mask false negatives of the same class in mini-batch
    temp_ntxent: float = 0.07
    temp_cls: float = 0.07
    temp_consist: float = 0.07
    lambda_ntxent: float = 1.0
    lambda_cls: float = 1.0
    lambda_consist: float = 1.0
    lambda_fg_kl: float = 0.1          # Uniform relative distance f-divergence regularizer (CLIP-AT Eq. 4)

    # Optimization & High-Speed Feature Caching (Kaggle T4/P100 Acceleration)
    batch_size: int = 64
    classes_per_batch: int = 16        # P classes per batch for balanced P-K sampling
    samples_per_class: int = 4         # K sketches per class in each batch (P * K = 64)
    use_pk_sampler: bool = True
    use_feature_cache: bool = True     # Pre-extract & cache frozen CLIP photo features (3x faster)
    num_epochs: int = 15
    lr: float = 1e-4                   # Optimal AdamW LR for LoRA + Residual Adapters
    ln_lr: float = 2e-5                # Lower LR for LayerNorm parameters
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip_norm: float = 1.0
    num_workers: int = 4
    seed: int = 42
    save_dir: str = "./outputs"

    # Out-of-Distribution (OOD) Detection
    ood_alpha: float = 0.65            # Weight between top-K visual similarity and text prototype similarity
    ood_std_multiplier: float = 1.5
    ood_energy_temp: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def set_reproducible_seed(seed: int = 42) -> None:
    """Locks all random seeds across Python, NumPy, and PyTorch for deterministic results."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
