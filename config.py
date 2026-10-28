#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: config.py
ROLE: Central Configuration, Zero-Shot Split Protocols & Reproducibility Utils
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

import random
from dataclasses import dataclass
from typing import List, Tuple

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


@dataclass
class SBIRConfig:
    """Hyperparameter & System Configuration Dataclass."""
    data_root: str = "/kaggle/input/sketchy-extended"
    photo_subdir: str = "photo"
    sketch_subdir: str = "sketch"
    split_protocol: str = "split2"  # 'split2' (104/21) or 'split1' (100/25)
    image_size: int = 224

    # Sketch CV Pipeline
    pad_ratio: float = 0.08
    enable_morph: bool = True
    morph_prob: float = 0.5
    enable_random_erase: bool = True
    erase_prob: float = 0.5

    # Backbone & LoRA Architecture
    clip_backbone: str = "ViT-B/32"
    embed_dim: int = 512
    bottleneck_dim: int = 256
    lora_rank: int = 8
    lora_alpha: float = 16.0
    res_scale: float = 0.2
    unfreeze_sketch_ln: bool = True

    # Multi-Objective Loss Hyperparameters (SeCo-SBIR MM'26)
    triplet_margin: float = 0.3
    temp_ntxent: float = 0.07
    temp_cls: float = 0.07
    temp_consist: float = 0.07
    lambda_cls: float = 1.0
    lambda_consist: float = 1.0

    # Optimization
    batch_size: int = 64
    num_epochs: int = 15
    lr: float = 2e-5
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    save_dir: str = "./checkpoints"

    # Out-of-Distribution (OOD) Detection
    ood_alpha: float = 0.7
    ood_std_multiplier: float = 1.5


def set_reproducible_seed(seed: int = 42) -> None:
    """Locks all random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False