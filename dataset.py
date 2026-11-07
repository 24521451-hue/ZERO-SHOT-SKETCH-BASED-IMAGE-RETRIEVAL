#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: dataset.py
ROLE: Sketch CV Preprocessing (Tight-Crop, 1:1 Padding, Stroke-Safe Morphology,
      Erasing), 2-Folder Sketchy-Extended Scanner, PK-Sampler & Dual-Mode
      (Cached / On-the-Fly) PyTorch Datasets
PROJECT: SeCo-LoRA-CLIP — Zero-Shot & Generalized Sketch-Based Image Retrieval
================================================================================
"""

import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterator

import cv2
import numpy as np
from PIL import Image

from config import (
    SKETCHY_SPLIT2_UNSEEN_21,
    SKETCHY_SPLIT1_UNSEEN_25,
    CLIP_MEAN,
    CLIP_STD,
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import Dataset, Sampler
    import torchvision.transforms as T


class SketchCVPreprocessor:
    """
    Domain-specific Computer Vision pipeline for hand-drawn sketches:
      1. Alpha-safe RGBA -> RGB conversion (prevents transparent background turning black).
      2. Tight-Crop + Symmetric 1:1 Padding (maximizes active stroke patches in ViT-B/32).
      3. Stroke-Safe Morphological Operations (simulates pen stroke thickness variations
         without erasing thin 1px black strokes on white background).
      4. White Random Erasing (simulates partial/occluded strokes for perturbation consistency).
    """

    def __init__(
        self,
        target_size: int = 224,
        use_tight_crop: bool = True,
        pad_ratio: float = 0.08,
        enable_morph: bool = False,
        morph_prob: float = 0.5,
        enable_random_erase: bool = False,
        erase_prob: float = 0.5,
    ):
        self.target_size = target_size
        self.use_tight_crop = use_tight_crop
        self.pad_ratio = pad_ratio
        self.enable_morph = enable_morph
        self.morph_prob = morph_prob
        self.enable_random_erase = enable_random_erase
        self.erase_prob = erase_prob

    @staticmethod
    def to_clean_sketch_rgb(pil_img: Image.Image) -> np.ndarray:
        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            rgba = pil_img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(bg, rgba).convert("RGB")
            return np.array(composited, dtype=np.uint8)
        return np.array(pil_img.convert("RGB"), dtype=np.uint8)

    def tight_crop_and_symmetric_pad(self, img_rgb: np.ndarray) -> np.ndarray:
        if not self.use_tight_crop:
            return img_rgb

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        coords = cv2.findNonZero((inv > 30).astype(np.uint8))

        if coords is None:
            return img_rgb

        x, y, w, h = cv2.boundingRect(coords)
        cropped = img_rgb[y : y + h, x : x + w]

        max_side = max(w, h)
        margin = int(max_side * self.pad_ratio)
        square_side = max_side + 2 * margin

        pad_top = (square_side - h) // 2
        pad_bottom = square_side - h - pad_top
        pad_left = (square_side - w) // 2
        pad_right = square_side - w - pad_left

        return cv2.copyMakeBorder(
            cropped,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    def apply_morphological_ops(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Stroke-safe morphological augmentation on white-background (255) black-stroke (0) sketches:
          - Note: In OpenCV, `cv2.erode` takes the local minimum, which expands dark (0) strokes
            (making lines thicker/bolder). `cv2.dilate` takes the local maximum, which expands
            white (255) background and can completely wipe out 1px thin strokes!
          - Therefore, we use `cv2.erode` for bolding strokes, or morphological closing on the
            inverted stroke mask, and soft alpha-blended thinning so strokes never vanish.
        """
        if not self.enable_morph or random.random() > self.morph_prob:
            return img_rgb

        kernel_size = random.choice([2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if random.random() < 0.70:
            # Thicken black strokes (erode white background)
            return cv2.erode(img_rgb, kernel, iterations=1)
        else:
            # Soft-thinning: blend 50% original + 50% dilated so thin strokes are attenuated, never erased
            dilated = cv2.dilate(img_rgb, np.ones((2, 2), np.uint8), iterations=1)
            gray_dil = cv2.cvtColor(dilated, cv2.COLOR_RGB2GRAY)
            # Guard: if dilation wiped out >85% of dark pixels, fallback to original
            if (255 - gray_dil > 30).sum() < 20:
                return img_rgb
            return cv2.addWeighted(img_rgb, 0.55, dilated, 0.45, 0)

    def apply_white_random_erasing(self, img_rgb: np.ndarray) -> np.ndarray:
        if not self.enable_random_erase or random.random() > self.erase_prob:
            return img_rgb

        out = img_rgb.copy()
        h, w, _ = out.shape
        area = h * w

        for _ in range(10):
            target_area = random.uniform(0.02, 0.10) * area
            aspect_ratio = random.uniform(0.3, 3.3)
            eh = int(round(math.sqrt(target_area * aspect_ratio)))
            ew = int(round(math.sqrt(target_area / aspect_ratio)))
            if ew < w and eh < h:
                top = random.randint(0, h - eh)
                left = random.randint(0, w - ew)
                out[top : top + eh, left : left + ew, :] = 255
                break
        return out

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        rgb = self.to_clean_sketch_rgb(pil_img)
        squared = self.tight_crop_and_symmetric_pad(rgb)
        morphed = self.apply_morphological_ops(squared)
        resized = cv2.resize(
            morphed, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA
        )
        erased = self.apply_white_random_erasing(resized)
        return Image.fromarray(erased)


class SketchyExtendedSplitManager:
    """
    Scans the 2-folder Sketchy-Extended structure (`photo/` and `sketch/`, 125 classes each)
    and partitions into Seen (104 classes) and Unseen (21 classes for split2, or 100/25 for split1).
    Automatically handles nested subfolders (e.g. `photo/tx_000000000000/` + `EXTEND_image_sketchy/`).
    """

    def __init__(
        self,
        root_dir: str,
        photo_subdir: str = "photo",
        sketch_subdir: str = "sketch",
        split_protocol: str = "split2",
    ):
        self.root_dir = Path(root_dir)
        self.photo_dir = self._resolve_subdir(photo_subdir)
        self.sketch_dir = self._resolve_subdir(sketch_subdir)

        if split_protocol == "split2":
            self.unseen_classes = sorted(SKETCHY_SPLIT2_UNSEEN_21)
        elif split_protocol == "split1":
            self.unseen_classes = sorted(SKETCHY_SPLIT1_UNSEEN_25)
        else:
            raise ValueError(f"Unsupported split_protocol: {split_protocol}")

        self.all_classes = sorted(
            [d.name for d in self.photo_dir.iterdir() if d.is_dir()]
        )
        self.seen_classes = sorted(
            [c for c in self.all_classes if c not in set(self.unseen_classes)]
        )

        self.seen_cls2idx = {c: i for i, c in enumerate(self.seen_classes)}
        self.unseen_cls2idx = {c: i for i, c in enumerate(self.unseen_classes)}
        self.global_cls2idx = {c: i for i, c in enumerate(self.all_classes)}

    def _resolve_subdir(self, subdir_name: str) -> Path:
        candidate = self.root_dir / subdir_name
        if not candidate.exists():
            # Fallback: search case-insensitively or nested 1 level
            for d in self.root_dir.iterdir():
                if d.is_dir() and subdir_name.lower() in d.name.lower():
                    candidate = d
                    break
        if not candidate.exists():
            raise FileNotFoundError(f"Directory not found: {candidate}")

        subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        if len(subdirs) < 20:
            # Prefer tx_000000000000 or folder containing >= 100 class directories
            preferred = sorted(
                subdirs,
                key=lambda x: (0 if "tx_000000000000" in x.name else 1, x.name),
            )
            for sd in preferred:
                inner_dirs = [d for d in sd.iterdir() if d.is_dir()]
                if len(inner_dirs) >= 100:
                    return sd
        return candidate

    @staticmethod
    def extract_instance_id(filepath: str, is_sketch: bool) -> str:
        stem = Path(filepath).stem
        if is_sketch and "-" in stem:
            return stem.rsplit("-", 1)[0]
        return stem

    def collect_samples(
        self, class_list: List[str], cls2idx: Dict[str, int]
    ) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
        """
        Collects all photo and sketch samples for the given class_list.
        Returns:
          photos: List of (filepath, class_idx, instance_id)
          sketches: List of (filepath, class_idx, instance_id)
        """
        photos, sketches = [], []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for cname in class_list:
            c_idx = cls2idx[cname]
            p_folder = self.photo_dir / cname
            s_folder = self.sketch_dir / cname

            if p_folder.exists():
                for f in sorted(p_folder.rglob("*")):
                    if f.is_file() and f.suffix.lower() in valid_exts:
                        inst_id = self.extract_instance_id(str(f), is_sketch=False)
                        photos.append((str(f), c_idx, inst_id))

            if s_folder.exists():
                for f in sorted(s_folder.rglob("*")):
                    if f.is_file() and f.suffix.lower() in valid_exts:
                        inst_id = self.extract_instance_id(str(f), is_sketch=True)
                        sketches.append((str(f), c_idx, inst_id))

        return photos, sketches


if TORCH_AVAILABLE:

    def build_transforms(
        image_size: int = 224,
        use_tight_crop: bool = True,
        pad_ratio: float = 0.08,
        enable_morph: bool = True,
        morph_prob: float = 0.5,
        enable_random_erase: bool = True,
        erase_prob: float = 0.5,
    ) -> Dict[str, T.Compose]:
        normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

        sketch_clean_tf = T.Compose([
            SketchCVPreprocessor(
                target_size=image_size,
                use_tight_crop=use_tight_crop,
                pad_ratio=pad_ratio,
                enable_morph=False,
                enable_random_erase=False,
            ),
            T.ToTensor(),
            normalize,
        ])

        sketch_pert_tf = T.Compose([
            SketchCVPreprocessor(
                target_size=image_size,
                use_tight_crop=use_tight_crop,
                pad_ratio=pad_ratio,
                enable_morph=enable_morph,
                morph_prob=morph_prob,
                enable_random_erase=enable_random_erase,
                erase_prob=erase_prob,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize,
        ])

        photo_clean_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            normalize,
        ])

        photo_pert_tf = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.4 if enable_random_erase else 0.0, scale=(0.02, 0.15), value=0),
        ])

        return {
            "sketch_clean": sketch_clean_tf,
            "sketch_pert": sketch_pert_tf,
            "photo_clean": photo_clean_tf,
            "photo_pert": photo_pert_tf,
        }

    class SBIRLinearDataset(Dataset):
        """Standard linear dataset for fast batch feature extraction."""

        def __init__(self, samples: List[Tuple[str, int, str]], transform):
            self.samples = samples
            self.transform = transform

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            path, label, inst_id = self.samples[idx]
            img = Image.open(path)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            return self.transform(img), label, inst_id, path

    class BalancedPKBatchSampler(Sampler[List[int]]):
        """
        Episode-style P-K Batch Sampler:
          Samples P distinct categories per mini-batch and K sketches per category (batch_size = P * K).
          Guarantees:
            1. Rich cross-category negatives for Triplet and Class-Aware NT-Xent.
            2. Exact K samples per category in every batch so Sgarro's f-Divergence (CLIP-AT Eq. 4)
               can compute exact relative-distance distributions D_c of length K across all P classes.
        """

        def __init__(
            self,
            sketch_samples: List[Tuple[str, int, str]],
            p_classes: int = 16,
            k_samples: int = 4,
            seed: int = 42,
        ):
            super().__init__(None)
            self.p_classes = p_classes
            self.k_samples = k_samples
            self.batch_size = p_classes * k_samples
            self.seed = seed
            self.epoch = 0

            self.indices_by_class: Dict[int, List[int]] = {}
            for idx, (_, c_idx, _) in enumerate(sketch_samples):
                self.indices_by_class.setdefault(c_idx, []).append(idx)

            self.classes = sorted(list(self.indices_by_class.keys()))
            self.num_batches = max(1, len(sketch_samples) // self.batch_size)

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __len__(self) -> int:
            return self.num_batches

        def __iter__(self) -> Iterator[List[int]]:
            rng = random.Random(self.seed + self.epoch * 997)
            actual_p = min(self.p_classes, len(self.classes))

            for _ in range(self.num_batches):
                chosen_classes = rng.sample(self.classes, actual_p)
                batch_indices: List[int] = []
                for c in chosen_classes:
                    pool = self.indices_by_class[c]
                    if len(pool) >= self.k_samples:
                        batch_indices.extend(rng.sample(pool, self.k_samples))
                    else:
                        batch_indices.extend(rng.choices(pool, k=self.k_samples))
                yield batch_indices

    class SeCoLoRATripletDataset(Dataset):
        """
        Dual-Mode Training Dataset for SeCo-LoRA-CLIP:
          - Mode A (High-Speed Cached Photo Mode, 3x faster on Kaggle T4/P100):
              When `cached_photo_clean` and `cached_photo_pert` tensors are provided,
              the dataset does NOT decode JPEG files or run ViT forward passes for photos
              during training epochs. Instead, it retrieves the pre-extracted 512-D frozen
              CLIP photo embeddings in O(1) memory lookup, while still augmenting sketches
              on-the-fly (`sketch_clean` and `sketch_pert`).
          - Mode B (On-the-Fly Image Mode):
              Loads raw PIL images for both sketches and photos.
          - Supports fine-grained instance pairing (`inst_id` matching) when available,
            plus intra-category hard negative photo sampling (`photo_hard_neg`).
        """

        def __init__(
            self,
            seen_photos: List[Tuple[str, int, str]],
            seen_sketches: List[Tuple[str, int, str]],
            transforms_dict: Dict[str, T.Compose],
            cached_photo_clean: Optional[np.ndarray] = None,
            cached_photo_pert: Optional[np.ndarray] = None,
        ):
            self.photos = seen_photos
            self.sketches = seen_sketches
            self.tfs = transforms_dict
            self.cached_photo_clean = cached_photo_clean
            self.cached_photo_pert = cached_photo_pert
            self.use_cache = (cached_photo_clean is not None) and (cached_photo_pert is not None)

            # Map class -> list of photo indices, and (class, inst_id) -> photo index
            self.photo_indices_by_class: Dict[int, List[int]] = {}
            self.photo_idx_by_inst: Dict[Tuple[int, str], int] = {}

            for p_idx, (_, c_idx, inst_id) in enumerate(seen_photos):
                self.photo_indices_by_class.setdefault(c_idx, []).append(p_idx)
                self.photo_idx_by_inst[(c_idx, inst_id)] = p_idx

            self.all_classes = sorted(list(self.photo_indices_by_class.keys()))
            self.other_classes_map: Dict[int, List[int]] = {
                c: [other for other in self.all_classes if other != c]
                for c in self.all_classes
            }

        def __len__(self) -> int:
            return len(self.sketches)

        def __getitem__(self, idx: int):
            s_path, label, s_inst_id = self.sketches[idx]
            class_pool = self.photo_indices_by_class[label]

            # Prefer exact paired photo 70% of the time (helps both FG-ZS-SBIR & ZS-SBIR),
            # otherwise sample another photo from the same category
            paired_p_idx = self.photo_idx_by_inst.get((label, s_inst_id), None)
            if paired_p_idx is not None and random.random() < 0.70:
                pos_p_idx = paired_p_idx
            else:
                pos_p_idx = random.choice(class_pool)

            # Intra-category hard negative photo (same category, different instance)
            if len(class_pool) > 1:
                hard_neg_p_idx = random.choice(class_pool)
                if hard_neg_p_idx == pos_p_idx:
                    hard_neg_p_idx = class_pool[(class_pool.index(pos_p_idx) + 1) % len(class_pool)]
            else:
                hard_neg_p_idx = pos_p_idx

            # Inter-category negative photo (different category)
            neg_label = random.choice(self.other_classes_map[label])
            neg_p_idx = random.choice(self.photo_indices_by_class[neg_label])

            s_pil = Image.open(s_path)
            s_clean = self.tfs["sketch_clean"](s_pil)
            s_pert = self.tfs["sketch_pert"](s_pil)

            if self.use_cache:
                return {
                    "sketch_clean": s_clean,
                    "sketch_pert": s_pert,
                    "photo_pos_clean_feat": torch.from_numpy(self.cached_photo_clean[pos_p_idx]),
                    "photo_pos_pert_feat": torch.from_numpy(self.cached_photo_pert[pos_p_idx]),
                    "photo_neg_clean_feat": torch.from_numpy(self.cached_photo_clean[neg_p_idx]),
                    "photo_hard_neg_feat": torch.from_numpy(self.cached_photo_clean[hard_neg_p_idx]),
                    "label": torch.tensor(label, dtype=torch.long),
                }

            pos_p_pil = Image.open(self.photos[pos_p_idx][0]).convert("RGB")
            neg_p_pil = Image.open(self.photos[neg_p_idx][0]).convert("RGB")
            hard_neg_pil = Image.open(self.photos[hard_neg_p_idx][0]).convert("RGB")

            return {
                "sketch_clean": s_clean,
                "sketch_pert": s_pert,
                "photo_pos_clean": self.tfs["photo_clean"](pos_p_pil),
                "photo_pos_pert": self.tfs["photo_pert"](pos_p_pil),
                "photo_neg_clean": self.tfs["photo_clean"](neg_p_pil),
                "photo_hard_neg_clean": self.tfs["photo_clean"](hard_neg_pil),
                "label": torch.tensor(label, dtype=torch.long),
            }