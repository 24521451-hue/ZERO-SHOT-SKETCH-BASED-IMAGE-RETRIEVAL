#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import math
import random
from pathlib import Path
from typing import List, Tuple, Dict

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
    from torch.utils.data import Dataset
    import torchvision.transforms as T


class SketchCVPreprocessor:
    """
    Domain-specific Computer Vision pipeline for hand-drawn sketches:
      1. Alpha-safe RGBA -> RGB conversion (prevents transparent background turning black).
      2. Tight-Crop + Symmetric 1:1 Padding (maximizes active stroke patches in ViT-B/32).
      3. Morphological Dilation / Erosion (simulates stroke width variations).
      4. White Random Erasing (simulates partial/missing strokes for perturbation consistency).
    """

    def __init__(
        self,
        target_size: int = 224,
        pad_ratio: float = 0.08,
        enable_morph: bool = False,
        morph_prob: float = 0.5,
        enable_random_erase: bool = False,
        erase_prob: float = 0.5,
    ):
        self.target_size = target_size
        self.pad_ratio = pad_ratio
        self.enable_morph = enable_morph
        self.morph_prob = morph_prob
        self.enable_random_erase = enable_random_erase
        self.erase_prob = erase_prob

    @staticmethod
    def to_clean_sketch_rgb(pil_img: Image.Image) -> np.ndarray:
        if pil_img.mode == "RGBA":
            bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(bg, pil_img).convert("RGB")
            return np.array(composited, dtype=np.uint8)
        return np.array(pil_img.convert("RGB"), dtype=np.uint8)

    def tight_crop_and_symmetric_pad(self, img_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        coords = cv2.findNonZero((inv > 30).astype(np.uint8))

        if coords is None:
            return cv2.resize(
                img_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA
            )

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
        if not self.enable_morph or random.random() > self.morph_prob:
            return img_rgb

        kernel_size = random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if random.random() < 0.65:
            return cv2.erode(img_rgb, kernel, iterations=1)
        return cv2.dilate(img_rgb, kernel, iterations=1)

    def apply_white_random_erasing(self, img_rgb: np.ndarray) -> np.ndarray:
        if not self.enable_random_erase or random.random() > self.erase_prob:
            return img_rgb

        out = img_rgb.copy()
        h, w, _ = out.shape
        area = h * w

        for _ in range(10):
            target_area = random.uniform(0.02, 0.12) * area
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
    and partitions into Seen (104 classes) and Unseen (21 classes).
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
            raise FileNotFoundError(f"Directory not found: {candidate}")
        subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        if len(subdirs) < 20:
            for sd in subdirs:
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
        photos, sketches = [], []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for cname in class_list:
            c_idx = cls2idx[cname]
            p_folder = self.photo_dir / cname
            s_folder = self.sketch_dir / cname

            if p_folder.exists():
                for f in sorted(p_folder.iterdir()):
                    if f.suffix.lower() in valid_exts:
                        inst_id = self.extract_instance_id(str(f), is_sketch=False)
                        photos.append((str(f), c_idx, inst_id))

            if s_folder.exists():
                for f in sorted(s_folder.iterdir()):
                    if f.suffix.lower() in valid_exts:
                        inst_id = self.extract_instance_id(str(f), is_sketch=True)
                        sketches.append((str(f), c_idx, inst_id))

        return photos, sketches


if TORCH_AVAILABLE:

    def build_transforms(image_size: int = 224, pad_ratio: float = 0.08) -> Dict[str, T.Compose]:
        normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

        sketch_clean_tf = T.Compose([
            SketchCVPreprocessor(
                target_size=image_size,
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
                pad_ratio=pad_ratio,
                enable_morph=True,
                morph_prob=0.6,
                enable_random_erase=True,
                erase_prob=0.5,
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
            T.RandomErasing(p=0.4, scale=(0.02, 0.15), value=0),
        ])

        return {
            "sketch_clean": sketch_clean_tf,
            "sketch_pert": sketch_pert_tf,
            "photo_clean": photo_clean_tf,
            "photo_pert": photo_pert_tf,
        }

    class SBIRLinearDataset(Dataset):
        def __init__(self, samples: List[Tuple[str, int, str]], transform):
            self.samples = samples
            self.transform = transform

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            path, label, inst_id = self.samples[idx]
            img = Image.open(path)
            if img.mode != "RGBA":
                img = img.convert("RGB")
            return self.transform(img), label, inst_id, path

    class SeCoLoRATripletDataset(Dataset):
        def __init__(
            self,
            seen_photos: List[Tuple[str, int, str]],
            seen_sketches: List[Tuple[str, int, str]],
            transforms_dict: Dict[str, T.Compose],
        ):
            self.sketches = seen_sketches
            self.tfs = transforms_dict

            self.photos_by_class: Dict[int, List[str]] = {}
            for path, c_idx, _ in seen_photos:
                self.photos_by_class.setdefault(c_idx, []).append(path)

            self.all_classes = sorted(list(self.photos_by_class.keys()))

        def __len__(self) -> int:
            return len(self.sketches)

        def __getitem__(self, idx: int):
            s_path, label, _ = self.sketches[idx]
            pos_p_path = random.choice(self.photos_by_class[label])
            neg_label = random.choice([c for c in self.all_classes if c != label])
            neg_p_path = random.choice(self.photos_by_class[neg_label])

            s_pil = Image.open(s_path)
            pos_p_pil = Image.open(pos_p_path).convert("RGB")
            neg_p_pil = Image.open(neg_p_path).convert("RGB")

            return {
                "sketch_clean": self.tfs["sketch_clean"](s_pil),
                "sketch_pert": self.tfs["sketch_pert"](s_pil),
                "photo_pos_clean": self.tfs["photo_clean"](pos_p_pil),
                "photo_pos_pert": self.tfs["photo_pert"](pos_p_pil),
                "photo_neg_clean": self.tfs["photo_clean"](neg_p_pil),
                "label": torch.tensor(label, dtype=torch.long),
            }
