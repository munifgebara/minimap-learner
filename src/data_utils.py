from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
import logging
import pandas as pd
import random
from collections import defaultdict
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from .seeds import worker_init

@dataclass
class ImageRecord:
    path: Path
    key: str
    label_raw: str

def discover_image_files(root_dir: str, allowed_exts: List[str]) -> List[Path]:
    """Recursively list image files under root_dir matching the allowed extensions."""
    root = Path(root_dir)
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if any(fn.lower().endswith(ext) for ext in allowed_exts):
                files.append(Path(dirpath) / fn)
    logging.info("Discovered %d image files under %s.", len(files), root_dir)
    return files

def _normalize_key_with_ext(s: str) -> str:
    """Normalize but keep extension."""
    s = str(s).strip().lower()
    return s

def build_records_from_csv(
    files: List[Path],
    csv_path: str,
    csv_key_column: str,
    csv_label_column: str,
) -> List[ImageRecord]:
    """Match image basenames (without extension) to CSV rows and extract labels."""
    df = pd.read_csv(csv_path)
    if csv_key_column not in df.columns:
        raise ValueError(f"CSV key column '{csv_key_column}' not found in {csv_path}")
    if csv_label_column not in df.columns:
        raise ValueError(f"CSV label column '{csv_label_column}' not found in {csv_path}")
    key_to_label: Dict[str, str] = {}
    keys_series = df[csv_key_column].astype(str).map(_normalize_key_with_ext)
    labels_series = df[csv_label_column].astype(str)

    for k, lab in zip(keys_series, labels_series):
        key_to_label[k] = lab
    records: List[ImageRecord] = []
    misses = 0
    for p in files:
        key = _normalize_key_with_ext(p.name)
        if key in key_to_label:
            records.append(ImageRecord(path=p, key=key, label_raw=key_to_label[key]))
        else:
            misses += 1
    logging.info("Matched %d images to CSV (misses: %d).", len(records), misses)
    return records

def apply_class_sampling(
    records: List[ImageRecord],
    min_samples_per_class: int,
    max_samples_per_class: int,
    seed: int,
) -> Tuple[List[ImageRecord], Dict[str, int]]:
    """Filter out classes with too few samples and downsample those with too many."""
    by_label: Dict[str, List[ImageRecord]] = defaultdict(list)
    for r in records:
        by_label[r.label_raw].append(r)
    kept: List[ImageRecord] = []
    kept_counts: Dict[str, int] = {}
    rng = random.Random(seed)
    ignored_small = []
    downsampled = []
    for label, items in by_label.items():
        n = len(items)
        if n < min_samples_per_class:
            ignored_small.append((label, n))
            continue
        if n > max_samples_per_class:
            items = rng.sample(items, max_samples_per_class)
            downsampled.append((label, n, max_samples_per_class))
        kept.extend(items)
        kept_counts[label] = len(items)
    if ignored_small:
        logging.warning("Ignored %d classes with < min_samples_per_class: %s", 
                        len(ignored_small), ignored_small[:10])
    if downsampled:
        logging.info("Downsampled %d classes > max_samples_per_class.", len(downsampled))
    logging.info("Final dataset: %d samples across %d classes.", len(kept), len(kept_counts))
    return kept, kept_counts

class MinimapDataset(Dataset):
    """Dataset for grayscale 128x128 minimap PNGs with flexible sizing/normalization."""
    def __init__(
        self,
        records: List[ImageRecord],
        label_to_id: Dict[str, int],
        expected_size: Tuple[int, int],
        enforce_size: str,  # "validate" | "resize_if_mismatch" | "error"
        grayscale_mode: str,  # "single" | "replicate_to_rgb"
        transform: Optional[torch.nn.Module] = None,
    ):
        self.records = records
        self.label_to_id = label_to_id
        self.expected_size = expected_size
        self.enforce_size = enforce_size
        self.grayscale_mode = grayscale_mode
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec.path).convert("L")  # grayscale
        if img.size != tuple(self.expected_size):
            if self.enforce_size == "error":
                raise ValueError(f"Image {rec.path} has size {img.size}, expected {self.expected_size}")
            elif self.enforce_size == "resize_if_mismatch":
                img = img.resize(self.expected_size, resample=Image.BILINEAR)
            else:
                logging.warning("Image %s has size %s (expected %s).", rec.path, img.size, self.expected_size)
        if self.transform is not None:
            img_t = self.transform(img)  # shape: [1,H,W]
        else:
            img_t = T.ToTensor()(img)
        if self.grayscale_mode == "replicate_to_rgb":
            img_t = img_t.repeat(3, 1, 1)  # to [3,H,W]
        label_id = self.label_to_id[self.records[idx].label_raw]
        return img_t, label_id

def build_transforms(mode: str, mean: List[float], std: List[float]):
    # coerce to floats in case YAML/CLI passed strings
    mean = [float(m) for m in (mean or [])]
    std  = [float(s) for s in (std or [])]
    if mode == "none":
        return T.Lambda(lambda x: torch.from_numpy(np.array(x)))
    elif mode == "to_tensor_only":
        return T.ToTensor()
    elif mode == "normalize":
        return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    else:
        raise ValueError(f"Unknown transforms.mode: {mode}")


def create_dataloaders(
    train_records: List[ImageRecord],
    val_records: List[ImageRecord],
    test_records: List[ImageRecord],
    label_to_id: Dict[str, int],
    batch_size: int,
    num_workers: int,
    expected_size: Tuple[int, int],
    enforce_size: str,
    grayscale_mode: str,
    transform_train,
    transform_eval,
):
    """Create PyTorch dataloaders for train/val/test splits."""
    train_ds = MinimapDataset(train_records, label_to_id, expected_size, enforce_size, grayscale_mode, transform_train)
    val_ds   = MinimapDataset(val_records,   label_to_id, expected_size, enforce_size, grayscale_mode, transform_eval)
    test_ds  = MinimapDataset(test_records,  label_to_id, expected_size, enforce_size, grayscale_mode, transform_eval)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init, pin_memory=True)
    return train_loader, val_loader, test_loader
