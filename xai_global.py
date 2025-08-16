#!/usr/bin/env python3
"""
xai_global.py — Global XAI for minimap-learner

Generates per-class and overall attribution summaries (Integrated Gradients) for each task in a run,
and compiles them into a single PDF report.

Usage:
  python xai_global.py [--run-dir outputs/run-YYYYmmdd-HHMMSS] [--out <report.pdf>]
                       [--per-class-samples 8] [--ig-steps 32]

Defaults:
  --run-dir : latest run under outputs/
  --out     : <run-dir>/xai_global_report.pdf
"""
import argparse
import os
import json
from pathlib import Path
import datetime as dt
import random

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from captum.attr import IntegratedGradients
import yaml

# Import project modules (assumes running from project root)
from src.seeds import seed_all
from src.data_utils import discover_image_files, build_records_from_csv, apply_class_sampling, build_transforms, create_dataloaders
from src.models import build_model


def find_latest_run(outputs_dir: str = "outputs") -> Path:
    root = Path(outputs_dir)
    runs = sorted([p for p in root.glob("run-*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run-* folders found under {outputs_dir}.")
    return runs[-1]


def load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found under {run_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_val_loader_for_label(cfg: dict, label_column: str, seed: int):
    """Reconstruct loaders similarly to training; uses same seed for deterministic split."""
    from sklearn.model_selection import train_test_split
    files = discover_image_files(cfg["data"]["root_dir"], cfg["data"]["allowed_extensions"])
    records = build_records_from_csv(files, cfg["data"]["csv_path"], cfg["data"]["csv_key_column"], label_column)
    records, counts = apply_class_sampling(
        records,
        int(cfg["sampling"]["min_samples_per_class"]),
        int(cfg["sampling"]["max_samples_per_class"]),
        int(seed),
    )
    if len(counts) < 2:
        raise RuntimeError(f"Not enough classes for label '{label_column}' after sampling.")
    classes = sorted(counts.keys())
    label_to_id = {c: i for i, c in enumerate(classes)}

    # stratified split
    y = np.array([label_to_id[r.label_raw] for r in records])
    idx = np.arange(len(records))
    from sklearn.model_selection import train_test_split
    tr_idx, te_idx, _, _ = train_test_split(idx, y, test_size=float(cfg["split"]["test"]), random_state=int(seed), stratify=y)
    y_tr = y[tr_idx]
    tr2_idx, va_idx, _, _ = train_test_split(tr_idx, y_tr, test_size=float(cfg["split"]["val"])/(float(cfg["split"]["train"])+float(cfg["split"]["val"])), random_state=int(seed), stratify=y_tr)

    # transforms
    tfm_eval = build_transforms(cfg["transforms"]["mode"], cfg["transforms"]["normalize"]["mean"], cfg["transforms"]["normalize"]["std"])

    # dataloaders (we only need val)
    _, val_loader, _ = create_dataloaders(
        [records[i] for i in tr2_idx],
        [records[i] for i in va_idx],
        [records[i] for i in te_idx],
        label_to_id,
        int(cfg["train"]["batch_size"]),
        int(cfg["train"]["num_workers"]),
        tuple(cfg["data"]["expected_size"]),
        cfg["data"]["enforce_size"],
        cfg["data"]["grayscale_mode"],
        tfm_eval,
        tfm_eval,
    )
    id_to_label = {i: c for c, i in label_to_id.items()}
    return val_loader, id_to_label


def load_model_for_task(cfg: dict, task_dir: Path, num_classes: int):
    arch = cfg["model"]["arch"]
    model = build_model(arch, num_classes=num_classes, pretrained=False, grayscale_mode=cfg["data"]["grayscale_mode"])
    ckpt = task_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        ckpt = task_dir / "checkpoints" / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found for {task_dir}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    return model


def compute_global_ig(task_dir: Path, cfg: dict, device: torch.device, per_class_samples: int = 8, ig_steps: int = 32, seed: int = 18081975):
    seed_all(seed)
    # Load label mapping for class order
    mapping_path = task_dir / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"{mapping_path} not found.")
    with open(mapping_path, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {int(v): k for k, v in label_to_id.items()}
    num_classes = len(id_to_label)

    # Build val loader consistent with config
    label_column = task_dir.name.replace("task_", "").replace("_", " ")
    val_loader, _ = build_val_loader_for_label(cfg, label_column, int(cfg["seed"]))

    # Model
    model = load_model_for_task(cfg, task_dir, num_classes=num_classes)
    model.eval().to(device)

    ig = IntegratedGradients(model)

    # For each class, pick samples with ground-truth == class (from val set)
    ds = val_loader.dataset
    # Build index list per class based on saved mapping
    per_class_indices = {cid: [] for cid in range(num_classes)}
    for idx in range(len(ds)):
        _, y = ds[idx]
        if int(y) in per_class_indices:
            per_class_indices[int(y)].append(idx)

    rng = random.Random(seed)
    out_dir = task_dir / "xai" / "global_ig"
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_sum = None
    overall_count = 0

    # Loop classes
    for cid in range(num_classes):
        cname = id_to_label.get(cid, f"class_{cid}")
        indices = per_class_indices.get(cid, [])
        if not indices:
            continue
        rng.shuffle(indices)
        indices = indices[:per_class_samples]

        class_sum = None
        class_count = 0

        for idx in indices:
            x, y_true = ds[idx]
            x_b = x.unsqueeze(0).to(device)
            x_b.requires_grad = True
            # attribute to predicted class to avoid degenerate attributions
            with torch.no_grad():
                pred_logits = model(x_b)
                pred_label = int(pred_logits.argmax(dim=1).item())
            target = pred_label

            baseline = torch.zeros_like(x_b).to(device)
            attrs = ig.attribute(x_b, baselines=baseline, target=target, n_steps=int(ig_steps))
            attrs = attrs.squeeze(0).abs().sum(dim=0).detach().cpu().numpy()  # [H,W]

            # normalize per-sample
            attrs = attrs - attrs.min()
            denom = (attrs.max() - attrs.min()) if attrs.max() > attrs.min() else 1.0
            attrs = attrs / denom

            if class_sum is None:
                class_sum = attrs
            else:
                class_sum += attrs
            class_count += 1

            if overall_sum is None:
                overall_sum = attrs.copy()
            else:
                overall_sum += attrs
            overall_count += 1

        if class_count > 0:
            class_mean = class_sum / class_count
            # Save heatmap
            fig = plt.figure(figsize=(3.2, 3.2), dpi=200)
            plt.imshow(class_mean, cmap="inferno")
            plt.axis("off")
            plt.title(f"{cname} (n={class_count})", fontsize=8)
            fig.savefig(out_dir / f"class_{cid:03d}_{cname}_mean_ig.png", bbox_inches="tight", pad_inches=0.0)
            plt.close(fig)

    if overall_count > 0:
        overall_mean = overall_sum / overall_count
        fig = plt.figure(figsize=(3.2, 3.2), dpi=200)
        plt.imshow(overall_mean, cmap="inferno")
        plt.axis("off")
        plt.title(f"Overall mean IG (n={overall_count})", fontsize=8)
        fig.savefig(out_dir / f"overall_mean_ig.png", bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

    return out_dir


def make_pdf(run_dir: Path, task_dirs, out_pdf: Path):
    with PdfPages(out_pdf) as pdf:
        # Cover
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0,0,1,1]); ax.axis("off")
        title = "Global XAI (Integrated Gradients) — Minimap Learner"
        ax.text(0.5, 0.85, title, ha="center", va="center", fontsize=16)
        ax.text(0.5, 0.80, f"Run: {run_dir.name}", ha="center", va="center", fontsize=10)
        ax.text(0.5, 0.75, dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ha="center", va="center", fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        # One section per task
        for task in task_dirs:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0,0,1,1]); ax.axis("off")
            ax.text(0.5, 0.95, f"Task: {task.name}", ha="center", fontsize=14)
            out_dir = task / "xai" / "global_ig"
            overall = out_dir / "overall_mean_ig.png"
            if overall.exists():
                img = plt.imread(str(overall))
                ax_img = fig.add_axes([0.35, 0.72, 0.3, 0.3])
                ax_img.axis("off")
                ax_img.imshow(img)
                ax.text(0.5, 0.68, "Overall mean attribution", ha="center", fontsize=10)

            # Grid of per-class maps (up to 12)
            class_imgs = sorted(out_dir.glob("class_*_mean_ig.png"))[:12]
            if class_imgs:
                cols = 4
                rows = int(np.ceil(len(class_imgs)/cols))
                start_y = 0.63
                cell_w = 0.22
                cell_h = 0.18
                x0 = 0.05
                y0 = start_y
                i = 0
                for r in range(rows):
                    for c in range(cols):
                        if i >= len(class_imgs): break
                        img = plt.imread(str(class_imgs[i]))
                        ax_img = fig.add_axes([x0 + c*(cell_w+0.02), y0 - r*(cell_h+0.04), cell_w, cell_h])
                        ax_img.axis("off")
                        ax_img.imshow(img)
                        i += 1

            pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--per-class-samples", type=int, default=8)
    ap.add_argument("--ig-steps", type=int, default=32)
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run("outputs")
    cfg = load_config(run_dir)
    out_pdf = Path(args.out) if args.out else run_dir / "xai_global_report.pdf"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(int(cfg.get("seed", 18081975)))

    # Collect task directories
    task_dirs = [p for p in run_dir.glob("task_*") if p.is_dir()]
    if not task_dirs:
        raise FileNotFoundError(f"No task_* folders found in {run_dir}")

    # Compute and save per-task global IG
    for task in task_dirs:
        print(f"[XAI] Processing {task.name} ...")
        compute_global_ig(task, cfg, device,
                          per_class_samples=int(args.per_class_samples),
                          ig_steps=int(args.ig_steps),
                          seed=int(cfg.get("seed", 18081975)))

    # Build a consolidated PDF
    make_pdf(run_dir, task_dirs, out_pdf)
    print(f"Global XAI report written to: {out_pdf}")


if __name__ == "__main__":
    main()


def run_xai_global(run_dir: str, per_class_samples: int = 8, ig_steps: int = 32, out: str | None = None):
    """Programmatic entrypoint to compute global XAI and write consolidated PDF."""
    import sys
    argv = ["xai_global.py", "--run-dir", str(run_dir), "--per-class-samples", str(per_class_samples), "--ig-steps", str(ig_steps)]
    if out:
        argv += ["--out", out]
    # Monkey-patch sys.argv for the existing argparse-based main()
    old = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old
