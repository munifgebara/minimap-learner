#!/usr/bin/env python3
"""
Build a single PDF report from a minimap-learner run folder.

Usage examples:
  python minimap_report.py --run-dir outputs/run-20250815-125708
  python minimap_report.py --out my_report.pdf
  python minimap_report.py --title "Minimap Experiments"

If --run-dir is omitted, the script picks the latest "outputs/run-*" folder.
The script searches **recursively** for per-task artifacts and is tolerant to missing files.
"""

import argparse
import os
import json
from pathlib import Path
import datetime as dt
import re

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------- Helpers ----------

def find_latest_run(outputs_dir: str = "outputs") -> Path:
    root = Path(outputs_dir)
    runs = sorted([p for p in root.glob("run-*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run-* folders found under {outputs_dir}.")
    return runs[-1]


def safe_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_text(path: Path, default=""):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


def try_imread(ax, image_path: Path, title: str):
    ax.axis("off")
    if image_path and image_path.exists():
        try:
            img = plt.imread(str(image_path))
            ax.imshow(img)
            ax.set_title(title or image_path.name, fontsize=10)
            return True
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load {image_path.name}:\n{e}", ha="center", va="center", fontsize=9, color="red")
            return False
    else:
        ax.text(0.5, 0.5, f"{title or 'Image'} not found.", ha="center", va="center", fontsize=10)
        return False


def collect_tasks(run_dir: Path):
    """
    Collect task-like directories. Strategy:
      1) Prefer subfolders named 'task_*' under run_dir.
      2) If none, treat any subfolder that has at least one of our expected artifacts as a task.
      3) If still none, treat the run_dir itself as a single task (best-effort).
    Returns list of dicts with paths to artifacts.
    """
    candidates = [p for p in run_dir.rglob("*") if p.is_dir() and p != run_dir]
    task_dirs = [p for p in run_dir.glob("task_*") if p.is_dir()]
    if not task_dirs:
        # Heuristic: find folders with at least one artifact
        for c in candidates:
            if any((c / "metrics" / "metrics.json").exists() or
                   (c / "metrics" / "classification_report.txt").exists() or
                   (c / "confusion_matrix.png").exists() or
                   (c / "roc_curves_micro_macro.png").exists() or
                   (c / "roc_curve.png").exists() for _ in [0]):
                task_dirs.append(c)
    if not task_dirs:
        task_dirs = [run_dir]

    tasks = []
    for tdir in sorted(task_dirs):
        label_guess = tdir.name.replace("task_", "").replace("_", " ")
        metrics_dir = tdir / "metrics"
        xai_dir = tdir / "xai" / "integrated_gradients"

        tasks.append({
            "dir": tdir,
            "label": label_guess,
            "mapping": tdir / "label_mapping.json",
            "metrics_json": metrics_dir / "metrics.json",
            "metrics_csv": metrics_dir / "metrics.csv",
            "report_txt": metrics_dir / "classification_report.txt",
            "cm_png": tdir / "confusion_matrix.png",
            "roc_pngs": [tdir / "roc_curves_micro_macro.png", tdir / "roc_curve.png"],
            "xai_dir": xai_dir
        })
    return tasks


# ---------- PDF Page builders ----------

def page_cover(pdf, title: str, run_dir: Path):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    subtitle = f"Run: {run_dir.name} • Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    authors = ("Munif Gebara Jr. (Principal)\n"
               "Igor S. Wiese (Co-advisor)\n"
               "Yandre M. G. Costa (Advisor)")
    txt = f"{title}\n\n{subtitle}\n\nAuthors:\n{authors}\n\nDirectory:\n{run_dir.resolve()}"
    ax.text(0.5, 0.5, txt, ha="center", va="center", wrap=True, fontsize=14)
    pdf.savefig(fig); plt.close(fig)


def page_run_summary(pdf, run_dir: Path):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_axes([0.05,0.08,0.9,0.84]); ax.axis("off")
    ax.set_title("Run Summary", fontsize=16, pad=12)

    summary_path = run_dir / "metrics" / "all_tasks_summary.json"
    data = safe_json(summary_path, default={})
    if not data:
        ax.text(0.02, 0.95, "No all_tasks_summary.json found.", va="top", fontsize=12)
        pdf.savefig(fig); plt.close(fig); return

    rows = []
    for task, m in data.items():
        rows.append([task, m.get("accuracy"), m.get("roc_auc_macro"), m.get("roc_auc_micro")])
    cols = ["Task", "Accuracy", "ROC AUC (macro)", "ROC AUC (micro)"]
    rendered = [[f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v)) for v in r] for r in rows]
    ax.table(cellText=rendered, colLabels=cols, loc="upper left", cellLoc="center")
    ax.text(0.02, 0.05, f"Source: {summary_path}", fontsize=8, alpha=0.6)
    pdf.savefig(fig); plt.close(fig)


def page_task_overview(pdf, task):
    tdir = task["dir"]
    label = task["label"]
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.05,0.08,0.9,0.84]); ax.axis("off")
    ax.set_title(f"Task: {label}", fontsize=16, pad=12)

    # Left column: metrics.json + label mapping count
    lines = []
    mj = safe_json(task["metrics_json"], default={})
    if mj:
        for k, v in mj.items():
            lines.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    else:
        lines.append("metrics.json not found.")

    mapping = safe_json(task["mapping"], default={})
    lines.append(f"Classes: {len(mapping) if isinstance(mapping, dict) else 'unknown'}")
    lines.append(f"Folder: {tdir}")
    ax.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=12)

    # Right column: learning curves if metrics.csv exists
    mcsv = task["metrics_csv"]
    if mcsv.exists():
        try:
            df = pd.read_csv(mcsv)
            ax2 = fig.add_axes([0.55,0.15,0.4,0.7])
            ax2.set_title("Learning Curves")
            if "train_loss" in df.columns and "val_loss" in df.columns:
                ax2.plot(df.index.values, df["train_loss"].values, label="train_loss")
                ax2.plot(df.index.values, df["val_loss"].values, label="val_loss")
            if "train_acc" in df.columns and "val_acc" in df.columns:
                ax2.plot(df.index.values, df["train_acc"].values, label="train_acc")
                ax2.plot(df.index.values, df["val_acc"].values, label="val_acc")
            ax2.set_xlabel("epoch"); ax2.set_ylabel("value"); ax2.legend()
        except Exception as e:
            ax.text(0.02, 0.10, f"Failed to plot learning curves: {e}", fontsize=10, color="red")
    pdf.savefig(fig); plt.close(fig)


def page_image(pdf, path: Path, title: str):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0,0,1,1])
    ok = try_imread(ax, path, title)
    pdf.savefig(fig); plt.close(fig)


def page_roc(pdf, task):
    # Try any available ROC image in preference order
    for p in task["roc_pngs"]:
        if p.exists():
            page_image(pdf, p, "ROC Curves")
            return
    # If none found, write a placeholder
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.text(0.5, 0.5, "ROC plot not found.", ha="center", va="center")
    pdf.savefig(fig); plt.close(fig)


def page_class_report(pdf, task):
    txt = safe_text(task["report_txt"], default="classification_report.txt not found.")
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.05,0.05,0.9,0.9]); ax.axis("off")
    ax.set_title("Classification Report", pad=12)
    ax.text(0.01, 0.98, txt, va="top", ha="left", family="monospace", fontsize=9)
    pdf.savefig(fig); plt.close(fig)


def page_xai(pdf, task):
    xai_dir = task["xai_dir"]
    if not xai_dir.exists():
        # no-op
        return
    imgs = list(xai_dir.rglob("*.png"))[:4]
    if not imgs:
        return
    fig = plt.figure(figsize=(11.69, 8.27))
    axes = [fig.add_axes([0.05,0.55,0.4,0.4]), fig.add_axes([0.55,0.55,0.4,0.4]),
            fig.add_axes([0.05,0.05,0.4,0.4]), fig.add_axes([0.55,0.05,0.4,0.4])]
    for ax, p in zip(axes, imgs):
        try_imread(ax, p, p.parent.name + " / " + p.name)
    pdf.savefig(fig); plt.close(fig)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None, help="outputs/run-YYYYmmdd-HHMMSS (defaults to latest).")
    ap.add_argument("--out", type=str, default=None, help="Output PDF path (defaults inside run dir).")
    ap.add_argument("--title", type=str, default="Software Feature Detection using Source Code Minimaps as Visual Signatures")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run("outputs")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    out_pdf = Path(args.out) if args.out else (run_dir / "minimap_report.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks(run_dir)
    if not tasks:
        raise FileNotFoundError(f"No task-like folders found under {run_dir}")

    with PdfPages(out_pdf) as pdf:
        page_cover(pdf, args.title, run_dir)
        page_run_summary(pdf, run_dir)
        for task in tasks:
            page_task_overview(pdf, task)
            page_image(pdf, task["cm_png"], "Confusion Matrix")
            page_roc(pdf, task)
            page_class_report(pdf, task)
            page_xai(pdf, task)

    print(f"[OK] Report written to: {out_pdf}")

if __name__ == "__main__":
    main()
