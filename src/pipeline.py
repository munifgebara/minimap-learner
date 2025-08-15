import os, json, logging, yaml, datetime as dt
from typing import List, Dict
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from .seeds import seed_all
from .logging_utils import setup_logging
from .data_utils import discover_image_files, build_records_from_csv, apply_class_sampling, build_transforms, create_dataloaders
from .models import build_model
from .train_eval import train_model, evaluate_model
from .xai import run_integrated_gradients


def _coerce_config_types(cfg: dict) -> dict:
    """Force numeric/bool types from possible string values to avoid 'float' vs 'str' errors."""
    def to_bool(x):
        if isinstance(x, bool): return x
        if isinstance(x, str): return x.strip().lower() in {"1","true","yes","y","on"}
        return bool(x)

    # data.split
    for k in ("train","val","test"):
        if "split" in cfg and k in cfg["split"]:
            cfg["split"][k] = float(cfg["split"][k])

    # sampling
    if "sampling" in cfg:
        cfg["sampling"]["min_samples_per_class"] = int(cfg["sampling"]["min_samples_per_class"])
        cfg["sampling"]["max_samples_per_class"] = int(cfg["sampling"]["max_samples_per_class"])

    # transforms
    if "transforms" in cfg and "normalize" in cfg["transforms"]:
        nm = cfg["transforms"]["normalize"]
        # deixa como lista de floats
        nm["mean"] = [float(v) for v in nm.get("mean", [])]
        nm["std"]  = [float(v) for v in nm.get("std", [])]

    # train
    if "train" in cfg:
        cfg["train"]["epochs"] = int(cfg["train"]["epochs"])
        cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"])
        cfg["train"]["num_workers"] = int(cfg["train"]["num_workers"])
        cfg["train"]["amp"] = to_bool(cfg["train"]["amp"])
        cfg["train"]["early_stopping_patience"] = int(cfg["train"]["early_stopping_patience"])

    # optim
    if "optim" in cfg:
        cfg["optim"]["lr"] = float(cfg["optim"]["lr"])
        cfg["optim"]["weight_decay"] = float(cfg["optim"]["weight_decay"])
        # momentum só é usado no SGD
        if "momentum" in cfg["optim"]:
            try:
                cfg["optim"]["momentum"] = float(cfg["optim"]["momentum"])
            except Exception:
                pass

    # scheduler
    if "scheduler" in cfg and cfg["scheduler"].get("name"):
        if "step_size" in cfg["scheduler"]:
            cfg["scheduler"]["step_size"] = int(cfg["scheduler"]["step_size"])
        if "gamma" in cfg["scheduler"]:
            cfg["scheduler"]["gamma"] = float(cfg["scheduler"]["gamma"])

    # seed
    if "seed" in cfg:
        cfg["seed"] = int(cfg["seed"])

    return cfg


def run_experiment_for_label(config: Dict, label_column: str, global_run_dir: str) -> Dict:
    """Run the full pipeline for a single label column and return summary metrics."""
    seed_all(int(config["seed"]))

    run_dir = os.path.join(global_run_dir, f"task_{label_column.replace(' ', '_')}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)

    logging.info("=== Task: %s ===", label_column)
    files = discover_image_files(config["data"]["root_dir"], config["data"]["allowed_extensions"])
    records = build_records_from_csv(files, config["data"]["csv_path"], config["data"]["csv_key_column"], label_column)

    records, counts = apply_class_sampling(
        records,
        config["sampling"]["min_samples_per_class"],
        config["sampling"]["max_samples_per_class"],
        int(config["seed"]),
    )
    if len(counts) < 2:
        raise RuntimeError(f"Not enough classes after filtering for label '{label_column}'.")

    classes = sorted(counts.keys())
    label_to_id = {c: i for i, c in enumerate(classes)}
    id_to_label = {i: c for c, i in label_to_id.items()}
    with open(os.path.join(run_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_to_id, f, indent=2, ensure_ascii=False)

    y = np.array([label_to_id[r.label_raw] for r in records])
    idx = np.arange(len(records))
    tr_idx, te_idx, _, _ = train_test_split(idx, y, test_size=config["split"]["test"], random_state=int(config["seed"]), stratify=y)
    y_tr = y[tr_idx]
    tr2_idx, va_idx, _, _ = train_test_split(tr_idx, y_tr, test_size=config["split"]["val"]/(config["split"]["train"]+config["split"]["val"]), random_state=int(config["seed"]), stratify=y_tr)

    train_records = [records[i] for i in tr2_idx]
    val_records   = [records[i] for i in va_idx]
    test_records  = [records[i] for i in te_idx]

    tfm_train = build_transforms(config["transforms"]["mode"], config["transforms"]["normalize"]["mean"], config["transforms"]["normalize"]["std"])
    tfm_eval  = build_transforms(config["transforms"]["mode"], config["transforms"]["normalize"]["mean"], config["transforms"]["normalize"]["std"])

    train_loader, val_loader, test_loader = create_dataloaders(
        train_records, val_records, test_records, label_to_id,
        config["train"]["batch_size"], config["train"]["num_workers"],
        tuple(config["data"]["expected_size"]), config["data"]["enforce_size"],
        config["data"]["grayscale_mode"], tfm_train, tfm_eval
    )

    arch = config["model"]["arch"]
    if arch == "vit_b_16" and config["data"]["grayscale_mode"] == "single":
        logging.warning("ViT requires 3-channel input. Switch dataset to 'replicate_to_rgb' or choose another arch.")
    model = build_model(arch, num_classes=len(classes), pretrained=bool(config["model"]["pretrained"]), grayscale_mode=config["data"]["grayscale_mode"])
    if bool(config["model"]["freeze_backbone"]):
        to_unfreeze = ["fc", "classifier", "heads.head"]
        for name, p in model.named_parameters():
            if not any(k in name for k in to_unfreeze):
                p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = train_model(
        model, train_loader, val_loader, device,
        int(config["train"]["epochs"]), config["optim"], config["scheduler"],
        bool(config["train"]["amp"]), int(config["train"]["early_stopping_patience"]), run_dir
    )

    metrics = evaluate_model(model, test_loader, device, [id_to_label[i] for i in range(len(classes))], run_dir)

    if bool(config["xai"]["enabled"]):
        try:
            run_integrated_gradients(model, val_loader, [id_to_label[i] for i in range(len(classes))], device, run_dir, int(config["xai"]["per_class_samples"]), int(config["seed"]))
        except Exception as e:
            logging.warning("XAI failed: %s", e)

    with open(os.path.join(run_dir, "metrics", "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"label_column": label_column, "classes": classes, "metrics": metrics}, f, indent=2, ensure_ascii=False)
    return metrics

def run_all(config_path: str, overrides: Dict = None) -> str:
    """Run the pipeline for all label columns in config and return the run folder path."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    overrides = overrides or {}

    config = _coerce_config_types(config)

    for k, v in overrides.items():
        if isinstance(v, dict) and k in config and isinstance(config[k], dict):
            config[k].update(v)
        else:
            config[k] = v
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("outputs", f"run-{ts}")
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(os.path.join(run_dir, "logs"))
    logging.info("Config loaded from %s", config_path)
    logging.info("Run dir: %s", run_dir)
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    label_columns: List[str] = config["data"]["csv_label_columns"]
    summary = {}
    for col in label_columns:
        try:
            m = run_experiment_for_label(config, col, run_dir)
            summary[col] = m
        except Exception as e:
            logging.error("Task failed for label '%s': %s", col, e)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    with open(os.path.join(run_dir, "metrics", "all_tasks_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info("All tasks finished. Summary: %s", summary)
    return run_dir
