from typing import Dict, Tuple, List
import os, json, logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    optim_config: Dict,
    scheduler_config: Dict,
    amp: bool,
    early_stopping_patience: int,
    run_dir: str,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train the model with progress bars and checkpointing. Returns best model and history."""
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if optim_config["name"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config["lr"],
            weight_decay=optim_config["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optim_config["lr"],
            weight_decay=optim_config["weight_decay"],
            momentum=optim_config["momentum"],
        )

    # Scheduler (optional)
    scheduler = None
    if scheduler_config["name"] == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config["step_size"],
            gamma=scheduler_config["gamma"],
        )

    scaler = GradScaler(enabled=amp)
    model.to(device)

    best_val_acc = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)

    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for xb, yb in pbar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)
        val_loss = val_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save last
        torch.save({"model": model.state_dict()}, os.path.join(checkpoints_dir, "last.pt"))
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict()}, os.path.join(checkpoints_dir, "best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1

        # Persist metrics as CSV each epoch
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "metrics", "metrics.csv"), index=False)

        logging.info(
            "Epoch %d: train_loss=%.4f, train_acc=%.4f, val_loss=%.4f, val_acc=%.4f",
            epoch, train_loss, train_acc, val_loss, val_acc
        )

        if early_stopping_patience and patience_counter >= early_stopping_patience:
            logging.info("Early stopping triggered (patience=%d).", early_stopping_patience)
            break

    return model, history


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def evaluate_model(model, loader, device, class_names: List[str], run_dir: str) -> Dict:
    """Evaluate model on test set, producing metrics, confusion matrix, and ROC curves."""
    model.to(device)
    model.eval()

    all_logits = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Evaluate[test]"):
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(yb.numpy())

    logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, len(class_names)))
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,), dtype=int)
    y_pred = logits.argmax(axis=1) if logits.size else np.zeros((0,), dtype=int)
    y_proba = _softmax(logits) if logits.size else logits

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    with open(os.path.join(run_dir, "metrics", "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig_cm = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    fig_cm.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig_cm)

    # ROC-AUC (macro/micro) + curves
    metrics: Dict[str, float] = {}
    try:
        if len(class_names) > 2:
            y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        else:
            y_true_bin = y_true
        macro_auc = roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="macro")
        micro_auc = roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="micro")
        metrics["roc_auc_macro"] = float(macro_auc)
        metrics["roc_auc_micro"] = float(micro_auc)

        # Curves (micro/macro)
        if len(class_names) > 2:
            fpr = {}
            tpr = {}
            for i in range(len(class_names)):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            # Micro-average
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
            # Macro-average
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(len(class_names)):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= len(class_names)

            fig = plt.figure(figsize=(8, 6))
            plt.plot(fpr_micro, tpr_micro, label=f"micro-average ROC (AUC = {metrics['roc_auc_micro']:.3f})")
            plt.plot(all_fpr, mean_tpr, label=f"macro-average ROC (AUC = {metrics['roc_auc_macro']:.3f})")
            plt.plot([0, 1], [0, 1], '--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves (micro & macro)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            fig.savefig(os.path.join(run_dir, "roc_curves_micro_macro.png"), dpi=200)
            plt.close(fig)
        else:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            fig = plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['roc_auc_macro']:.3f})")
            plt.plot([0, 1], [0, 1], '--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (binary)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            fig.savefig(os.path.join(run_dir, "roc_curve.png"), dpi=200)
            plt.close(fig)
    except Exception as e:
        logging.warning("ROC-AUC computation failed: %s", e)

    # Accuracy
    acc = (y_true == y_pred).mean() if y_true.size else 0.0
    metrics.update({"accuracy": float(acc)})

    with open(os.path.join(run_dir, "metrics", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
