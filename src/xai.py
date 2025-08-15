from typing import List, Dict
import os, logging, random
import torch
from captum.attr import IntegratedGradients
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _tensor_to_numpy_img(x):
    """Convert a tensor [C,H,W] in [0,1] (or normalized) to grayscale uint8 image for overlay."""
    x = x.detach().cpu().float()
    if x.shape[0] == 3:
        x = x.mean(dim=0, keepdim=True)  # approximate to grayscale
    x = x.squeeze(0)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    x = (x * 255).clamp(0,255).byte().numpy()
    return x

def run_integrated_gradients(model, loader, class_names: List[str], device, run_dir: str, per_class_samples: int = 2, seed: int = 18081975):
    """Run Integrated Gradients on a few validation samples per class and save heatmaps.
    We attribute to the predicted label for each sample to avoid degenerate maps.
    """
    random.seed(seed)
    model.eval()
    ig = IntegratedGradients(model)

    dataset = loader.dataset
    per_class_indices: Dict[int, List[int]] = {}
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        per_class_indices.setdefault(int(y), []).append(idx)
    for k in per_class_indices:
        random.shuffle(per_class_indices[k])
        per_class_indices[k] = per_class_indices[k][:per_class_samples]

    out_dir = os.path.join(run_dir, "xai", "integrated_gradients")
    os.makedirs(out_dir, exist_ok=True)
    for class_id, indices in per_class_indices.items():
        cname = class_names[class_id]
        class_dir = os.path.join(out_dir, f"class_{class_id}_{cname}")
        os.makedirs(class_dir, exist_ok=True)
        for idx in indices:
            x, y_true = dataset[idx]
            x_b = x.unsqueeze(0).to(device)
            x_b.requires_grad = True
            pred_logits = model(x_b)
            pred_label = int(pred_logits.argmax(dim=1).item())
            target = pred_label

            baseline = torch.zeros_like(x_b).to(device)
            attrs, delta = ig.attribute(x_b, baselines=baseline, target=target, return_convergence_delta=True)
            attrs = attrs.squeeze(0)
            attrs_np = attrs.detach().abs().sum(dim=0).cpu().numpy()
            attrs_np = attrs_np - attrs_np.min()
            attrs_np = attrs_np / (attrs_np.max() + 1e-8)

            img_np = _tensor_to_numpy_img(x)

            fig = plt.figure(figsize=(4,4), dpi=200)
            plt.imshow(img_np, cmap="gray")
            plt.imshow(attrs_np, cmap="jet", alpha=0.4)
            plt.axis("off")
            plt.tight_layout()
            fig.savefig(os.path.join(class_dir, f"idx{idx}_true{y_true}_pred{pred_label}.png"))
            plt.close(fig)
    logging.info("Integrated Gradients saved under %s.", out_dir)
