# Software Feature Detection using Source Code Minimaps as Visual Signatures

This repository trains deep learning classifiers on grayscale 128×128 minimap images, using GPU when available.  
It automatically runs **three classification tasks** in one execution, using the label columns from the provided CSV:
- `type` (≈63 classes)  
- `project` (≈99 classes)  
- `author` (≈97 classes)  

The pipeline is **clarity-first**, modular, and reproducible (fixed seed). It logs progress,  
saves confusion matrices, ROC curves (micro/macro AUC), classification reports, checkpoints, and XAI (Integrated Gradients).

## Authors
**Munif Gebara Jr.**✠¹ (Principal) — ✠munifgebara@gmail.com  
**Igor S. Wiese**∗³ (Co-advisor) — *igor.wiese@gmail.com*  
**Yandre M. G. Costa**⁺¹ ² (Advisor) — +yandre@din.uem.br  

1. Graduate Program in Computer Science, State University of Maringá, Maringá, Brazil.  
2. Department of Informatics, State University of Maringá, Maringá, Brazil.  
3. Federal University of Technology (UTFPR), Campo Mourão, Brazil.  

## Dataset & CSV
Place your images under `data/minimaps_root/` (recursively) and the CSV at `data/catalogo.csv` (or set paths via CLI/YAML).  
The CSV must include:  
- **image** → key column that matches image basenames (without extension).  
- One of the label columns: **type**, **project**, or **author**.  
- Other descriptive columns such as **file_path** and **file_name** can be used for traceability.  

## Quickstart (PyCharm + venv)
```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train all three tasks (type, project, author):
python train_minimaps.py --config configs/resnet18.yaml
```

## Outputs
A new run folder is created under `outputs/run-YYYYmmdd-HHMMSS/`. For each task, you will find:
- `metrics/metrics.json`, `metrics.csv`, `classification_report.txt`
- `confusion_matrix.png`
- `roc_curves_micro_macro.png` (micro/macro ROC curves), plus per-class AUC in metrics
- `checkpoints/best.pt`, `last.pt`
- `xai/integrated_gradients/` (attributions on a few validation samples per class)

## Configuration
Edit `configs/default.yaml` or override via CLI (`--root-dir`, `--csv-path`, `--csv-label-columns` etc.).

## How to cite (Zenodo)
After creating a Zenodo record for this dataset/code, replace the DOI placeholder below.

**BibTeX:**





@dataset{minimaps-zenodo,
  title = {Software Feature Detection using Source Code Minimaps as Visual Signatures},
  
  year = {2025},
  publisher = {Zenodo},
  version = {1.1},
  doi = {10.5281/zenodo.16929672},
  url = {https://zenodo.org/records/16894040](https://zenodo.org/records/16929672}
}





## License
This code is provided for research purposes. Check dataset/image licenses before redistribution.
