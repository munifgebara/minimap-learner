#!/usr/bin/env python
"""
Entrypoint that runs training for all configured label columns.
Comments are in English to match publication standards.
"""
import argparse
from src.pipeline import run_all

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    p.add_argument("--root-dir", type=str, help="Images root directory (recursive).")
    p.add_argument("--csv-path", type=str, help="CSV metadata path.")
    p.add_argument("--csv-key-column", type=str, help="CSV key column for image matching.")
    p.add_argument("--csv-label-columns", type=str, nargs="+", help="Label columns to train (default: from YAML).")
    return p.parse_args()

def main():
    args = parse_args()
    overrides = {}
    if args.root_dir: overrides.setdefault("data", {})["root_dir"] = args.root_dir
    if args.csv_path: overrides.setdefault("data", {})["csv_path"] = args.csv_path
    if args.csv_key_column: overrides.setdefault("data", {})["csv_key_column"] = args.csv_key_column
    if args.csv_label_columns: overrides.setdefault("data", {})["csv_label_columns"] = args.csv_label_columns
    run_dir = run_all(args.config, overrides)
    print(f"Finished. Outputs in: {run_dir}")

if __name__ == "__main__":
    main()
