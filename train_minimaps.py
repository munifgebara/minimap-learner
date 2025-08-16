#!/usr/bin/env python
"""
Executa múltiplas configurações (YAML) da pasta especificada,
chamando src.pipeline.run_all para cada arquivo encontrado.

- Mantém seus artefatos/resultados como estão.
- Gera um JSONL consolidado com hora de início/fim e resolved_config por execução.
"""

import argparse
import glob
import json
import os
import random
import time
from datetime import datetime

import yaml
from src.pipeline import run_all


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_args():
    p = argparse.ArgumentParser(description="Rodar múltiplas configs em sequência.")
    # Modo antigo (um único arquivo) ainda suportado para compatibilidade:
    p.add_argument("--config", type=str, help="Caminho de um YAML específico (modo legado).")

    # Novo: varrer uma pasta de configs
    p.add_argument("--config-dir", type=str, default="configs",
                   help="Pasta com os arquivos YAML (default: config)")
    p.add_argument("--pattern", type=str, default="*.y*ml",
                   help='Glob para encontrar arquivos (default: "*.y*ml")')
    p.add_argument("--shuffle", action="store_true",
                   help="Embaralha a ordem de execução")
    p.add_argument("--limit", type=int, default=None,
                   help="Limita a N arquivos")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed para embaralhamento (default: 42)")
    p.add_argument("--out-log", type=str, default="runs_multi.jsonl",
                   help="Arquivo JSONL consolidado (default: runs_multi.jsonl)")

    # Overrides opcionais (aplicados em TODAS as configs)
    p.add_argument("--root-dir", type=str, help="Diretório raiz das imagens.")
    p.add_argument("--csv-path", type=str, help="Caminho do CSV de metadados.")
    p.add_argument("--csv-key-column", type=str, help="Coluna-chave do CSV.")
    p.add_argument("--csv-label-columns", type=str, nargs="+",
                   help="Colunas de rótulo (default: as do YAML).")
    return p.parse_args()


def build_overrides(args) -> dict:
    overrides = {}
    if args.root_dir:
        overrides.setdefault("data", {})["root_dir"] = args.root_dir
    if args.csv_path:
        overrides.setdefault("data", {})["csv_path"] = args.csv_path
    if args.csv_key_column:
        overrides.setdefault("data", {})["csv_key_column"] = args.csv_key_column
    if args.csv_label_columns:
        overrides.setdefault("data", {})["csv_label_columns"] = args.csv_label_columns
    return overrides


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    overrides = build_overrides(args)

    # Se o usuário passar --config explicitamente, roda somente ele (modo legado)
    if args.config:
        cfg_files = [args.config]
    else:
        # Varrer a pasta config/ por padrão
        pattern_path = os.path.join(args.config_dir, args.pattern)
        cfg_files = sorted(f for f in glob.glob(pattern_path) if os.path.isfile(f))
        if not cfg_files:
            print(f"[WARN] Nenhum arquivo encontrado em: {pattern_path}")
            return
        if args.shuffle:
            rng = random.Random(args.seed)
            rng.shuffle(cfg_files)
        if args.limit is not None:
            cfg_files = cfg_files[:args.limit]

    print(f"[INFO] Executando {len(cfg_files)} configuração(ões).")
    print(f"[INFO] Log consolidado: {args.out_log}")
    os.makedirs(os.path.dirname(args.out_log) or ".", exist_ok=True)

    with open(args.out_log, "a", encoding="utf-8") as flog:
        for i, cfg_path in enumerate(cfg_files, start=1):
            print(f"\n=== [{i}/{len(cfg_files)}] {cfg_path} ===")
            started_at = iso_now()
            t0 = time.time()
            status = "ok"
            error = None
            run_dir = None

            try:
                # Carrega o YAML para registrar como resolved_config (mínimo requerido)
                resolved_config = load_yaml(cfg_path)

                # Executa seu pipeline normal para UM arquivo
                run_dir = run_all(cfg_path, overrides)

            except KeyboardInterrupt:
                status = "interrupted"
                error = "KeyboardInterrupt"
                print("\n[INTERRUPTED] Execução interrompida pelo usuário.")
                # Loga e re-levanta
                elapsed = round(time.time() - t0, 3)
                ended_at = iso_now()
                record = {
                    "config_file": cfg_path,
                    "config_name": os.path.splitext(os.path.basename(cfg_path))[0],
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "elapsed_seconds": elapsed,
                    "status": status,
                    "run_dir": run_dir,
                    "resolved_config": resolved_config if "resolved_config" in locals() else None,
                    "error": error,
                }
                flog.write(json.dumps(record, ensure_ascii=False) + "\n")
                flog.flush()
                raise

            except Exception as e:
                status = "error"
                error = repr(e)
                print(f"[ERRO] {cfg_path}: {e}")

            elapsed = round(time.time() - t0, 3)
            ended_at = iso_now()

            record = {
                "config_file": cfg_path,
                "config_name": os.path.splitext(os.path.basename(cfg_path))[0],
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_seconds": elapsed,
                "status": status,
                "run_dir": run_dir,
                # resolved_config: aqui usamos o YAML carregado; se seu run_all retornar
                # um dict final, você pode substituí-lo por ele.
                "resolved_config": resolved_config if "resolved_config" in locals() else None,
                "error": error,
            }
            flog.write(json.dumps(record, ensure_ascii=False) + "\n")
            flog.flush()

            tag = "[OK]" if status == "ok" else f"[{status.upper()}]"
            destino = f" -> {run_dir}" if run_dir else ""
            print(f"{tag} {cfg_path} | {elapsed}s{destino}")

    # Mensagem final (compatível com modo antigo)
    if len(cfg_files) == 1 and run_dir:
        print(f"Finished. Outputs in: {run_dir}")
    else:
        print("Finished. Consulte o JSONL consolidado para os detalhes de cada execução.")


if __name__ == "__main__":
    main()
