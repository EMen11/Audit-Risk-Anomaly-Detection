# src/export_utils.py
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any
import pandas as pd

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def save_csv(df: pd.DataFrame, path: Path) -> Path:
    _ensure_parent(path)
    df.to_csv(path, index=False)
    return path

def save_json(obj: Dict[str, Any], path: Path) -> Path:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

# --- Fichiers demandés par Power BI ---
def write_transactions_scored(df: pd.DataFrame, out_dir: Path) -> Path:
    return save_csv(df, out_dir / "transactions_scored.csv")

def write_thresholds(df: pd.DataFrame, out_dir: Path) -> Path:
    return save_csv(df, out_dir / "thresholds.csv")

def write_cases(df: pd.DataFrame, out_dir: Path) -> Path:
    return save_csv(df, out_dir / "cases.csv")

def write_metrics_by_type(df: pd.DataFrame, out_dir: Path) -> Path:
    return save_csv(df, out_dir / "metrics_by_type.csv")

def write_confusion(df: pd.DataFrame, out_dir: Path) -> Path:
    return save_csv(df, out_dir / "confusion_matrix.csv")

def write_runlog(obj: Dict[str, Any], out_dir: Path) -> Path:
    return save_json(obj, out_dir / "run.json")

