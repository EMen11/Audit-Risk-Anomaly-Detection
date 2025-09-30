# run_exports_demo.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import platform, sys

from src.paths import run_dir
from src.export_utils import (
    write_transactions_scored, write_thresholds, write_cases,
    write_metrics_by_type, write_confusion, write_runlog
)

def demo_transactions_scored(n=500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2025-01-01", periods=n, freq="H")
    df = pd.DataFrame({
        "transaction_id": np.arange(n),
        "customer_id": rng.integers(1000, 1050, size=n),
        "timestamp": ts,
        "amount": np.round(rng.normal(120, 40, size=n).clip(1, None), 2),
        "country": rng.choice(["CH","FR","DE","IT","US"], size=n, p=[.6,.1,.1,.1,.1]),
        "type": rng.choice(["ACHAT_CB","VIREMENT","RETRAIT_DAB","PRELEVEMENT"], size=n),
    })
    # scores & flags simulés
    df["score"] = np.clip(rng.normal(0.5, 0.2, size=n), 0, 1)
    df["is_ml_flag"] = df["score"] > 0.85
    df["rule_night_high_amount"] = (df["amount"] > 200) & (df["timestamp"].dt.hour.isin([0,1,2,3,4]))
    df["rule_new_country"] = rng.random(n) < 0.02
    df["rule_burst"] = rng.random(n) < 0.01
    df["rule_travel"] = rng.random(n) < 0.005
    df["rule_duplicate"] = rng.random(n) < 0.005
    rule_cols = ["rule_night_high_amount","rule_new_country","rule_burst","rule_travel","rule_duplicate"]
    df["rule_count"] = df[rule_cols].sum(axis=1)
    df["is_rule_flag"] = df["rule_count"] > 0
    df["is_flag"] = df["is_ml_flag"] | df["is_rule_flag"]
    return df

def demo_thresholds() -> pd.DataFrame:
    return pd.DataFrame({
        "threshold":   [0.80, 0.85, 0.90],
        "precision":   [0.60, 0.70, 0.82],
        "recall":      [0.65, 0.55, 0.40],
        "fpr":         [0.020, 0.014, 0.009],
        "tnr":         [0.980, 0.986, 0.991],
        "flagged_count":[120,   80,   50],
        "workload_pct":[12.0,   8.0,  5.0],
        "expected_cost":[0.45, 0.42, 0.44],
        "mode":        ["coverage","lite","custom"]
    })

def demo_cases(df_tx: pd.DataFrame) -> pd.DataFrame:
    flagged = df_tx[df_tx["is_flag"]].copy().head(50)
    flagged["reasons_top3"] = np.where(
        flagged["rule_night_high_amount"],
        "amount_z>3; 02:00-04:00; high_score",
        "high_score; new_country; short_gap"
    )
    flagged["triggered_rules"] = flagged[[
        "rule_night_high_amount","rule_new_country","rule_burst","rule_travel","rule_duplicate"
    ]].apply(lambda r: ",".join([c for c,v in r.items() if v]), axis=1)
    flagged["time_since_last_min"] = np.random.randint(1, 720, size=len(flagged))
    flagged["amount_z"] = np.round(np.random.normal(2.5, 0.8, size=len(flagged)), 2)
    flagged.rename(columns={"timestamp":"date"}, inplace=True)
    cols = ["transaction_id","customer_id","date","amount","type","score",
            "is_ml_flag","is_rule_flag","reasons_top3","triggered_rules",
            "time_since_last_min","amount_z","country"]
    return flagged[cols]

def demo_metrics_by_type() -> pd.DataFrame:
    return pd.DataFrame({
        "anomaly_type": ["MONTANT_ANORMAL","HORAIRE_SUSPECT","DUPLICATION","LOCALISATION_SUSPECTE","FREQUENCE_ANORMALE"],
        "precision": [0.67, 0.52, 0.58, 0.60, 0.62],
        "recall":    [0.72, 0.48, 0.51, 0.43, 0.40],
        "f1":        [0.69, 0.50, 0.54, 0.50, 0.49],
        "support":   [120,   80,   60,   40,   30],
        "tp":        [86, 38, 20, 26, 12],
        "fp":        [42, 35, 15, 17, 20],
        "fn":        [34, 42, 19, 34, 18],
        "tn":        [900, 900, 900, 900, 900],
    })

def demo_confusion() -> pd.DataFrame:
    tn, fp, fn, tp = 9500, 140, 108, 252
    acc = (tn+tp)/(tn+fp+fn+tp)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = 2*(prec*rec)/(prec+rec)
    prev = (tp+fn)/(tn+fp+fn+tp)
    return pd.DataFrame([{
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "prevalence": prev
    }])

def main():
    out = run_dir()  # ex: exports/run_2025-09-25_18-55-12
    print(f"[i] Exports -> {out}")

    df_tx = demo_transactions_scored()
    df_th = demo_thresholds()
    df_cs = demo_cases(df_tx)
    df_bt = demo_metrics_by_type()
    df_cm = demo_confusion()

    p1 = write_transactions_scored(df_tx, out)
    p2 = write_thresholds(df_th, out)
    p3 = write_cases(df_cs, out)
    p4 = write_metrics_by_type(df_bt, out)
    p5 = write_confusion(df_cm, out)

    runlog = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "rows": {
            "transactions_scored": len(df_tx),
            "thresholds": len(df_th),
            "cases": len(df_cs),
            "metrics_by_type": len(df_bt),
            "confusion_matrix": len(df_cm),
        },
        "outputs": { "tx": str(p1), "thresholds": str(p2), "cases": str(p3),
                     "by_type": str(p4), "confusion": str(p5) }
    }
    p6 = write_runlog(runlog, out)
    print("[ok] Fichiers écrits.")
    for k,v in runlog["outputs"].items():
        print(f" - {k}: {v}")
    print(f" - runlog: {p6}")

if __name__ == "__main__":
    main()
