# src/data_generator.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.paths import run_dir
from src.export_utils import save_csv

def generate_transactions(n_customers=200, n_tx=8000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- customers ---
    customers = np.arange(1, n_customers+1)

    # --- timestamps ---
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(minutes=int(x)) for x in rng.integers(0, 60*24*90, n_tx)]  # 90 jours

    # --- base features ---
    df = pd.DataFrame({
        "transaction_id": np.arange(1, n_tx+1),
        "customer_id": rng.choice(customers, size=n_tx),
        "timestamp": timestamps,
        "amount": np.round(rng.normal(120, 40, size=n_tx).clip(1, None), 2),
        "country": rng.choice(["CH","FR","DE","IT","US"], size=n_tx, p=[.6,.1,.1,.1,.1]),
        "channel": rng.choice(["web","app","branch","atm"], size=n_tx, p=[.3,.3,.2,.2]),
        "device": rng.choice(["ios","android","pc","terminal"], size=n_tx),
        "merchant_category": rng.choice(["grocery","electronics","luxury","restaurant","travel","atm"], size=n_tx),
        "ip": [f"192.168.{rng.integers(0,255)}.{rng.integers(0,255)}" for _ in range(n_tx)]
    })

    # --- anomalies container ---
    df["is_anomaly"] = False
    df["anomaly_type"] = "normal"

    # --- inject anomalies ---
    anomalies_idx = []

    # 1) Montants extrêmes
    idx = rng.choice(df.index, size=int(0.5/100*n_tx), replace=False)
    df.loc[idx, "amount"] = df.loc[idx, "amount"] * rng.integers(10, 30, size=len(idx))
    df.loc[idx, ["is_anomaly","anomaly_type"]] = True, "HIGH_AMOUNT"
    anomalies_idx.extend(idx)

    # 2) Horaires nocturnes suspects
    idx = rng.choice(df.index, size=int(0.5/100*n_tx), replace=False)
    df.loc[idx, "timestamp"] = [ts.replace(hour=rng.integers(2,5), minute=rng.integers(0,60)) for ts in df.loc[idx,"timestamp"]]
    df.loc[idx, ["is_anomaly","anomaly_type"]] = True, "ODD_HOUR"
    anomalies_idx.extend(idx)

    # 3) Impossible travel
    idx = rng.choice(df.index, size=int(0.3/100*n_tx), replace=False)
    df.loc[idx, "country"] = "CN"  # voyage soudain en Chine
    df.loc[idx, ["is_anomaly","anomaly_type"]] = True, "IMPOSSIBLE_TRAVEL"
    anomalies_idx.extend(idx)

    # 4) Burst transactions (fréquence)
    burst_customers = rng.choice(customers, size=5, replace=False)
    for cust in burst_customers:
        idx = rng.choice(df.index[df.customer_id==cust], size=5, replace=False)
        base_time = datetime(2025, 2, 15, 12, 0)
        df.loc[idx, "timestamp"] = [base_time + timedelta(seconds=int(x)) for x in range(0, 5*10, 10)]
        df.loc[idx, ["is_anomaly","anomaly_type"]] = True, "BURST"
        anomalies_idx.extend(idx)

    # 5) Doublons rapides
    idx = rng.choice(df.index, size=int(0.3/100*n_tx), replace=False)
    df.loc[idx, ["amount","merchant_category"]] = [999.0,"electronics"]
    df.loc[idx, ["is_anomaly","anomaly_type"]] = True, "DUPLICATE"
    anomalies_idx.extend(idx)

    return df.sort_values("timestamp").reset_index(drop=True)

def main():
    out = run_dir("data")
    df = generate_transactions()
    p = save_csv(df, out / "transactions_synth.csv")
    print(f"[ok] Données générées: {p}")
    print(df.head())

if __name__ == "__main__":
    main()


