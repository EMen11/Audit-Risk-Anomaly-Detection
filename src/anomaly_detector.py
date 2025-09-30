# src/anomaly_detector.py
from __future__ import annotations

import argparse
import sys, platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)

from src.paths import exports_dir, run_dir
from src.export_utils import (
    write_transactions_scored, write_thresholds, write_cases,
    write_metrics_by_type, write_confusion, write_runlog
)

# ---------------------------------------------------------------------
# Helpers — localiser le dernier dataset généré (transactions_synth.csv)
# ---------------------------------------------------------------------
def find_latest_data_csv() -> Path | None:
    base = exports_dir()
    data_dirs = sorted([p for p in base.glob("data_*") if p.is_dir()])
    for d in reversed(data_dirs):  # plus récent d'abord
        cand = d / "transactions_synth.csv"
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------
# Feature engineering — variables utiles au modèle et aux règles
# ---------------------------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Variables temporelles
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    # Montants (stabilise les extrêmes)
    df["amount_log"] = np.log1p(df["amount"])

    # Temps depuis dernière transaction du même client (en secondes)
    df = df.sort_values(["customer_id", "timestamp"])
    df["time_since_last"] = (
        df.groupby("customer_id")["timestamp"]
          .diff().dt.total_seconds()
          .fillna(0)
    )

    # Z-score du montant par client (interprétabilité)
    g = df.groupby("customer_id")["amount"]
    df["amount_z"] = ((df["amount"] - g.transform("mean")) /
                      g.transform("std")).clip(-5, 5).fillna(0)

    # "Novelty" marchand par client (rare si <1% des achats du client)
    counts_client = df.groupby("customer_id").size().rename("client_tx_count")
    counts_pair = df.groupby(["customer_id", "merchant_category"]).size().rename("pair_tx_count")
    df = df.join(counts_client, on="customer_id").join(counts_pair, on=["customer_id", "merchant_category"])
    df["merchant_freq"] = (df["pair_tx_count"] / df["client_tx_count"]).fillna(0.0)
    df["merchant_novelty"] = df["merchant_freq"] <= 0.01

    return df


# ---------------------------------------------------------------------
# Règles métier — version crédible (historique, fenêtres glissantes)
# ---------------------------------------------------------------------
def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["customer_id", "timestamp"])

    # Montant élevé la nuit (0h–4h)
    df["rule_night_high_amount"] = (df["amount"] > 500) & (df["hour"].between(0, 4))

    # Nouveau pays — historique incrémental par client
    seen: dict[int, set] = {}
    flags = []
    for cid, country in zip(df["customer_id"], df["country"]):
        s = seen.setdefault(int(cid), set())
        flags.append(country not in s)
        s.add(country)
    df["rule_new_country"] = flags

    # Burst : >5 transactions en 10 minutes (fenêtre glissante par client)
    # ts_epoch en secondes
    ts_epoch = df["timestamp"].view("int64") // 10**9
    df["ts_epoch"] = ts_epoch
    df["burst_count_10m"] = (
        df.groupby("customer_id")["ts_epoch"]
          .transform(lambda x: x.rolling(window=600, min_periods=1).count())
    )
    df["rule_burst"] = df["burst_count_10m"] > 5

    # Voyage "impossible" (simplifié) : pays = CN (injecté dans le générateur)
    df["rule_travel"] = df["country"].eq("CN")

    # Doublon rapide : même (client, amount, merchant_category) en <= 5 minutes
    df = df.sort_values(["customer_id", "amount", "merchant_category", "timestamp"])
    dt = df.groupby(["customer_id", "amount", "merchant_category"])["timestamp"] \
           .diff().dt.total_seconds().fillna(1e9)
    df["rule_duplicate"] = dt.le(300)

    # Agrégat des règles
    rule_cols = [
        "rule_night_high_amount", "rule_new_country", "rule_burst",
        "rule_travel", "rule_duplicate"
    ]
    df["rule_count"] = df[rule_cols].sum(axis=1)
    df["is_rule_flag"] = df["rule_count"] > 0
    return df


# ---------------------------------------------------------------------
# Modèle — Isolation Forest (non supervisé)
# ---------------------------------------------------------------------
def fit_iforest(df: pd.DataFrame, seed: int = 42):
    feats = ["amount_log", "hour", "dayofweek", "time_since_last"]
    X = df[feats].fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.025,  # ≈ prévalence synthétique
        random_state=seed,
    )
    model.fit(X)

    # decision_function: plus petit = plus anormal → on inverse et normalise 0–1
    raw = -model.decision_function(X)
    score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)

    df = df.copy()
    df["score"] = score
    df["is_ml_flag"] = model.predict(X) == -1
    return df, model


# ---------------------------------------------------------------------
# Courbe de seuils → métriques + workload + coût (inclut règles)
# + Modes "lite"/"coverage" basés sur quantiles du score
# ---------------------------------------------------------------------
def compute_thresholds(df: pd.DataFrame, y_true: pd.Series, n_points: int = 25) -> pd.DataFrame:
    # seuils par quantiles du score (80% → 99.5%)
    qs = np.linspace(0.80, 0.995, n_points)
    ths = np.quantile(df["score"].to_numpy(), qs)
    ths = np.unique(np.round(ths, 3))

    rows = []
    p = float(y_true.mean())
    C_fp, C_fn = 1.0, 10.0

    for t in ths:
        # --- ML only ---
        y_pred_ml = (df["score"] >= t)
        prec_ml = precision_score(y_true, y_pred_ml, zero_division=0)
        rec_ml  = recall_score(y_true, y_pred_ml, zero_division=0)
        tn_ml, fp_ml, fn_ml, tp_ml = confusion_matrix(y_true, y_pred_ml).ravel()
        fpr_ml = fp_ml / (fp_ml + tn_ml + 1e-12)
        wl_ml  = float(y_pred_ml.mean() * 100)  # %

        # --- Hybrid (ML OR règles) ---
        y_pred_h = y_pred_ml | df["is_rule_flag"]
        prec_h = precision_score(y_true, y_pred_h, zero_division=0)
        rec_h  = recall_score(y_true, y_pred_h, zero_division=0)
        tn_h, fp_h, fn_h, tp_h = confusion_matrix(y_true, y_pred_h).ravel()
        fpr_h = fp_h / (fp_h + tn_h + 1e-12)
        wl_h  = float(y_pred_h.mean() * 100)    # %
        expected_cost_h = fpr_h*(1-p)*C_fp + (1-rec_h)*p*C_fn

        rows.append({
            "threshold": float(t),

            # ML-only
            "precision_ml": float(prec_ml),
            "recall_ml":    float(rec_ml),
            "fpr_ml":       float(fpr_ml),
            "workload_pct_ml": wl_ml,

            # Hybrid
            "precision_h": float(prec_h),
            "recall_h":    float(rec_h),
            "fpr_h":       float(fpr_h),
            "workload_pct_h": wl_h,
            "flagged_count_h": int(y_pred_h.sum()),
            "expected_cost_h": float(expected_cost_h),
        })

    return pd.DataFrame(rows)




# ---------------------------------------------------------------------
# Fiches cas — top N alertes + raisons (lisibles pour un auditeur)
# ---------------------------------------------------------------------
def build_cases(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    flagged = df[df["is_flag"]].copy().sort_values("score", ascending=False).head(top_n)

    # Explications rapides (et stables)
    flagged["reasons_top3"] = flagged.apply(lambda r: "; ".join([
        f"amount_z={r['amount_z']:.1f}",
        f"hour={int(r['hour'])}",
        f"new_country={bool(r['rule_new_country'])}",
        f"novel_merchant={bool(r['merchant_novelty'])}"
    ]), axis=1)

    # Règles déclenchées (liste)
    flagged["triggered_rules"] = flagged[[
        "rule_night_high_amount","rule_new_country","rule_burst",
        "rule_travel","rule_duplicate"
    ]].apply(lambda s: ",".join([c for c, v in s.items() if v]), axis=1)

    return flagged[[
        "transaction_id","customer_id","timestamp","amount","country","channel",
        "score","is_ml_flag","is_rule_flag","is_flag","reasons_top3","triggered_rules",
        "amount_z","merchant_novelty","hour"
    ]]


# ---------------------------------------------------------------------
# Métriques par type d’anomalie — pour prioriser les améliorations
# ---------------------------------------------------------------------
def metrics_by_type(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in sorted(df["anomaly_type"].unique()):
        if t == "normal":
            continue
        mask = df["anomaly_type"].eq(t)
        y_true = df.loc[mask, "is_anomaly"].astype(int)
        y_pred = df.loc[mask, "is_flag"].astype(bool)
        if y_true.empty:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        rows.append({
            "anomaly_type": t,
            "precision": float(prec),
            "recall":    float(rec),
            "f1":        float(f1),
            "support":   int(len(y_true)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main — charge les données, entraîne, applique règles, exporte
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default=None,
        help="Chemin vers exports/data_*/transactions_synth.csv (sinon auto-détection du plus récent)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default_threshold", type=float, default=0.85,
                        help="Seuil par défaut pour la confusion globale (score >= t OR règle)")
    args = parser.parse_args()

    # 1) Trouver/charger les données
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {data_path}")
    else:
        data_path = find_latest_data_csv()
        if data_path is None:
            # Générer à la volée si rien n'existe
            try:
                from src.data_generator import generate_transactions
                print("[i] Aucune donnée trouvée, génération en cours…")
                df_gen = generate_transactions(seed=args.seed)
                out_data = run_dir("data")
                data_path = out_data / "transactions_synth.csv"
                df_gen.to_csv(data_path, index=False)
                print(f"[ok] Données générées: {data_path}")
            except Exception as e:
                raise FileNotFoundError(
                    "Impossible de localiser ou générer transactions_synth.csv. "
                    "Passe un chemin avec --data."
                ) from e

    print(f"[i] Lecture données: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])

    # 2) Feature engineering
    df = feature_engineering(df)

    # 3) Modèle (Isolation Forest)
    df, model = fit_iforest(df, seed=args.seed)

    # 4) Règles métier
    df = apply_rules(df)

    # 5) Hybride ML + Règles
    df["is_flag"] = df["is_ml_flag"] | df["is_rule_flag"]

    # 6) Seuils & métriques (table thresholds)
    y_true = df["is_anomaly"].astype(int)
    df_th = compute_thresholds(df, y_true, n_points=25)

    # 7) Confusion globale (au seuil par défaut)
    t = float(np.round(args.default_threshold, 3))
    y_pred = (df["score"] >= t) | df["is_rule_flag"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    df_cm = pd.DataFrame([{
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": (tp+tn)/(tp+tn+fp+fn+1e-12),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "prevalence": float(y_true.mean()),
        "threshold_used": t
    }])

    # 8) Fiches cas & perfs par type
    df_cases  = build_cases(df, top_n=100)
    df_bytype = metrics_by_type(df)

    # 9) Exports horodatés
    out = run_dir("model")
    write_transactions_scored(df, out)
    write_thresholds(df_th, out)
    write_cases(df_cases, out)
    write_metrics_by_type(df_bytype, out)
    write_confusion(df_cm, out)
    write_runlog({
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": int(args.seed),
        "model": {"algo": "IsolationForest", "n_estimators": 200, "contamination": 0.025},
        "costs": {"C_fp": 1.0, "C_fn": 10.0},
        "data": {"path": str(data_path), "rows": int(len(df))},
        "exports": {
            "transactions_scored": str(out / "transactions_scored.csv"),
            "thresholds":          str(out / "thresholds.csv"),
            "cases":               str(out / "cases.csv"),
            "metrics_by_type":     str(out / "metrics_by_type.csv"),
            "confusion":           str(out / "confusion_matrix.csv"),
        }
    }, out)
    print(f"[ok] Exports -> {out}")


if __name__ == "__main__":
    main()
