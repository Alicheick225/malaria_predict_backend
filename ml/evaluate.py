"""
Backtesting multi-horizon — évalue la stratégie de prévision RÉCURSIVE utilisée en
production par `pipeline_service` sur le jeu de test (2021-2022, jamais vu à l'entraînement).

Pour chaque district et chaque mois de départ du test, génère une prévision récursive
sur MAX_HORIZON mois (mêmes hypothèses qu'en production : climatologie saisonnière
pour les variables exogènes futures non observées), et compare au taux d'incidence
réellement observé à chaque horizon h = 1..MAX_HORIZON.

Produit :
  - horizon_metrics.csv      RMSE / MAE / R² agrégés par horizon (1 à 8 mois)
  - backtest_predictions.csv détail observé/prédit par district, date et horizon
"""
from __future__ import annotations

import sys
from pathlib import Path

# IMPORTANT : importer TensorFlow AVANT pandas/pyarrow — sinon interblocage des
# pools de threads d'Arrow et de TensorFlow au démarrage (cf. ml/train.py).
import tensorflow as tf  # noqa: E402,F401

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.data_service import FEATURES, SEQ_LEN, TARGET, load_dataset  # noqa: E402
from services.model_service import model_service  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
SAVED_DIR = BASE_DIR / "saved_model"
TEST_START = pd.Timestamp("2021-01-01")
MAX_HORIZON = 8


def _recursive_forecast(window: np.ndarray, last_date: pd.Timestamp, clim: pd.DataFrame, horizon: int):
    cur_window = window.copy()
    cur_date = pd.Timestamp(last_date)
    preds = []
    for _ in range(horizon):
        pred = model_service.predict(cur_window)
        next_date = cur_date + pd.offsets.MonthBegin(1)
        preds.append((next_date, pred))
        next_feats = clim.loc[next_date.month].to_numpy(dtype=np.float32)
        cur_window = np.vstack([cur_window[1:], next_feats])
        cur_date = next_date
    return preds


def main():
    model_service.load_model()
    if not model_service.loaded:
        print("Modèle introuvable — lancez d'abord `python ml/train.py`.")
        return

    df = load_dataset()
    rows = []
    for did, sub in df.groupby("district_id"):
        sub = sub.sort_values("date").reset_index(drop=True)
        clim = sub.groupby(sub["date"].dt.month)[FEATURES].mean()
        feats = sub[FEATURES].to_numpy(dtype=np.float32)
        target = sub[TARGET].to_numpy(dtype=np.float32)
        dates = sub["date"]

        for i in range(SEQ_LEN, len(sub) - MAX_HORIZON + 1):
            if dates.iloc[i] < TEST_START:
                continue
            window = feats[i - SEQ_LEN:i]
            preds = _recursive_forecast(window, dates.iloc[i - 1], clim, MAX_HORIZON)
            for h, (_, pred_val) in enumerate(preds, start=1):
                obs_idx = i + h - 1
                if obs_idx >= len(sub):
                    break
                rows.append({
                    "district_id": did, "horizon": h,
                    "date": dates.iloc[obs_idx],
                    "observed": float(target[obs_idx]),
                    "predicted": pred_val,
                })

    res = pd.DataFrame(rows)
    if res.empty:
        print("Pas assez de données de test pour le backtesting multi-horizon.")
        return

    metrics = []
    for h, g in res.groupby("horizon"):
        rmse = float(np.sqrt(np.mean((g.observed - g.predicted) ** 2)))
        mae = float(np.mean(np.abs(g.observed - g.predicted)))
        ss_res = float(np.sum((g.observed - g.predicted) ** 2))
        ss_tot = float(np.sum((g.observed - g.observed.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        metrics.append({"horizon": int(h), "rmse": round(rmse, 3), "mae": round(mae, 3),
                         "r2": round(r2, 3), "n": int(len(g))})

    metrics_df = pd.DataFrame(metrics).sort_values("horizon")
    metrics_df.to_csv(SAVED_DIR / "horizon_metrics.csv", index=False)
    res.to_csv(SAVED_DIR / "backtest_predictions.csv", index=False)

    print("\nMétriques de backtesting par horizon (mois) :")
    print(metrics_df.to_string(index=False))
    print(f"\nRésultats sauvegardés dans {SAVED_DIR}")


if __name__ == "__main__":
    main()
