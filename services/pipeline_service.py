"""
Pipeline d'inférence — pour chaque district, construit la fenêtre d'entrée du LSTM,
produit une prévision récursive sur l'horizon demandé, calcule les contributions
des features (attribution par gradient, approximation légère de SHAP) et persiste
le tout dans data/predictions/.

Granularité : MENSUELLE (le dataset réel agrège district x mois). Le champ
`week_predicted` désigne donc le 1er jour du mois prédit — nom conservé pour
rester cohérent avec le contrat d'API mais documenté ici pour éviter toute confusion.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from services import data_service
from services.model_service import model_service

PREDICTIONS_DIR = data_service.PREDICTIONS_DIR
DEFAULT_HORIZON = 8       # on génère systématiquement le plus grand horizon proposé (4/6/8)
                          # afin que toute sélection ultérieure soit servie depuis le cache
N_SHAP_BACKGROUND = 10


def _seasonal_climatology(district_id: str) -> pd.DataFrame:
    """Moyenne historique (2010-2022) de chaque feature par mois calendaire (1-12), pour un district."""
    df = data_service.load_dataset()
    sub = df[df["district_id"] == district_id]
    return sub.groupby(sub["date"].dt.month)[data_service.FEATURES].mean()


def _forecast_district(district_id: str, horizon: int) -> list[dict]:
    """
    Prévision récursive sur `horizon` mois.
    À chaque pas, le LSTM prédit le mois suivant à partir de la fenêtre des 8
    derniers mois connus. Les features exogènes (climat, végétation, couverture
    moustiquaires...) des mois futurs ne sont pas observées : on les approxime par
    la climatologie saisonnière du district (moyenne historique du même mois
    calendaire) — une hypothèse usuelle en prévision épidémiologique lorsqu'on ne
    dispose pas de prévisions climatiques détaillées à plusieurs mois.
    """
    win = data_service.latest_window(district_id)
    if win is None:
        return []
    window, last_date = win
    clim = _seasonal_climatology(district_id)

    cur_window = window.copy()
    cur_date = pd.Timestamp(last_date)
    out = []
    for _ in range(horizon):
        pred_rate = model_service.predict(cur_window)
        score = model_service.risk_score(pred_rate)
        next_date = cur_date + pd.offsets.MonthBegin(1)
        out.append({
            "district_id": district_id,
            "week_predicted": next_date.date().isoformat(),
            "cases_predicted": round(pred_rate, 3),
            "risk_score": round(score, 4),
            "risk_level": model_service.get_risk_level(score),
        })
        next_feats = clim.loc[next_date.month].to_numpy(dtype=np.float32)
        cur_window = np.vstack([cur_window[1:], next_feats])
        cur_date = next_date
    return out


def _compute_feature_contributions(districts: list[dict]) -> pd.DataFrame:
    """
    Calcule, pour la fenêtre la plus récente de chaque district, la contribution de
    chaque feature à la prédiction via un attributeur par gradient (Gradient × Input,
    implémenté avec shap.GradientExplainer — approximation différentiable de SHAP
    adaptée aux réseaux récurrents). Résultat mis en cache (un seul calcul par run).
    """
    cols = ["district_id", "feature", "value", "contribution"]
    if not model_service.loaded:
        return pd.DataFrame(columns=cols)

    windows, ids = [], []
    for d in districts:
        win = data_service.latest_window(d["district_id"])
        if win is None:
            continue
        windows.append(model_service._scale_x(win[0]))
        ids.append(d["district_id"])
    if not windows:
        return pd.DataFrame(columns=cols)

    X = np.stack(windows).astype(np.float32)  # [n_districts, seq_len, n_features]

    try:
        import shap
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(X), size=min(N_SHAP_BACKGROUND, len(X)), replace=False)
        explainer = shap.GradientExplainer(model_service.model, X[bg_idx])
        raw = explainer.shap_values(X)
        sv = np.array(raw)
        if sv.ndim == 4:                      # [n_districts, seq_len, n_features, n_outputs] (sortie scalaire)
            sv = sv[..., 0]
        contrib = np.abs(sv).sum(axis=1)      # somme |contribution| sur la fenêtre temporelle -> [n_districts, n_features]
    except Exception as e:
        print(f"[pipeline] SHAP indisponible ({e!r}) — contributions ignorées")
        return pd.DataFrame(columns=cols)

    rows = []
    for i, did in enumerate(ids):
        for j, feat in enumerate(data_service.FEATURES):
            rows.append({
                "district_id": did,
                "feature": feat,
                "value": float(X[i, -1, j]),
                "contribution": float(contrib[i, j]),
            })
    return pd.DataFrame(rows)


def run_pipeline(horizon: int = DEFAULT_HORIZON) -> dict:
    """Exécute le pipeline complet (prévisions + contributions) et persiste les résultats."""
    horizon = max(horizon, DEFAULT_HORIZON)
    districts = data_service.list_districts()

    rows: list[dict] = []
    for d in districts:
        rows.extend(_forecast_district(d["district_id"], horizon))

    df = pd.DataFrame(rows)
    if df.empty:
        return {"status": "echec", "nb_districts_processed": 0, "timestamp": date.today()}

    names = {d["district_id"]: d["district_name"] for d in districts}
    df["district_name"] = df["district_id"].map(names)
    df.to_csv(PREDICTIONS_DIR / f"predictions_{date.today().isoformat()}.csv", index=False)
    df.to_csv(PREDICTIONS_DIR / "latest.csv", index=False)

    shap_df = _compute_feature_contributions(districts)
    if not shap_df.empty:
        shap_df.to_csv(PREDICTIONS_DIR / "feature_contributions.csv", index=False)

    return {
        "status": "ok",
        "nb_districts_processed": int(df["district_id"].nunique()),
        "timestamp": date.today(),
    }


def latest_predictions(horizon: int = 6) -> pd.DataFrame:
    """Charge les prédictions les plus récentes, en régénérant le cache si besoin/insuffisant."""
    latest_path = PREDICTIONS_DIR / "latest.csv"
    needs_run = not latest_path.exists()
    if not needs_run:
        cached = pd.read_csv(latest_path, parse_dates=["week_predicted"])
        if cached.groupby("district_id")["week_predicted"].count().min() < horizon:
            needs_run = True
    if needs_run:
        run_pipeline(max(horizon, DEFAULT_HORIZON))

    df = pd.read_csv(latest_path, parse_dates=["week_predicted"])
    cutoff = sorted(df["week_predicted"].unique())[:horizon]
    return df[df["week_predicted"].isin(cutoff)].reset_index(drop=True)


def top_features(district_id: str, n: int = 5) -> list[dict]:
    """Retourne les `n` features qui contribuent le plus à la prédiction du district (cache SHAP)."""
    path = PREDICTIONS_DIR / "feature_contributions.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    sub = (
        df[df["district_id"] == district_id]
        .sort_values("contribution", ascending=False)
        .head(n)
    )
    return sub[["feature", "value", "contribution"]].to_dict(orient="records")
