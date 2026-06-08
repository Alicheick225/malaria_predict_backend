"""Routes exposant les artefacts de performance du modèle LSTM entraîné (métriques, courbes, SHAP)."""
from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter

from services import data_service
from services.model_service import SAVED_DIR

router = APIRouter(prefix="/api/v1/model", tags=["model"])

CONTRIBUTIONS_PATH = data_service.PREDICTIONS_DIR / "feature_contributions.csv"


def _read_csv_records(path) -> list[dict] | None:
    if not path.exists():
        return None
    return pd.read_csv(path).to_dict(orient="records")


@router.get("/info", summary="Métadonnées et artefacts de performance du modèle entraîné")
def get_model_info():
    """
    Agrège en une seule réponse les artefacts générés par `ml/train.py` et `ml/evaluate.py`
    (métadonnées d'entraînement, métriques par horizon, prédictions du jeu de test,
    historique d'apprentissage) ainsi que les contributions globales des features (SHAP).

    Le frontend est déployé séparément (Streamlit Cloud) et n'a donc pas accès au système
    de fichiers du backend : ces artefacts doivent transiter par l'API plutôt que d'être
    lus directement sur disque.
    """
    metadata_path = SAVED_DIR / "metadata.json"
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "metadata": metadata,
        "horizon_metrics": _read_csv_records(SAVED_DIR / "horizon_metrics.csv"),
        "test_predictions": _read_csv_records(SAVED_DIR / "test_predictions.csv"),
        "training_history": _read_csv_records(SAVED_DIR / "training_history.csv"),
        "feature_contributions": _read_csv_records(CONTRIBUTIONS_PATH),
    }
