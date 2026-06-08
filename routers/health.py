"""Route de healthcheck — état de l'API et du modèle chargé."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from models.schemas import HealthResponse
from services import pipeline_service
from services.model_service import model_service

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Statut de l'API et du modèle")
def get_health() -> HealthResponse:
    last_date = None
    latest_path = pipeline_service.PREDICTIONS_DIR / "latest.csv"
    if latest_path.exists():
        df = pd.read_csv(latest_path, parse_dates=["week_predicted"])
        if not df.empty:
            last_date = df["week_predicted"].min().date()

    return HealthResponse(
        status="ok",
        model_version=model_service.version,
        model_loaded=model_service.loaded,
        last_prediction_date=last_date,
    )
