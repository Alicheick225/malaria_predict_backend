"""Routes liées aux prédictions du modèle LSTM (niveau national / par district)."""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from models.schemas import HistoryPoint, PipelineStatus, PredictionResponse, PredictionsSummary
from services import data_service, pipeline_service

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


@router.get("/latest", response_model=list[PredictionResponse], summary="Dernières prédictions pour tous les districts")
def get_latest_predictions(
    horizon: int = Query(6, ge=1, le=8, description="Horizon de prévision (mois)"),
    semaine: Optional[str] = Query(None, description="Filtrer sur un mois prédit précis (YYYY-MM-DD)"),
):
    df = pipeline_service.latest_predictions(horizon=horizon)
    if semaine:
        try:
            target = date.fromisoformat(semaine)
        except ValueError:
            raise HTTPException(400, "Format de date invalide — attendu YYYY-MM-DD")
        df = df[df["week_predicted"].dt.date == target]

    return [
        PredictionResponse(
            district_id=r.district_id,
            district_name=r.district_name,
            risk_score=r.risk_score,
            risk_level=r.risk_level,
            cases_predicted=r.cases_predicted,
            week_predicted=r.week_predicted.date(),
        )
        for r in df.itertuples()
    ]


@router.get(
    "/summary",
    response_model=PredictionsSummary,
    summary="KPIs globaux : répartition des risques, tendance, dernière mise à jour",
)
def get_predictions_summary(horizon: int = Query(6, ge=1, le=8)):
    df = pipeline_service.latest_predictions(horizon=horizon)
    if df.empty:
        raise HTTPException(503, "Aucune prédiction disponible")

    weeks = sorted(df["week_predicted"].unique())
    first_week, second_week = weeks[0], (weeks[1] if len(weeks) > 1 else weeks[0])

    cur = df[df["week_predicted"] == first_week]
    prev = df[df["week_predicted"] == second_week]

    counts = cur["risk_level"].value_counts()
    trend = 0.0
    if prev["risk_score"].mean():
        trend = float((cur["risk_score"].mean() - prev["risk_score"].mean()) / prev["risk_score"].mean() * 100)

    return PredictionsSummary(
        nb_districts_high=int(counts.get("Élevé", 0)),
        nb_districts_moderate=int(counts.get("Modéré", 0)),
        nb_districts_low=int(counts.get("Faible", 0)),
        trend_vs_previous_week=round(trend, 2),
        last_update=date.today(),
    )


@router.get(
    "/{district_id}/history",
    response_model=list[HistoryPoint],
    summary="Historique observé + prédit pour un district",
)
def get_district_history(
    district_id: str,
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    districts = {d["district_id"] for d in data_service.list_districts()}
    if district_id not in districts:
        raise HTTPException(404, f"District inconnu : {district_id}")

    observed = data_service.district_history(district_id, start_date, end_date)
    points = [
        HistoryPoint(date=r.date.date(), cases_observed=float(r.incidence_rate_1k), cases_predicted=None, risk_level=None)
        for r in observed.itertuples()
    ]

    forecast = pipeline_service.latest_predictions(horizon=8)
    forecast = forecast[forecast["district_id"] == district_id]
    points += [
        HistoryPoint(
            date=r.week_predicted.date(),
            cases_observed=None,
            cases_predicted=float(r.cases_predicted),
            risk_level=r.risk_level,
        )
        for r in forecast.itertuples()
    ]
    return points


@router.post(
    "/run",
    response_model=PipelineStatus,
    summary="Déclenche manuellement le pipeline d'inférence complet",
)
def run_prediction_pipeline(horizon: int = Query(8, ge=1, le=8)):
    result = pipeline_service.run_pipeline(horizon=horizon)
    if result["status"] != "ok":
        raise HTTPException(500, "Échec du pipeline d'inférence")
    return PipelineStatus(**result)
