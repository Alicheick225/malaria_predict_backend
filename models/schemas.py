"""
Schémas Pydantic — contrats de données pour les requêtes/réponses de l'API.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


# ── Prédictions ────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    district_id: str
    district_name: str
    risk_score: float = Field(..., ge=0, le=1, description="Score de risque normalisé [0, 1]")
    risk_level: str = Field(..., description="Faible | Modéré | Élevé")
    cases_predicted: float
    week_predicted: date


class HistoryPoint(BaseModel):
    date: date
    cases_observed: Optional[float] = None
    cases_predicted: Optional[float] = None
    risk_level: Optional[str] = None


class FeatureHistoryPoint(BaseModel):
    """Point d'historique des variables explicatives observées (climat, végétation...)."""
    date: date
    t2m_c: float
    rh_pct: float
    tp_mm: float
    ndvi: float
    ndwi: float
    pfpr: float
    itn_use: float
    itn_access: float
    population: float


class PipelineStatus(BaseModel):
    status: str
    nb_districts_processed: int
    timestamp: date


# ── Districts ──────────────────────────────────────────────────────────────────
class DistrictSummary(BaseModel):
    district_id: str
    district_name: str
    region: Optional[str] = None
    geometry: dict = Field(..., description="Géométrie GeoJSON du district")
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    cases_predicted: Optional[float] = None


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class DistrictDetail(BaseModel):
    district_id: str
    district_name: str
    region: Optional[str] = None
    population: Optional[float] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    history: list[HistoryPoint] = []
    feature_history: list[FeatureHistoryPoint] = []
    feature_means: dict[str, float] = {}
    national_feature_means: dict[str, float] = {}
    top_features: list[FeatureContribution] = []


# ── Résumé / KPIs ──────────────────────────────────────────────────────────────
class PredictionsSummary(BaseModel):
    nb_districts_high: int
    nb_districts_moderate: int
    nb_districts_low: int
    trend_vs_previous_week: float = Field(..., description="Variation du score moyen vs semaine précédente (%)")
    last_update: date


# ── Health ─────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    last_prediction_date: Optional[date] = None
