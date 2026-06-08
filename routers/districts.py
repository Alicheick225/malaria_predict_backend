"""Routes liées aux districts : géométries, infos générales, détail et explicabilité."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from models.schemas import (
    DistrictDetail,
    DistrictSummary,
    FeatureContribution,
    FeatureHistoryPoint,
    HistoryPoint,
)
from services import data_service, pipeline_service

router = APIRouter(prefix="/api/v1/districts", tags=["districts"])


@router.get("", response_model=list[DistrictSummary], summary="Districts + géométries GeoJSON + dernière prédiction")
def get_districts(horizon: int = Query(6, ge=1, le=8)):
    geo = data_service.load_districts_geo()
    preds = pipeline_service.latest_predictions(horizon=horizon)
    first_week = sorted(preds["week_predicted"].unique())[0] if not preds.empty else None
    current = preds[preds["week_predicted"] == first_week] if first_week is not None else preds
    by_id = current.set_index("district_id")

    out = []
    for row in geo.itertuples():
        geometry = data_service.district_geometry(row.district_id)
        pred = by_id.loc[row.district_id] if row.district_id in by_id.index else None
        out.append(
            DistrictSummary(
                district_id=row.district_id,
                district_name=row.district_name,
                region=row.region,
                geometry=geometry,
                risk_score=float(pred["risk_score"]) if pred is not None else None,
                risk_level=str(pred["risk_level"]) if pred is not None else None,
                cases_predicted=float(pred["cases_predicted"]) if pred is not None else None,
            )
        )
    return out


@router.get("/{district_id}", response_model=DistrictDetail, summary="Détail complet d'un district")
def get_district_detail(district_id: str):
    geo = data_service.load_districts_geo()
    row = geo[geo["district_id"] == district_id]
    if row.empty:
        raise HTTPException(404, f"District inconnu : {district_id}")
    row = row.iloc[0]

    df = data_service.load_dataset()
    sub = df[df["district_id"] == district_id]
    population = float(sub["population"].iloc[-1]) if not sub.empty else None

    history = [
        HistoryPoint(date=r.date.date(), cases_observed=float(r.incidence_rate_1k))
        for r in sub.sort_values("date").itertuples()
    ]
    forecast = pipeline_service.latest_predictions(horizon=8)
    forecast = forecast[forecast["district_id"] == district_id].sort_values("week_predicted")
    history += [
        HistoryPoint(date=r.week_predicted.date(), cases_predicted=float(r.cases_predicted), risk_level=r.risk_level)
        for r in forecast.itertuples()
    ]

    feature_history = [
        FeatureHistoryPoint(date=r.date.date(), **{f: float(getattr(r, f)) for f in data_service.FEATURES})
        for r in sub.sort_values("date").itertuples()
    ]

    current = forecast.iloc[0] if not forecast.empty else None
    top = pipeline_service.top_features(district_id, n=5)

    return DistrictDetail(
        district_id=row["district_id"],
        district_name=row["district_name"],
        region=row["region"],
        population=population,
        risk_score=float(current["risk_score"]) if current is not None else None,
        risk_level=str(current["risk_level"]) if current is not None else None,
        history=history,
        feature_history=feature_history,
        feature_means=data_service.district_feature_means(district_id).to_dict(),
        national_feature_means=data_service.national_feature_means().to_dict(),
        top_features=[FeatureContribution(**f) for f in top],
    )
