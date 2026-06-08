"""
Service de données — charge le dataset réel agrégé (district x mois) ainsi que
les géométries GADM, et construit les fenêtres temporelles utilisées par le LSTM.

NOTE IMPORTANTE : le dataset réel construit à partir de MAP / ERA5 / MODIS / WorldPop
a une granularité MENSUELLE (2010-01 → 2022-12, 33 districts), et non hebdomadaire.
L'application est donc adaptée à cette granularité réelle : les "horizons" et
"fenêtres" exprimés ailleurs en semaines sont ici exprimés en MOIS.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
SHAPEFILE_PATH = BASE_DIR / "data" / "shapefiles" / "gadm41_CIV_2.shp"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Features utilisées en entrée du LSTM (ordre important — doit correspondre à l'entraînement)
FEATURES = [
    "t2m_c", "rh_pct", "tp_mm", "ndvi", "ndwi",
    "pfpr", "itn_use", "itn_access", "population",
]
TARGET = "incidence_rate_1k"
SEQ_LEN = 8  # longueur de la fenêtre temporelle d'entrée (mois)


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """Charge le dataset agrégé district x mois (mis en cache en mémoire)."""
    df = pd.read_parquet(PROCESSED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["district_id", "date"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def load_districts_geo() -> gpd.GeoDataFrame:
    """Charge les géométries des districts (niveau 2 GADM) en WGS84."""
    gdf = gpd.read_file(SHAPEFILE_PATH)
    gdf = gdf.rename(columns={"GID_2": "district_id", "NAME_2": "district_name", "NAME_1": "region"})
    gdf = gdf[["district_id", "district_name", "region", "geometry"]]
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def list_districts() -> list[dict]:
    """Retourne la liste des districts (id, nom, région) triée par nom."""
    gdf = load_districts_geo()
    return (
        gdf[["district_id", "district_name", "region"]]
        .sort_values("district_name")
        .to_dict(orient="records")
    )


def district_geometry(district_id: str) -> dict | None:
    gdf = load_districts_geo()
    row = gdf[gdf["district_id"] == district_id]
    if row.empty:
        return None
    return json.loads(row.to_json())["features"][0]["geometry"]


def all_geometries_geojson() -> dict:
    """GeoJSON FeatureCollection de tous les districts (sans données de prédiction)."""
    gdf = load_districts_geo()
    return json.loads(gdf.to_json())


def district_history(district_id: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Historique observé (incidence) pour un district, filtré sur une période optionnelle."""
    df = load_dataset()
    sub = df[df["district_id"] == district_id].copy()
    if start_date:
        sub = sub[sub["date"] >= pd.to_datetime(start_date)]
    if end_date:
        sub = sub[sub["date"] <= pd.to_datetime(end_date)]
    return sub[["date", "incidence_count", "incidence_rate_1k"] + FEATURES]


def latest_window(district_id: str, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, pd.Timestamp] | None:
    """
    Retourne (fenêtre [seq_len, n_features], date_de_référence) — les `seq_len` derniers
    mois disponibles pour le district, dans l'ordre chronologique. Sert d'entrée au LSTM
    pour produire la prédiction du mois suivant (et, par récurrence, de l'horizon demandé).
    """
    df = load_dataset()
    sub = df[df["district_id"] == district_id].sort_values("date")
    if len(sub) < seq_len:
        return None
    window = sub.iloc[-seq_len:]
    return window[FEATURES].to_numpy(dtype=np.float32), window["date"].iloc[-1]


def national_feature_means() -> pd.Series:
    """Moyennes nationales des features clés — utilisées pour le radar de comparaison."""
    df = load_dataset()
    return df[FEATURES].mean()


def district_feature_means(district_id: str) -> pd.Series:
    df = load_dataset()
    sub = df[df["district_id"] == district_id]
    return sub[FEATURES].mean()


def dataset_period() -> tuple[pd.Timestamp, pd.Timestamp]:
    df = load_dataset()
    return df["date"].min(), df["date"].max()
