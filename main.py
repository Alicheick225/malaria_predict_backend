"""
MalariaWatch CI — API de prédiction des épidémies de paludisme en Côte d'Ivoire.

Architecture : FastAPI (backend) + Streamlit (frontend), modèle LSTM entraîné sur
des données réelles agrégées (MAP, ERA5/CDS, MODIS, WorldPop, GADM) à l'échelle
des 33 districts sanitaires, granularité mensuelle, période 2010-2022.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

# IMPORTANT : importer TensorFlow AVANT pandas/pyarrow/geopandas (chargés en cascade
# par data_service via les routers ci-dessous). Dans l'ordre inverse, les pools de
# threads d'Arrow et de TensorFlow entrent en interblocage au démarrage de l'API.
import tensorflow as tf  # noqa: E402,F401

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import districts, health, predictions
from services.model_service import model_service

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle LSTM (et ses scalers) une seule fois, au démarrage de l'API."""
    print("[startup] Chargement du modèle LSTM...")
    model_service.load_model()
    yield
    print("[shutdown] Arrêt de l'API MalariaWatch CI.")


app = FastAPI(
    title="MalariaWatch CI — API de prédiction du paludisme",
    description=(
        "API REST exposant les prédictions du modèle LSTM de prévision spatio-temporelle "
        "des épidémies de paludisme en Côte d'Ivoire (33 districts sanitaires, granularité "
        "mensuelle). Sources de données : Malaria Atlas Project, ERA5/Copernicus, "
        "MODIS (NDVI/NDWI), WorldPop, GADM."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

# CORS — autorise le frontend Streamlit (et un usage local sans restriction de port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(districts.router)


@app.get("/", tags=["health"], summary="Racine de l'API")
def root():
    return {
        "service": "MalariaWatch CI API",
        "version": app.version,
        "docs": "/docs",
        "health": f"{API_PREFIX}/health",
    }
