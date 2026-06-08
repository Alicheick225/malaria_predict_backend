"""
Service de chargement et d'inférence du modèle LSTM entraîné sur les données
réelles (district x mois, 2010-2022). Calcule également le score et le niveau
de risque associés à chaque prédiction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from services import data_service

BASE_DIR = Path(__file__).resolve().parent.parent
SAVED_DIR = BASE_DIR / "ml" / "saved_model"
MODEL_PATH = SAVED_DIR / "lstm_malaria.keras"
SCALERS_PATH = SAVED_DIR / "scalers.npz"
METADATA_PATH = SAVED_DIR / "metadata.json"

# Seuils de classification du score de risque (rang percentile dans la distribution historique)
RISK_THRESHOLDS = (0.33, 0.66)
RISK_LABELS = ("Faible", "Modéré", "Élevé")


class ModelService:
    """Encapsule le modèle LSTM, ses normalisateurs (scalers) et le scoring du risque."""

    def __init__(self):
        self.model = None
        self.metadata: dict = {}
        self.x_min = self.x_max = self.y_min = self.y_max = None
        self._risk_distribution: np.ndarray | None = None
        self.loaded = False

    def load_model(self):
        """Charge le modèle entraîné. En son absence, bascule en mode 'données simulées'."""
        try:
            import tensorflow as tf  # import tardif : coûteux, inutile si modèle absent
            self.model = tf.keras.models.load_model(MODEL_PATH)
            scalers = np.load(SCALERS_PATH)
            self.x_min, self.x_max = scalers["x_min"], scalers["x_max"]
            self.y_min, self.y_max = scalers["y_min"], scalers["y_max"]
            with open(METADATA_PATH, encoding="utf-8") as f:
                self.metadata = json.load(f)
            df = data_service.load_dataset()
            self._risk_distribution = np.sort(df[data_service.TARGET].to_numpy())
            self.loaded = True
            print(f"[ModelService] Modèle LSTM chargé — version {self.metadata.get('version', '?')}")
        except Exception as e:
            print(f"[ModelService] Modèle indisponible ({e!r}) — mode données simulées activé")
            self.loaded = False

    @property
    def version(self) -> str:
        return self.metadata.get("version", "mock-0.0.0") if self.loaded else "mock-0.0.0"

    # ── Normalisation ──────────────────────────────────────────────────────────
    def _scale_x(self, window: np.ndarray) -> np.ndarray:
        span = np.where(self.x_max - self.x_min == 0, 1, self.x_max - self.x_min)
        return (window - self.x_min) / span

    def _inverse_y(self, scaled: float) -> float:
        return float(scaled * (self.y_max - self.y_min) + self.y_min)

    # ── Inférence ──────────────────────────────────────────────────────────────
    def predict(self, window: np.ndarray) -> float:
        """window : array [seq_len, n_features] non normalisé → taux d'incidence prédit (pour 1000 hab.)."""
        if not self.loaded:
            return self._mock_prediction(window)
        scaled = self._scale_x(window)[None, ...]
        pred_scaled = float(self.model.predict(scaled, verbose=0)[0, 0])
        return max(0.0, self._inverse_y(pred_scaled))

    @staticmethod
    def _mock_prediction(window: np.ndarray) -> float:
        """Prédiction de repli réaliste (utilisée tant que le modèle n'est pas entraîné)."""
        seed = int(abs(np.nansum(window)) * 1000) % (2**32)
        rng = np.random.default_rng(seed)
        rainfall_signal = float(np.nanmean(window[:, 2]))  # tp_mm — colonne précipitations
        base = 80 + rainfall_signal * 40
        return max(0.0, base * rng.uniform(0.85, 1.15))

    # ── Scoring du risque ──────────────────────────────────────────────────────
    def risk_score(self, predicted_rate: float) -> float:
        """Rang percentile de la prédiction dans la distribution historique nationale → score continu [0, 1]."""
        dist = self._risk_distribution
        if dist is None or len(dist) == 0:
            return float(np.clip(predicted_rate / 500.0, 0, 1))
        rank = np.searchsorted(dist, predicted_rate, side="right")
        return float(np.clip(rank / len(dist), 0, 1))

    @staticmethod
    def get_risk_level(score: float) -> str:
        low, high = RISK_THRESHOLDS
        if score < low:
            return RISK_LABELS[0]
        if score < high:
            return RISK_LABELS[1]
        return RISK_LABELS[2]


# Instance unique partagée par l'application (chargée au démarrage via le lifespan de FastAPI)
model_service = ModelService()
