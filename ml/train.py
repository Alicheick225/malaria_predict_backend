"""
Entraînement du modèle LSTM de prévision de l'incidence du paludisme.

Découpage STRICTEMENT CHRONOLOGIQUE (train < 2020-01, val = 2020, test >= 2021)
afin d'éviter toute fuite temporelle (le modèle ne doit jamais voir le futur
pendant l'apprentissage). Les normalisateurs (MinMaxScaler) sont ajustés
uniquement sur le train, pour la même raison.

Entrée  : fenêtre glissante de SEQ_LEN=8 mois x 9 features (climat, végétation,
          interventions, population) — par district.
Sortie  : taux d'incidence (incidence_rate_1k) du mois suivant.

Artefacts produits dans ml/saved_model/ :
  - lstm_malaria.keras      modèle entraîné
  - scalers.npz             bornes min/max des normalisateurs (X et y)
  - metadata.json           version, métriques de test, périodes des splits
  - training_history.csv    courbes de loss train/val par epoch
  - test_predictions.csv    observé vs prédit sur le jeu de test (pour le scatter)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# IMPORTANT : importer TensorFlow AVANT pandas/pyarrow. Dans l'ordre inverse, les
# pools de threads d'Arrow et de TensorFlow entrent en interblocage au démarrage
# (deadlock observé à la création des threads — `pthread_create` ne retourne jamais).
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.data_service import FEATURES, SEQ_LEN, TARGET, load_dataset  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
SAVED_DIR = BASE_DIR / "saved_model"
SAVED_DIR.mkdir(exist_ok=True)

SEED = 42
VAL_START  = pd.Timestamp("2020-01-01")   # train: 2010-2019 | val: 2020 | test: 2021-2022
TEST_START = pd.Timestamp("2021-01-01")   # données 2023-2024 dans la fenêtre d'inférence (post-dataset)
MODEL_VERSION = "2.0.0"


def build_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """Construit les fenêtres glissantes (X, y) par district, sans mélanger les districts."""
    X, y, dates, dids = [], [], [], []
    for did, sub in df.groupby("district_id"):
        sub = sub.sort_values("date").reset_index(drop=True)
        feats = sub[FEATURES].to_numpy(dtype=np.float32)
        target = sub[TARGET].to_numpy(dtype=np.float32)
        dts = sub["date"].to_numpy()
        for i in range(seq_len, len(sub)):
            X.append(feats[i - seq_len:i])
            y.append(target[i])
            dates.append(dts[i])
            dids.append(did)
    return np.array(X), np.array(y), pd.to_datetime(np.array(dates)), np.array(dids)


def minmax_fit(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return arr.min(axis=0), arr.max(axis=0)


def minmax_apply(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi - lo == 0, 1, hi - lo)
    return (arr - lo) / span


def minmax_inverse(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return arr * (hi - lo) + lo


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("Chargement du dataset agrégé district x mois...")
    df = load_dataset()
    X, y, dates, dids = build_sequences(df)
    print(f"Séquences construites : {X.shape[0]} (fenêtre={SEQ_LEN} mois, {X.shape[-1]} features)")

    train_mask = dates < VAL_START
    val_mask = (dates >= VAL_START) & (dates < TEST_START)
    test_mask = dates >= TEST_START
    print(f"  train: {train_mask.sum()}  |  val: {val_mask.sum()}  |  test: {test_mask.sum()}")

    # Normalisation — bornes apprises UNIQUEMENT sur le train (anti-fuite)
    n_feat = X.shape[-1]
    x_lo, x_hi = minmax_fit(X[train_mask].reshape(-1, n_feat))
    y_lo, y_hi = float(y[train_mask].min()), float(y[train_mask].max())

    def sx(arr):
        shp = arr.shape
        return minmax_apply(arr.reshape(-1, n_feat), x_lo, x_hi).reshape(shp)

    def sy(arr):
        return minmax_apply(arr, y_lo, y_hi)

    X_train, X_val, X_test = sx(X[train_mask]), sx(X[val_mask]), sx(X[test_mask])
    y_train, y_val, y_test = sy(y[train_mask]), sy(y[val_mask]), sy(y[test_mask])

    model = keras.Sequential([
        keras.layers.Input(shape=(SEQ_LEN, n_feat)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="lstm_malaria_ci")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    print("\nEntraînement...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    # ── Évaluation sur le jeu de test (2021-2022, jamais vu) ──────────────────
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()
    y_test_inv = minmax_inverse(y_test, y_lo, y_hi)
    y_pred_inv = minmax_inverse(y_pred_scaled, y_lo, y_hi)

    rmse = float(np.sqrt(np.mean((y_test_inv - y_pred_inv) ** 2)))
    mae = float(np.mean(np.abs(y_test_inv - y_pred_inv)))
    ss_res = float(np.sum((y_test_inv - y_pred_inv) ** 2))
    ss_tot = float(np.sum((y_test_inv - y_test_inv.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\nTest (2021-2022) — RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}")

    # ── Sauvegarde des artefacts ──────────────────────────────────────────────
    model.save(SAVED_DIR / "lstm_malaria.keras")
    np.savez(SAVED_DIR / "scalers.npz", x_min=x_lo, x_max=x_hi, y_min=np.array(y_lo), y_max=np.array(y_hi))

    meta = {
        "features": FEATURES,
        "target": TARGET,
        "seq_len": SEQ_LEN,
        "version": MODEL_VERSION,
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_metrics": {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)},
        "train_period": [str(dates[train_mask].min().date()), str(dates[train_mask].max().date())],
        "val_period": [str(dates[val_mask].min().date()), str(dates[val_mask].max().date())],
        "test_period": [str(dates[test_mask].min().date()), str(dates[test_mask].max().date())],
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_districts": int(df["district_id"].nunique()),
    }
    with open(SAVED_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    pd.DataFrame(history.history).to_csv(SAVED_DIR / "training_history.csv", index_label="epoch")
    pd.DataFrame({
        "date": dates[test_mask],
        "district_id": dids[test_mask],
        "observed": y_test_inv,
        "predicted": y_pred_inv,
    }).to_csv(SAVED_DIR / "test_predictions.csv", index=False)

    print(f"\nArtefacts sauvegardés dans {SAVED_DIR}")


if __name__ == "__main__":
    main()
