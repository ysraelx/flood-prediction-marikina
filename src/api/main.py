# src/api/main.py

import os
import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

from src.models.lstm_model import (
    LastTimestep,
    build_hidden_state_extractor,
    load_and_merge,
    normalize_data,
    build_sequences,
    LOOK_BACK,
)
from src.models.augmentation import (
    compute_h_aug,
    compute_r_acc,
    compute_roc,
    normalize_r_acc,
    normalize_roc,
    get_static_features,
    get_roc_range,
)
from src.models.lstm_model import temporal_train_val_test_split
from src.data.preprocessor import assign_risk_class, THRESHOLDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
STATIONS = [
    'Sto Nino', 'Tumana Bridge', 'Rodriguez', 'Nangka',
    'San Mateo-1', 'Montalban', 'Burgos',
]
HORIZONS = [1, 3, 6]

LSTM_DIR = 'models/lstm'
PROD_DIR = 'models/rf/production'
WL_PATH = 'data/processed/water_level_hourly.csv'
RF_PATH = 'data/processed/rainfall_hourly.csv'
N_PCA_COMPONENTS = 10

app = FastAPI(
    title="Marikina Flood Prediction API",
    description="Real-time flood prediction and hotspot mapping "
                 "for Marikina River barangays",
    version="1.0.0",
)

# ── In-memory model cache — loaded once at startup, reused per request ────────
MODEL_CACHE = {}   # key: (station, horizon) -> dict of loaded artifacts
LATEST_PREDICTIONS = {}   # key: (station, horizon) -> prediction dict
LATEST_UPDATE_TIME: Optional[datetime] = None


def safe_name(station: str) -> str:
    return station.replace(' ', '_').replace('-', '_')


def load_all_models():
    """
    Loads every LSTM model + extractor + stats + roc_range, plus every
    production PCA/RF pair, once, into MODEL_CACHE. Called at API startup.
    """
    log.info("Loading all 21 station-horizon models into memory...")
    for station in STATIONS:
        for horizon in HORIZONS:
            safe_station = safe_name(station)
            model_name = f"{safe_station}_h{horizon}"

            lstm_path = os.path.join(LSTM_DIR, f"{model_name}_final.keras")
            stats_path = os.path.join(LSTM_DIR, f"{model_name}_stats.json")
            pca_path = os.path.join(PROD_DIR, f"{model_name}_pca.joblib")
            rf_path = os.path.join(PROD_DIR, f"{model_name}_rf.joblib")

            if not all(os.path.exists(p) for p in
                       [lstm_path, stats_path, pca_path, rf_path]):
                log.warning(f"  Skipping {station} h{horizon} — missing model files")
                continue

            lstm = tf.keras.models.load_model(
                lstm_path, custom_objects={'LastTimestep': LastTimestep}
            )
            extractor = build_hidden_state_extractor(lstm)
            with open(stats_path) as f:
                stats = json.load(f)
            pca = joblib.load(pca_path)
            rf = joblib.load(rf_path)

            # ROC range needs the training split — computed once at load time
            df = load_and_merge(WL_PATH, RF_PATH, station)
            train_df, _, _ = temporal_train_val_test_split(df)
            roc_min, roc_max = get_roc_range(train_df)

            MODEL_CACHE[(station, horizon)] = {
                'lstm': lstm,
                'extractor': extractor,
                'stats': stats,
                'pca': pca,
                'rf': rf,
                'roc_min': roc_min,
                'roc_max': roc_max,
            }
            log.info(f"  Loaded {station} h{horizon}")

    log.info(f"Model loading complete: {len(MODEL_CACHE)}/21 combinations ready")


def run_single_prediction(station: str, horizon: int) -> dict:
    """
    Runs the full pipeline for ONE station-horizon using the most recent
    LOOK_BACK hours of data: LSTM prediction -> H_aug -> PCA -> RF risk class.
    """
    cache = MODEL_CACHE.get((station, horizon))
    if cache is None:
        raise ValueError(f"No cached model for {station} h{horizon}")

    df = load_and_merge(WL_PATH, RF_PATH, station)
    df_norm, _ = normalize_data(df, cache['stats'])

    if len(df_norm) < LOOK_BACK + horizon:
        raise ValueError(f"Not enough recent data for {station} h{horizon}")

    # Most recent LOOK_BACK-hour window
    recent = df_norm.tail(LOOK_BACK)
    X = recent[['rf_norm', 'wl_norm']].values.reshape(1, LOOK_BACK, 2).astype(np.float32)
    last_datetime = recent['datetime'].iloc[-1]

    hidden_states = cache['extractor'].predict(X, verbose=0)     # (1, 24, 64)
    w_hat_norm = cache['lstm'].predict(X, verbose=0).flatten()[0]

    h_aug = compute_h_aug(hidden_states)[0]                       # (64,)

    rainfall_norm_seq = X[:, :, 0]
    wl_norm_seq = X[:, :, 1]

    r_acc_raw = compute_r_acc(rainfall_norm_seq)[0]
    roc_raw = compute_roc(wl_norm_seq)[0]

    r_acc = normalize_r_acc(np.array([r_acc_raw]), cache['stats'])[0]
    roc = normalize_roc(np.array([roc_raw]), cache['stats'],
                         cache['roc_min'], cache['roc_max'])[0]

    static_feats = get_static_features(station)

    scalars = np.array([w_hat_norm, r_acc, roc, *static_feats], dtype=np.float32)
    h_aug_pca = cache['pca'].transform(h_aug.reshape(1, -1))[0]

    X_final = np.concatenate([h_aug_pca, scalars]).reshape(1, -1)

    risk_class = cache['rf'].predict(X_final)[0]
    risk_proba = cache['rf'].predict_proba(X_final)[0]
    class_labels = cache['rf'].classes_

    proba_dict = {cls: float(p) for cls, p in zip(class_labels, risk_proba)}
    for cls in ['Normal', 'Alert', 'Critical']:
        proba_dict.setdefault(cls, 0.0)

    # Predicted water level in meters, for display
    from src.models.lstm_model import inverse_normalize
    w_hat_meters = inverse_normalize(float(w_hat_norm), cache['stats'])

    target_time = pd.to_datetime(last_datetime) + pd.Timedelta(hours=horizon)

    return {
        'station': station,
        'horizon': horizon,
        'as_of': str(last_datetime),
        'target_time': str(target_time),
        'predicted_water_level_m': round(w_hat_meters, 3),
        'risk_class': risk_class,
        'probabilities': {
            'Normal': round(proba_dict['Normal'], 4),
            'Alert': round(proba_dict['Alert'], 4),
            'Critical': round(proba_dict['Critical'], 4),
        },
        'hotspot_severity_score': round(
            proba_dict['Alert'] + 2 * proba_dict['Critical'], 4
        ),
    }


def run_all_predictions():
    """Runs predictions for all 21 station-horizon combos, updates the cache."""
    global LATEST_UPDATE_TIME
    log.info("Running scheduled prediction sweep...")
    count = 0
    for station in STATIONS:
        for horizon in HORIZONS:
            try:
                result = run_single_prediction(station, horizon)
                LATEST_PREDICTIONS[(station, horizon)] = result
                count += 1
            except Exception as e:
                log.error(f"  Prediction failed for {station} h{horizon}: {e}")
    LATEST_UPDATE_TIME = datetime.now()
    log.info(f"Prediction sweep complete: {count}/21 succeeded at {LATEST_UPDATE_TIME}")


# ── Scheduler — hourly automated predictions ──────────────────────────────────
scheduler = BackgroundScheduler()


@app.on_event("startup")
def startup_event():
    load_all_models()
    run_all_predictions()   # populate cache immediately, don't wait an hour
    scheduler.add_job(run_all_predictions, 'interval', hours=1, id='hourly_predict')
    scheduler.start()
    log.info("Scheduler started — predictions will refresh hourly")


@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()


# ── Request/response models ────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    station: str
    horizon: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        'status': 'ok',
        'models_loaded': len(MODEL_CACHE),
        'expected_models': len(STATIONS) * len(HORIZONS),
        'last_prediction_sweep': str(LATEST_UPDATE_TIME) if LATEST_UPDATE_TIME else None,
    }


@app.get("/latest")
def latest(station: Optional[str] = None):
    """
    Returns the most recent predictions for all stations/horizons, or
    filtered to one station if `station` query param is given.
    """
    if not LATEST_PREDICTIONS:
        raise HTTPException(status_code=503, detail="No predictions available yet")

    results = []
    for (st, hz), pred in LATEST_PREDICTIONS.items():
        if station and st != station:
            continue
        results.append(pred)

    if station and not results:
        raise HTTPException(status_code=404, detail=f"No data for station '{station}'")

    return {
        'last_updated': str(LATEST_UPDATE_TIME),
        'count': len(results),
        'predictions': results,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Runs a fresh prediction on demand for one station/horizon
    (bypasses the hourly cache — always uses the latest available data).
    """
    if req.station not in STATIONS:
        raise HTTPException(status_code=400,
                             detail=f"Unknown station. Valid: {STATIONS}")
    if req.horizon not in HORIZONS:
        raise HTTPException(status_code=400,
                             detail=f"Unknown horizon. Valid: {HORIZONS}")

    try:
        result = run_single_prediction(req.station, req.horizon)
        return result
    except Exception as e:
        log.error(f"Predict failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)    