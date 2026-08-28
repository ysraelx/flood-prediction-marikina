# src/evaluation/lstm_metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import json
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def compute_lstm_metrics(y_true_norm: np.ndarray,
                         y_pred_norm: np.ndarray,
                         stats: dict,
                         station: str,
                         horizon: int) -> dict:
    """
    Compute RMSE, MAE, R² in original meters scale.
    Thesis thresholds: RMSE < 0.50m, MAE < 0.35m, R² >= 0.85
    """
    # Inverse normalize back to meters
    wl_range = stats['wl_max'] - stats['wl_min']
    y_true = y_true_norm * wl_range + stats['wl_min']
    y_pred = y_pred_norm * wl_range + stats['wl_min']

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # Thesis thresholds
    rmse_pass = rmse < 0.50
    mae_pass  = mae  < 0.35
    r2_pass   = r2   >= 0.85

    results = {
        'station':   station,
        'horizon':   horizon,
        'rmse':      round(rmse, 4),
        'mae':       round(mae,  4),
        'r2':        round(r2,   4),
        'rmse_pass': rmse_pass,
        'mae_pass':  mae_pass,
        'r2_pass':   r2_pass,
        'all_pass':  rmse_pass and mae_pass and r2_pass,
    }

    log.info(f"\n{'─'*50}")
    log.info(f"Station : {station} | Horizon: +{horizon}hr")
    log.info(f"RMSE    : {rmse:.4f}m  {'PASS ✓' if rmse_pass else 'FAIL ✗'} (threshold: <0.50m)")
    log.info(f"MAE     : {mae:.4f}m   {'PASS ✓' if mae_pass  else 'FAIL ✗'} (threshold: <0.35m)")
    log.info(f"R²      : {r2:.4f}     {'PASS ✓' if r2_pass   else 'FAIL ✗'} (threshold: >=0.85)")
    log.info(f"Overall : {'ALL PASS' if results['all_pass'] else 'NEEDS REVIEW'}")

    return results


def evaluate_saved_model(station: str,
                         horizon: int,
                         model_dir: str = 'models/lstm',
                         wl_path: str = 'data/processed/water_level_hourly.csv',
                         rf_path: str = 'data/processed/rainfall_hourly.csv'):
    """
    Load a saved model and evaluate it on the test set.
    """
    from src.models.lstm_model import (
        load_and_merge,
        normalize_data,
        temporal_train_val_test_split,
        build_sequences,
    )

    safe_station = station.replace(' ', '_').replace('-', '_')
    model_name   = f"{safe_station}_h{horizon}"

    # Load model and stats
    model_path = os.path.join(model_dir, f"{model_name}_final.keras")
    stats_path = os.path.join(model_dir, f"{model_name}_stats.json")

    model = tf.keras.models.load_model(model_path)
    with open(stats_path) as f:
        stats = json.load(f)

    # Prepare test data
    df = load_and_merge(wl_path, rf_path, station)
    train_df, val_df, test_df = temporal_train_val_test_split(df)
    _, _         = normalize_data(train_df)
    test_df, _   = normalize_data(test_df, stats)
    X_test, y_test = build_sequences(test_df, horizon)

    # Predict
    y_pred = model.predict(X_test, verbose=0).flatten()

    return compute_lstm_metrics(y_test, y_pred, stats, station, horizon)


if __name__ == "__main__":
    # Evaluate the smoke test model
    results = evaluate_saved_model(
        station='Sto Nino',
        horizon=1,
    )
    print(f"\nResults: {results}")