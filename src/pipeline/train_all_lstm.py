# src/pipeline/train_all_lstm.py

import os
import json
import pandas as pd
import logging
from src.models.lstm_model import train_lstm, FORECAST_HORIZONS
from src.evaluation.lstm_metrics import compute_lstm_metrics
from src.models.lstm_model import (
    load_and_merge, normalize_data,
    temporal_train_val_test_split
)

logging.basicConfig(    
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

STATIONS = [
    'Sto Nino',
    'Tumana Bridge',
    'Rodriguez',
    'Nangka',
    'San Mateo-1',
    'Montalban',
    'Burgos',
]

WL_PATH  = 'data/processed/water_level_hourly.csv'
RF_PATH  = 'data/processed/rainfall_hourly.csv'
OUT_DIR  = 'models/lstm'
RESULTS  = []


def train_and_evaluate(station: str, horizon: int):
    log.info(f"\n{'='*60}")
    log.info(f"Station: {station} | Horizon: +{horizon}hr")
    log.info(f"{'='*60}")

    model, history, X_test, y_test, stats, test_df = train_lstm(
        station=station,
        horizon=horizon,
        wl_path=WL_PATH,
        rf_path=RF_PATH,
        model_dir=OUT_DIR,
    )

    import numpy as np
    y_pred = model.predict(X_test, verbose=0).flatten()

    result = compute_lstm_metrics(
        y_test, y_pred, stats, station, horizon
    )
    result['val_loss']     = round(min(history.history['val_loss']), 6)
    result['epochs']       = len(history.history['loss'])
    result['best_epoch']   = history.history['val_loss'].index(
                                 min(history.history['val_loss'])
                             ) + 1

    RESULTS.append(result)
    return result


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("FULL LSTM TRAINING — ALL STATIONS ALL HORIZONS")
    log.info(f"Stations : {STATIONS}")
    log.info(f"Horizons : {FORECAST_HORIZONS}")
    log.info(f"Total runs: {len(STATIONS) * len(FORECAST_HORIZONS)}")
    log.info("=" * 60)

    failed = []

    for station in STATIONS:
        for horizon in FORECAST_HORIZONS:
            try:
                result = train_and_evaluate(station, horizon)
                status = "ALL PASS" if result['all_pass'] else "NEEDS REVIEW"
                log.info(f"  → {station} h{horizon}: "
                         f"RMSE={result['rmse']} "
                         f"MAE={result['mae']} "
                         f"R²={result['r2']} | {status}")
            except Exception as e:
                log.error(f"FAILED: {station} h{horizon} | {e}")
                failed.append({'station': station, 'horizon': horizon,
                               'error': str(e)})

    # Save results summary
    os.makedirs('models', exist_ok=True)
    results_df = pd.DataFrame(RESULTS)
    results_df.to_csv('models/lstm_results_summary.csv', index=False)

    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE")
    log.info(f"Results saved: models/lstm_results_summary.csv")
    log.info(f"Failed runs  : {len(failed)}")
    log.info("=" * 60)

    log.info("\nFINAL RESULTS TABLE:")
    log.info("-" * 60)
    if RESULTS:
        for r in RESULTS:
            status = "PASS" if r['all_pass'] else "FAIL"
            log.info(
                f"  {r['station']:15} h{r['horizon']} | "
                f"RMSE={r['rmse']:.4f} "
                f"MAE={r['mae']:.4f} "
                f"R²={r['r2']:.4f} | {status}"
            )