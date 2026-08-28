# retrain_failing.py
import logging
import numpy as np
from src.models.lstm_model import train_lstm
from src.evaluation.lstm_metrics import compute_lstm_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Stations and horizons that failed excluding Burgos
FAILING = [
    ('Nangka',     3),
    ('Nangka',     6),
    ('San Mateo-1', 6),
    ('Montalban',  6),
]

all_results = []

for station, horizon in FAILING:
    log.info(f"\nRetraining: {station} h{horizon}")
    try:
        model, history, X_test, y_test, stats, test_df = train_lstm(
            station=station,
            horizon=horizon,
        )
        y_pred  = model.predict(X_test, verbose=0).flatten()
        result  = compute_lstm_metrics(
            y_test, y_pred, stats, station, horizon
        )
        status  = "PASS" if result['all_pass'] else "FAIL"
        log.info(f"  {station} h{horizon}: "
                 f"RMSE={result['rmse']} "
                 f"MAE={result['mae']} "
                 f"R²={result['r2']} | {status}")
        all_results.append(result)
    except Exception as e:
        log.error(f"  FAILED: {station} h{horizon} | {e}")

log.info("\n" + "=" * 50)
log.info("RETRAIN SUMMARY")
log.info("=" * 50)
for r in all_results:
    status = "PASS" if r['all_pass'] else "FAIL"
    log.info(f"  {r['station']:15} h{r['horizon']} | "
             f"RMSE={r['rmse']:.4f} "
             f"MAE={r['mae']:.4f} "
             f"R²={r['r2']:.4f} | {status}")