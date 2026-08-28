# retrain_burgos.py
import logging
from src.models.lstm_model import train_lstm, FORECAST_HORIZONS
from src.evaluation.lstm_metrics import compute_lstm_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

for horizon in FORECAST_HORIZONS:
    model, history, X_test, y_test, stats, test_df = train_lstm(
        station='Burgos',
        horizon=horizon,
    )
    import numpy as np
    y_pred = model.predict(X_test, verbose=0).flatten()
    result = compute_lstm_metrics(y_test, y_pred, stats, 'Burgos', horizon)
    log.info(f"Burgos h{horizon}: RMSE={result['rmse']} "
             f"MAE={result['mae']} R²={result['r2']} | "
             f"{'PASS' if result['all_pass'] else 'FAIL'}")