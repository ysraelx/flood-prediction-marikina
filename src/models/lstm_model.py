# src/models/lstm_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── Constants ──────────────────────────────────────────────────────────────────
LOOK_BACK         = 24
FORECAST_HORIZONS = [1, 3, 6]
LSTM_UNITS        = 64
DROPOUT_RATE      = 0.2
BATCH_SIZE        = 64
MAX_EPOCHS        = 100
PATIENCE          = 10

RF_TO_WL = {
    'Sto Nino':      'Marikina (Youth Camp)',
    'Tumana Bridge': 'Marikina (Youth Camp)',
    'Nangka':        'Nangka',
    'San Mateo-1':   'San Mateo-2',
    'Montalban':     'Sitio Wawa',
    'Burgos':        ['Sitio Wawa', 'Mt. Campana'],
    'Rodriguez':     ['Boso Boso', 'Macabud'],
}

# ── Data Preparation ───────────────────────────────────────────────────────────

def load_and_merge(wl_path: str, rf_path: str, station: str) -> pd.DataFrame:
    """
    Load water level and rainfall for one station.
    If multiple rainfall stations mapped, average their readings.
    """
    wl = pd.read_csv(wl_path, parse_dates=['datetime'])
    rf = pd.read_csv(rf_path, parse_dates=['datetime'])

    wl_s = wl[wl['station_name'] == station][
        ['datetime', 'water_level']
    ].copy()

    rf_stations = RF_TO_WL[station]

    # Handle single or multiple rainfall stations
    if isinstance(rf_stations, list):
        rf_parts = []
        for rf_st in rf_stations:
            part = rf[rf['station_name'] == rf_st][
                ['datetime', 'rainfall_mm']
            ].copy()
            part = part.rename(columns={'rainfall_mm': f'rf_{rf_st}'})
            rf_parts.append(part)

        rf_merged = rf_parts[0]
        for part in rf_parts[1:]:
            rf_merged = pd.merge(rf_merged, part, on='datetime', how='outer')

        # Average across all rainfall stations
        rf_cols = [c for c in rf_merged.columns if c.startswith('rf_')]
        rf_merged['rainfall_mm'] = rf_merged[rf_cols].mean(axis=1)
        rf_s = rf_merged[['datetime', 'rainfall_mm']].copy()
    else:
        rf_s = rf[rf['station_name'] == rf_stations][
            ['datetime', 'rainfall_mm']
        ].copy()

    merged = pd.merge(wl_s, rf_s, on='datetime', how='inner')
    merged = merged.sort_values('datetime').reset_index(drop=True)
    merged = merged.dropna()

    log.info(f"  {station}: {len(merged):,} hourly rows after merge")
    return merged


def normalize_data(df: pd.DataFrame, stats: dict = None):
    if stats is None:
        stats = {
            'wl_min': float(df['water_level'].min()),
            'wl_max': float(df['water_level'].max()),
            'rf_min': float(df['rainfall_mm'].min()),
            'rf_max': float(df['rainfall_mm'].max()),
        }

    df = df.copy()
    wl_range = stats['wl_max'] - stats['wl_min']
    rf_range  = stats['rf_max'] - stats['rf_min']

    df['wl_norm'] = (df['water_level'] - stats['wl_min']) / wl_range \
                    if wl_range > 0 else 0.0
    df['rf_norm'] = (df['rainfall_mm'] - stats['rf_min']) / rf_range \
                    if rf_range > 0 else 0.0

    return df, stats


def inverse_normalize(value: float, stats: dict) -> float:
    return value * (stats['wl_max'] - stats['wl_min']) + stats['wl_min']


def build_sequences(df: pd.DataFrame,
                    horizon: int,
                    look_back: int = LOOK_BACK):
    X, y = [], []
    values = df[['rf_norm', 'wl_norm']].values

    for i in range(look_back, len(values) - horizon):
        X.append(values[i - look_back:i])
        y.append(values[i + horizon - 1, 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def temporal_train_val_test_split(df: pd.DataFrame):
    """
    Date-based chronological split.
    Train : Jul 2024 – Jun 2025 (full first year, dry + rainy)
    Val   : Jul 2025 – Aug 2025 (2 months, start of rainy season)
    Test  : Sep 2025 – Jun 2026 (includes typhoon season 2025)

    This ensures the test set contains real Alert/Critical events
    from the 2025 typhoon season, making R² meaningful.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    train = df[df['datetime'] < '2025-07-01'].copy()
    val   = df[(df['datetime'] >= '2025-07-01') &
               (df['datetime'] <  '2025-09-01')].copy()
    test  = df[df['datetime'] >= '2025-09-01'].copy()

    log.info(f"  Train: {len(train):,} | "
             f"{train['datetime'].min().date()} to "
             f"{train['datetime'].max().date()}")
    log.info(f"  Val  : {len(val):,} | "
             f"{val['datetime'].min().date()} to "
             f"{val['datetime'].max().date()}")
    log.info(f"  Test : {len(test):,} | "
             f"{test['datetime'].min().date()} to "
             f"{test['datetime'].max().date()}")

    return train, val, test


# ── Custom Layer — serialization safe ─────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class LastTimestep(tf.keras.layers.Layer):
    """Extracts the last timestep from an LSTM sequence output."""
    def call(self, x):
        return x[:, -1, :]

    def get_config(self):
        return super().get_config()


# ── Model Architecture ─────────────────────────────────────────────────────────

def build_lstm_model(look_back: int = LOOK_BACK,
                     n_features: int = 2,
                     units: int = LSTM_UNITS,
                     dropout: float = DROPOUT_RATE) -> Model:
    """
    Stacked LSTM model.
    Both LSTM layers use return_sequences=True so hidden states
    from all time steps are available for augmentation (H_aug).
    LastTimestep extracts h(t) for the prediction output.
    """
    inputs = Input(shape=(look_back, n_features), name='input')

    x = LSTM(units, return_sequences=True, name='lstm_1')(inputs)
    x = Dropout(dropout, name='dropout_1')(x)

    x = LSTM(units, return_sequences=True, name='lstm_2')(x)
    x = Dropout(dropout, name='dropout_2')(x)

    last = LastTimestep(name='last_step')(x)
    out  = Dense(1, name='output')(last)

    model = Model(inputs=inputs, outputs=out, name='lstm_flood')
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def build_hidden_state_extractor(model: Model) -> Model:
    """
    Secondary model that returns full hidden state sequence
    from lstm_2. Output shape: (batch, look_back, 64)
    Used for augmentation step (H_aug construction).
    """
    return Model(
        inputs=model.input,
        outputs=model.get_layer('lstm_2').output,
        name='hidden_state_extractor'
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train_lstm(station: str,
               horizon: int,
               wl_path: str = 'data/processed/water_level_hourly.csv',
               rf_path: str = 'data/processed/rainfall_hourly.csv',
               model_dir: str = 'models/lstm'):

    os.makedirs(model_dir, exist_ok=True)
    safe_station = station.replace(' ', '_').replace('-', '_')
    model_name   = f"{safe_station}_h{horizon}"

    log.info(f"\n{'='*60}")
    log.info(f"Training LSTM: {station} | Horizon: +{horizon}hr")
    log.info(f"{'='*60}")

    df                       = load_and_merge(wl_path, rf_path, station)
    train_df, val_df, test_df = temporal_train_val_test_split(df)

    train_df, stats = normalize_data(train_df)
    val_df,   _     = normalize_data(val_df,   stats)
    test_df,  _     = normalize_data(test_df,  stats)

    stats_path = os.path.join(model_dir, f"{model_name}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    X_train, y_train = build_sequences(train_df, horizon)
    X_val,   y_val   = build_sequences(val_df,   horizon)
    X_test,  y_test  = build_sequences(test_df,  horizon)

    log.info(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    log.info(f"  X_val  : {X_val.shape}   | y_val  : {y_val.shape}")
    log.info(f"  X_test : {X_test.shape}  | y_test : {y_test.shape}")

    model = build_lstm_model()
    model.summary(print_fn=log.info)

    checkpoint_path = os.path.join(model_dir, f"{model_name}_best.keras")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    log.info("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = os.path.join(model_dir, f"{model_name}_final.keras")
    model.save(final_path)
    log.info(f"Model saved: {final_path}")

    return model, history, X_test, y_test, stats, test_df


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, history, X_test, y_test, stats, test_df = train_lstm(
        station='Sto Nino',
        horizon=1,
    )
    log.info("Smoke test complete.")
    log.info(f"Final val_loss: {min(history.history['val_loss']):.6f}")
    log.info(f"Epochs trained: {len(history.history['loss'])}")