# src/models/augmentation.py

import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os
import logging

from src.models.lstm_model import (
    LastTimestep,
    build_hidden_state_extractor,
    load_and_merge,
    normalize_data,
    temporal_train_val_test_split,
    build_sequences,
    LOOK_BACK,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
LAMBDA          = 0.5     # temporal decay rate for hidden state weighting
N_WEIGHTED_STEPS = 6      # h(t-5) ... h(t)
ROC_MARGIN       = 1.1    # safety margin applied to observed train-set ROC range,
                          # so a slightly larger val/test swing doesn't get clipped

# ── Static barangay attributes ────────────────────────────────────────────────
# Estimated from Google Earth / PAGASA station elevation data; flood_frequency
# derived from the 2-year dataset. No formal GIS data collected yet — documented
# in Chapter 3 as a limitation pending GIS collection.
BARANGAY_ATTRS = {
    'Sto Nino': {
        'elevation':         12.0,
        'distance_to_river': 0.05,
        'flood_frequency':   0.85,
    },
    'Tumana Bridge': {
        'elevation':         13.0,
        'distance_to_river': 0.05,
        'flood_frequency':   0.80,
    },
    'Rodriguez': {
        'elevation':         27.0,
        'distance_to_river': 0.10,
        'flood_frequency':   0.45,
    },
    'Nangka': {
        'elevation':         15.0,
        'distance_to_river': 0.08,
        'flood_frequency':   0.70,
    },
    'San Mateo-1': {
        'elevation':         14.0,
        'distance_to_river': 0.10,
        'flood_frequency':   0.60,
    },
    'Montalban': {
        'elevation':         21.0,
        'distance_to_river': 0.12,
        'flood_frequency':   0.50,
    },
    'Burgos': {
        'elevation':         26.0,
        'distance_to_river': 0.15,
        'flood_frequency':   0.40,
    },
}

# Normalization ranges for static attributes, derived from BARANGAY_ATTRS itself
# (min-max across the 7 stations) so RF features stay in a comparable [0,1] range.
_ELEV_VALUES = [v['elevation'] for v in BARANGAY_ATTRS.values()]
_DIST_VALUES = [v['distance_to_river'] for v in BARANGAY_ATTRS.values()]
ELEV_MIN, ELEV_MAX = min(_ELEV_VALUES), max(_ELEV_VALUES)
DIST_MIN, DIST_MAX = min(_DIST_VALUES), max(_DIST_VALUES)
# flood_frequency is already a 0-1 probability, no normalization needed


# ── Temporal weights (Eq. 3.8b) ───────────────────────────────────────────────

def compute_temporal_weights(n_steps: int = N_WEIGHTED_STEPS,
                              lam: float = LAMBDA) -> np.ndarray:
    """
    w_j = exp(lam * j) / sum(exp(lam * j))  for j = 1..n_steps
    Most recent hour (j = n_steps) gets the highest weight.
    """
    j = np.arange(1, n_steps + 1, dtype=np.float64)
    raw = np.exp(lam * j)
    weights = raw / raw.sum()
    return weights.astype(np.float32)


TEMPORAL_WEIGHTS = compute_temporal_weights()


# ── Step 1-3: H_aug construction ──────────────────────────────────────────────

def compute_h_aug(hidden_states: np.ndarray,
                   n_steps: int = N_WEIGHTED_STEPS,
                   weights: np.ndarray = None) -> np.ndarray:
    """
    hidden_states: (batch, look_back, units) — full h(t) sequence from lstm_2
    Uses the last n_steps timesteps: h(t-5) ... h(t)
    Returns: (batch, units) weighted trajectory vector H_aug
    """
    if weights is None:
        weights = TEMPORAL_WEIGHTS

    last_steps = hidden_states[:, -n_steps:, :]  # (batch, n_steps, units)
    # weighted sum over the timestep axis
    h_aug = np.einsum('btu,t->bu', last_steps, weights)
    return h_aug.astype(np.float32)


# ── Step 4: scalar features ───────────────────────────────────────────────────

def compute_r_acc(rainfall_norm: np.ndarray, window: int = 6) -> np.ndarray:
    """
    6-hour accumulated rainfall, computed on the normalized rainfall series
    already used as LSTM input (last `window` steps of each sequence).
    rainfall_norm: (batch, look_back) — the rf_norm channel of X
    Returns: (batch,) sum of last `window` normalized rainfall values.
    """
    last_window = rainfall_norm[:, -window:]
    return last_window.sum(axis=1).astype(np.float32)


def compute_roc(water_level_norm: np.ndarray) -> np.ndarray:
    """
    Rate of change of water level between the last two look-back timesteps,
    in normalized units per hour. water_level_norm: (batch, look_back)
    """
    roc = water_level_norm[:, -1] - water_level_norm[:, -2]
    return roc.astype(np.float32)


def normalize_r_acc(r_acc_norm_sum: np.ndarray, stats: dict) -> np.ndarray:
    """
    R_acc is a sum of already-normalized rainfall values (0-1 each over
    `window` steps), so it can exceed 1. Rescale it back to millimeters using
    rf_min/rf_max, then re-normalize to [0,1] over the same station rf range
    times the accumulation window, per the agreed approach: reuse rf_min/rf_max
    from the station's stats JSON.
    """
    rf_range = stats['rf_max'] - stats['rf_min']
    r_acc_mm = r_acc_norm_sum * rf_range  # approx accumulated mm over the window
    max_possible = rf_range * 6 if rf_range > 0 else 1.0
    r_acc_final = r_acc_mm / max_possible if max_possible > 0 else r_acc_mm * 0.0
    return np.clip(r_acc_final, 0.0, 1.0).astype(np.float32)


def get_roc_range(train_df: pd.DataFrame, margin: float = ROC_MARGIN) -> tuple:
    """
    Data-derived ROC normalization range for one station, computed from the
    TRAINING split's actual hour-to-hour water level changes (m/hr) — same
    pattern as wl_min/max and rf_min/max. A safety margin is applied so
    val/test swings slightly beyond the training range don't get clipped
    to exactly 0 or 1.

    Replaces a fixed -5..+5 m/hr range, which was far wider than any
    station's real dynamics and left ROC compressed near 0.5 for genuine
    flood events (verified on a Sto Nino typhoon-season spot check).
    """
    wl_sorted = train_df.sort_values('datetime')
    deltas = wl_sorted['water_level'].diff().dropna()
    roc_min = float(deltas.min()) * margin
    roc_max = float(deltas.max()) * margin
    return roc_min, roc_max


def normalize_roc(roc_norm_units: np.ndarray,
                   stats: dict,
                   roc_min: float,
                   roc_max: float) -> np.ndarray:
    """
    Convert normalized-water-level rate of change back to m/hr using the
    station's wl range, then normalize to [0,1] using the station's own
    data-derived roc_min/roc_max (see get_roc_range).
    """
    wl_range = stats['wl_max'] - stats['wl_min']
    roc_m_per_hr = roc_norm_units * wl_range
    roc_range = roc_max - roc_min
    if roc_range <= 0:
        return np.zeros_like(roc_m_per_hr, dtype=np.float32)
    roc_final = (roc_m_per_hr - roc_min) / roc_range
    return np.clip(roc_final, 0.0, 1.0).astype(np.float32)


def get_static_features(station: str) -> np.ndarray:
    """
    Returns normalized [elevation, distance_to_river, flood_frequency] for a
    station as a length-3 array. Elevation and distance are min-max normalized
    across the 7 stations; flood_frequency is already 0-1.
    """
    attrs = BARANGAY_ATTRS[station]
    elev_norm = (attrs['elevation'] - ELEV_MIN) / (ELEV_MAX - ELEV_MIN) \
        if ELEV_MAX > ELEV_MIN else 0.0
    dist_norm = (attrs['distance_to_river'] - DIST_MIN) / (DIST_MAX - DIST_MIN) \
        if DIST_MAX > DIST_MIN else 0.0
    flood_freq = attrs['flood_frequency']
    return np.array([elev_norm, dist_norm, flood_freq], dtype=np.float32)


# ── Full pipeline: build the 70-dim RF feature vector ─────────────────────────

def build_augmented_features(station: str,
                              horizon: int,
                              model_dir: str = 'models/lstm',
                              wl_path: str = 'data/processed/water_level_hourly.csv',
                              rf_path: str = 'data/processed/rainfall_hourly.csv'):
    """
    Loads the trained LSTM for (station, horizon), runs it over the full
    dataset (train+val+test, in chronological order) and builds the 70-dim
    augmented feature matrix used downstream by the Random Forest.

    Returns:
        features   : (N, 70) float32 array
                     [0:64]   H_aug (weighted LSTM hidden-state trajectory)
                     [64]     W_hat  — predicted water level at t+horizon (normalized)
                     [65]     R_acc  — 6hr accumulated rainfall (normalized)
                     [66]     ROC    — rate of change of water level (normalized)
                     [67]     E_b    — elevation (normalized)
                     [68]     D_b    — distance to river (normalized)
                     [69]     HS_b   — historical flood frequency
        y_true_norm: (N,) normalized true water level at t+horizon (for downstream
                     risk-class labeling / evaluation)
        datetimes  : (N,) aligned datetime index for each row, for traceability
        raw_features: (N, 7) [last_rf_norm, last_wl_norm, R_acc, ROC, E_b, D_b, HS_b]
                     — no LSTM involvement, used for the M2 standalone-RF baseline
    """
    safe_station = station.replace(' ', '_').replace('-', '_')
    model_name = f"{safe_station}_h{horizon}"

    model_path = os.path.join(model_dir, f"{model_name}_final.keras")
    stats_path = os.path.join(model_dir, f"{model_name}_stats.json")

    log.info(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(
        model_path, custom_objects={'LastTimestep': LastTimestep}
    )
    with open(stats_path) as f:
        stats = json.load(f)

    extractor = build_hidden_state_extractor(model)

    # Build sequences across the FULL chronological range (train+val+test),
    # so downstream RF training/evaluation can re-split however it needs.
    df = load_and_merge(wl_path, rf_path, station)
    train_df, val_df, test_df = temporal_train_val_test_split(df)

    # Data-derived ROC range, computed from the TRAINING split only (no
    # leakage from val/test), same pattern as wl_min/max and rf_min/max.
    roc_min, roc_max = get_roc_range(train_df)
    log.info(f"  ROC range (train-derived): [{roc_min:.4f}, {roc_max:.4f}] m/hr")

    train_df, _ = normalize_data(train_df, stats)
    val_df, _   = normalize_data(val_df, stats)
    test_df, _  = normalize_data(test_df, stats)

    full_norm_df = pd.concat([train_df, val_df, test_df]).sort_values('datetime')

    X, y = build_sequences(full_norm_df, horizon)
    log.info(f"  Built {len(X):,} sequences for {station} h{horizon}")

    # Datetimes aligned to each sequence's target timestep (t + horizon)
    dt_values = full_norm_df['datetime'].values
    offset = LOOK_BACK + horizon - 1
    datetimes = dt_values[offset: offset + len(X)]

    # ── LSTM outputs ───────────────────────────────────────────────────────
    hidden_states = extractor.predict(X, verbose=0)      # (N, 24, 64)
    w_hat = model.predict(X, verbose=0).flatten()          # (N,) normalized

    # ── H_aug ──────────────────────────────────────────────────────────────
    h_aug = compute_h_aug(hidden_states)                   # (N, 64)

    # ── Scalar features ───────────────────────────────────────────────────
    rainfall_norm_seq = X[:, :, 0]      # rf_norm channel
    wl_norm_seq       = X[:, :, 1]      # wl_norm channel

    r_acc_raw = compute_r_acc(rainfall_norm_seq)
    roc_raw   = compute_roc(wl_norm_seq)

    r_acc = normalize_r_acc(r_acc_raw, stats)
    roc   = normalize_roc(roc_raw, stats, roc_min, roc_max)

    # ── Static features (broadcast to N rows) ─────────────────────────────
    static_feats = get_static_features(station)             # (3,)
    static_block = np.tile(static_feats, (len(X), 1))        # (N, 3)

    # ── Raw last-step features (for M2 — standalone RF, no LSTM) ───────────
    last_rf_norm = rainfall_norm_seq[:, -1]     # most recent normalized rainfall
    last_wl_norm = wl_norm_seq[:, -1]           # most recent normalized water level
    raw_features = np.concatenate([
        last_rf_norm.reshape(-1, 1),
        last_wl_norm.reshape(-1, 1),
        r_acc.reshape(-1, 1),
        roc.reshape(-1, 1),
        static_block,
    ], axis=1).astype(np.float32)   # (N, 7) — no LSTM involvement at all

    # ── Assemble final 70-dim matrix ──────────────────────────────────────
    features = np.concatenate([
        h_aug,                       # 64
        w_hat.reshape(-1, 1),        # 1
        r_acc.reshape(-1, 1),        # 1
        roc.reshape(-1, 1),          # 1
        static_block,                # 3
    ], axis=1).astype(np.float32)

    assert features.shape[1] == 70, f"Expected 70 features, got {features.shape[1]}"
    log.info(f"  Feature matrix: {features.shape}")

    return features, y, datetimes, raw_features


# ── Entry point — smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    features, y_true_norm, datetimes, raw_features = build_augmented_features(
        station='Sto Nino',
        horizon=1,
    )

    log.info("\n" + "=" * 60)
    log.info("SMOKE TEST — Sto Nino h1")
    log.info("=" * 60)
    log.info(f"Features shape : {features.shape}")
    log.info(f"y_true shape   : {y_true_norm.shape}")
    log.info(f"Datetimes range: {datetimes[0]} to {datetimes[-1]}")
    log.info(f"H_aug[0][:5]   : {features[0, :5]}")
    log.info(f"W_hat[0]       : {features[0, 64]:.4f}")
    log.info(f"R_acc[0]       : {features[0, 65]:.4f}")
    log.info(f"ROC[0]         : {features[0, 66]:.4f}")
    log.info(f"Static[0]      : {features[0, 67:70]}")
    log.info("Smoke test complete.")