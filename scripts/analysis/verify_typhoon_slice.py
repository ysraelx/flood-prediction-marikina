# verify_typhoon_slice.py — run from project root: python verify_typhoon_slice.py

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.models.augmentation import build_augmented_features
from src.models.lstm_model import load_and_merge, temporal_train_val_test_split

STATION = 'Sto Nino'
HORIZON = 1
ALERT_WL = 15.0

def main():
    features, y_true_norm, datetimes = build_augmented_features(
        station=STATION, horizon=HORIZON,
    )

    # Reload raw (un-normalized) water level to know actual meters at each datetime
    df = load_and_merge(
        'data/processed/water_level_hourly.csv',
        'data/processed/rainfall_hourly.csv',
        STATION,
    )
    wl_lookup = df.set_index('datetime')['water_level']

    dt_index = pd.to_datetime(datetimes)
    actual_wl = wl_lookup.reindex(dt_index).values

    h_aug_mean = features[:, :64].mean(axis=1)
    w_hat      = features[:, 64]
    r_acc      = features[:, 65]
    roc        = features[:, 66]

    result = pd.DataFrame({
        'datetime':   dt_index,
        'actual_wl':  actual_wl,
        'H_aug_mean': h_aug_mean,
        'W_hat':      w_hat,
        'R_acc':      r_acc,
        'ROC':        roc,
    })

    alert_rows = result[result['actual_wl'] >= ALERT_WL].copy()
    print(f"\nTotal rows: {len(result):,} | Rows at/above Alert ({ALERT_WL}m): {len(alert_rows):,}\n")

    if len(alert_rows) == 0:
        print("No rows found at/above Alert threshold — check data range or threshold value.")
        return

    # Show a contiguous slice from the middle of the alert period, not just the first few
    mid = len(alert_rows) // 2
    start = max(0, mid - 10)
    end = min(len(alert_rows), mid + 15)
    slice_df = alert_rows.iloc[start:end]

    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(slice_df.to_string(index=False))

    print(f"\nAlert-period stats:")
    print(f"  R_acc  mean: {alert_rows['R_acc'].mean():.4f} | overall mean: {result['R_acc'].mean():.4f}")
    print(f"  ROC    mean: {alert_rows['ROC'].mean():.4f}   | overall mean: {result['ROC'].mean():.4f}")

if __name__ == "__main__":
    main()