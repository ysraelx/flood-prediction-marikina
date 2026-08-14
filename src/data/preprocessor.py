# src/data/preprocessor.py

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ── Risk class thresholds per station ─────────────────────────────────────────
THRESHOLDS = {
    "Sto Nino":      {"alert": 15.00, "critical": 17.00},
    "Tumana Bridge": {"alert": 17.26, "critical": 19.26},
    "Nangka":        {"alert": 16.50, "critical": 17.70},
    "Rodriguez":     {"alert": 28.80, "critical": 30.70},
    "San Mateo-1":   {"alert": 18.00, "critical": 20.00},
    "Montalban":     {"alert": 22.40, "critical": 23.60},
    "Burgos":        {"alert": 27.40, "critical": 28.40},
}


def assign_risk_class(water_level: float, station_name: str) -> str:
    """
    Assign flood risk class based on official PAGASA thresholds.
    Normal   → below alert level
    Alert    → at or above alert, below critical
    Critical → at or above critical
    """
    if pd.isna(water_level):
        return None

    thresholds = THRESHOLDS.get(station_name)
    if not thresholds:
        return None

    if water_level >= thresholds["critical"]:
        return "Critical"
    elif water_level >= thresholds["alert"]:
        return "Alert"
    else:
        return "Normal"


def load_water_level(filepath: str) -> pd.DataFrame:
    log.info(f"Loading water level data from {filepath}")
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['water_level'] = pd.to_numeric(df['water_level'], errors='coerce')
    log.info(f"  Loaded {len(df):,} rows | "
             f"Stations: {list(df['station_name'].unique())}")
    return df


def load_rainfall(filepath: str) -> pd.DataFrame:
    log.info(f"Loading rainfall data from {filepath}")
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['rainfall_mm'] = pd.to_numeric(df['rainfall_mm'], errors='coerce')
    df['rainfall_mm'] = df['rainfall_mm'].fillna(0.0)
    log.info(f"  Loaded {len(df):,} rows | "
             f"Stations: {list(df['station_name'].unique())}")
    return df


def resample_water_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 10-minute water level readings to hourly.
    Uses mean of the 6 readings per hour.
    Preserves alert/alarm/critical threshold columns.
    """
    log.info("Resampling water level to hourly (mean)...")

    results = []
    for station, group in df.groupby('station_name'):
        group = group.set_index('datetime').sort_index()

        # Resample numeric columns
        hourly = group['water_level'].resample('h').mean()

        # Keep threshold columns (static per station, take first value)
        thresholds = group[['alertwl', 'alarmwl', 'criticalwl']].resample('h').first()
        obscd      = group['obscd'].iloc[0]

        station_df = pd.DataFrame({
            'station_name': station,
            'obscd':        obscd,
            'water_level':  hourly,
            'alertwl':      thresholds['alertwl'],
            'alarmwl':      thresholds['alarmwl'],
            'criticalwl':   thresholds['criticalwl'],
        })
        results.append(station_df)

    resampled = pd.concat(results).reset_index()
    resampled = resampled.rename(columns={'index': 'datetime'})
    log.info(f"  After resample: {len(resampled):,} hourly rows")
    return resampled


def resample_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 10-minute rainfall readings to hourly.
    Uses sum of the 6 readings per hour (rainfall accumulates).
    """
    log.info("Resampling rainfall to hourly (sum)...")

    results = []
    for station, group in df.groupby('station_name'):
        group = group.set_index('datetime').sort_index()

        hourly = group['rainfall_mm'].resample('h').sum()
        obscd  = group['obscd'].iloc[0]

        station_df = pd.DataFrame({
            'station_name': station,
            'obscd':        obscd,
            'rainfall_mm':  hourly,
        })
        results.append(station_df)

    resampled = pd.concat(results).reset_index()
    resampled = resampled.rename(columns={'index': 'datetime'})
    log.info(f"  After resample: {len(resampled):,} hourly rows")
    return resampled


def interpolate_gaps(df: pd.DataFrame,
                     value_col: str,
                     max_gap_hours: int = 3) -> pd.DataFrame:
    """
    For each station:
    - Interpolate gaps of <= max_gap_hours consecutive missing values
    - Drop entire gap periods > max_gap_hours (set to NaN, handled later)
    """
    log.info(f"Interpolating gaps <= {max_gap_hours}h in '{value_col}'...")

    results = []
    total_interpolated = 0
    total_dropped      = 0

    for station, group in df.groupby('station_name'):
        group = group.set_index('datetime').sort_index()
        series = group[value_col].copy()

        # Find consecutive NaN runs
        is_null       = series.isna()
        null_groups   = (is_null != is_null.shift()).cumsum()
        null_run_lens = is_null.groupby(null_groups).transform('sum')

        # Interpolate short gaps only
        short_gap_mask = is_null & (null_run_lens <= max_gap_hours)
        long_gap_mask  = is_null & (null_run_lens > max_gap_hours)

        series_interp = series.copy()
        if short_gap_mask.any():
            series_interp = series_interp.interpolate(
                method='linear', limit=max_gap_hours
            )
            total_interpolated += short_gap_mask.sum()

        # Long gaps stay as NaN — will be dropped in final step
        series_interp[long_gap_mask] = np.nan
        total_dropped += long_gap_mask.sum()

        group[value_col] = series_interp
        results.append(group.reset_index())

    log.info(f"  Interpolated : {total_interpolated:,} values")
    log.info(f"  Long gaps (NaN kept) : {total_dropped:,} values")
    return pd.concat(results, ignore_index=True)


def add_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add flood risk class label based on water level and station thresholds."""
    log.info("Adding risk class labels...")
    df['risk_class'] = df.apply(
        lambda row: assign_risk_class(row['water_level'], row['station_name']),
        axis=1
    )
    counts = df['risk_class'].value_counts()
    log.info(f"  Risk class distribution:\n{counts.to_string()}")
    return df


def normalize(df: pd.DataFrame,
              col: str,
              stats: dict = None) -> tuple:
    """
    Min-Max normalize a column using training set stats.
    If stats not provided, compute from data (use only for training set).
    Returns (normalized_df, stats_dict)
    """
    if stats is None:
        stats = {
            'min': df[col].min(),
            'max': df[col].max(),
        }

    col_range = stats['max'] - stats['min']
    if col_range == 0:
        df[f'{col}_norm'] = 0.0
    else:
        df[f'{col}_norm'] = (df[col] - stats['min']) / col_range

    return df, stats


def inverse_normalize(value: float, stats: dict) -> float:
    """Convert normalized value back to original scale (meters)."""
    return value * (stats['max'] - stats['min']) + stats['min']


def run_preprocessing(
    wl_path:  str = "data/raw/water_level_marikina.csv",
    rf_path:  str = "data/raw/rainfall_marikina.csv",
    out_dir:  str = "data/processed",
):
    os.makedirs(out_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("PREPROCESSING PIPELINE START")
    log.info("=" * 60)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    wl_raw = load_water_level(wl_path)
    wl_raw = wl_raw[wl_raw['station_name'] != 'Angono'].copy()
    log.info("Excluded Angono (no official PAGASA thresholds)")
    rf_raw = load_rainfall(rf_path)

    # ── 2. Resample to hourly ──────────────────────────────────────────────────
    wl_hourly = resample_water_level(wl_raw)
    rf_hourly = resample_rainfall(rf_raw)

    # ── 3. Interpolate gaps ────────────────────────────────────────────────────
    wl_hourly = interpolate_gaps(wl_hourly, 'water_level', max_gap_hours=3)
    rf_hourly = interpolate_gaps(rf_hourly, 'rainfall_mm', max_gap_hours=3)

    # ── 4. Drop remaining NaN rows (long gaps > 3 hours) ──────────────────────
    wl_before = len(wl_hourly)
    rf_before = len(rf_hourly)
    wl_hourly = wl_hourly.dropna(subset=['water_level'])
    rf_hourly = rf_hourly.dropna(subset=['rainfall_mm'])
    log.info(f"Dropped {wl_before - len(wl_hourly):,} WL rows (long gaps)")
    log.info(f"Dropped {rf_before - len(rf_hourly):,} RF rows (long gaps)")

    # ── 5. Add risk class labels ───────────────────────────────────────────────
    wl_hourly = add_risk_labels(wl_hourly)

    # ── 6. Sort ────────────────────────────────────────────────────────────────
    wl_hourly = wl_hourly.sort_values(
        ['station_name', 'datetime']
    ).reset_index(drop=True)
    rf_hourly = rf_hourly.sort_values(
        ['station_name', 'datetime']
    ).reset_index(drop=True)

    # ── 7. Save clean hourly files ─────────────────────────────────────────────
    wl_out = os.path.join(out_dir, "water_level_hourly.csv")
    rf_out = os.path.join(out_dir, "rainfall_hourly.csv")

    wl_hourly.to_csv(wl_out, index=False)
    rf_hourly.to_csv(rf_out, index=False)

    log.info("=" * 60)
    log.info("PREPROCESSING COMPLETE")
    log.info(f"Water level hourly : {wl_out}")
    log.info(f"  Shape            : {wl_hourly.shape}")
    log.info(f"  Stations         : "
             f"{list(wl_hourly['station_name'].unique())}")
    log.info(f"  Date range       : {wl_hourly['datetime'].min()} "
             f"to {wl_hourly['datetime'].max()}")
    log.info(f"Rainfall hourly    : {rf_out}")
    log.info(f"  Shape            : {rf_hourly.shape}")
    log.info(f"  Stations         : "
             f"{list(rf_hourly['station_name'].unique())}")
    log.info("=" * 60)

    # ── 8. Print risk class summary ────────────────────────────────────────────
    log.info("RISK CLASS SUMMARY PER STATION")
    log.info("-" * 60)
    for station in wl_hourly['station_name'].unique():
        s = wl_hourly[wl_hourly['station_name'] == station]
        counts = s['risk_class'].value_counts()
        normal   = counts.get('Normal',   0)
        alert    = counts.get('Alert',    0)
        critical = counts.get('Critical', 0)
        total    = len(s)
        log.info(
            f"  {station:20} | "
            f"Normal: {normal:5,} ({normal/total*100:.1f}%) | "
            f"Alert: {alert:4,} ({alert/total*100:.1f}%) | "
            f"Critical: {critical:4,} ({critical/total*100:.1f}%)"
        )
    log.info("=" * 60)

    return wl_hourly, rf_hourly


if __name__ == "__main__":
    run_preprocessing()