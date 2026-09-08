# src/pipeline/train_all_augmentation.py

import os
import logging
import numpy as np

from src.models.augmentation import build_augmented_features

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
HORIZONS = [1, 3, 6]

OUTPUT_DIR = 'data/processed/augmented'


def run_all(model_dir: str = 'models/lstm',
            wl_path: str = 'data/processed/water_level_hourly.csv',
            rf_path: str = 'data/processed/rainfall_hourly.csv',
            output_dir: str = OUTPUT_DIR):

    os.makedirs(output_dir, exist_ok=True)

    results_summary = []
    failed = []

    total = len(STATIONS) * len(HORIZONS)
    count = 0

    for station in STATIONS:
        for horizon in HORIZONS:
            count += 1
            safe_station = station.replace(' ', '_').replace('-', '_')
            out_name = f"{safe_station}_h{horizon}.npz"
            out_path = os.path.join(output_dir, out_name)

            log.info(f"\n{'='*60}")
            log.info(f"[{count}/{total}] {station} | horizon +{horizon}hr")
            log.info(f"{'='*60}")

            try:
                features, y_true_norm, datetimes, raw_features = build_augmented_features(
                    station=station,
                    horizon=horizon,
                    model_dir=model_dir,
                    wl_path=wl_path,
                    rf_path=rf_path,
                )

                np.savez_compressed(
                    out_path,
                    features=features,
                    y_true_norm=y_true_norm,
                    datetimes=datetimes.astype('datetime64[ns]'),
                    raw_features=raw_features,
                )

                log.info(f"Saved: {out_path} | shape {features.shape}")
                results_summary.append({
                    'station': station,
                    'horizon': horizon,
                    'n_rows': features.shape[0],
                    'n_features': features.shape[1],
                    'output': out_path,
                })

            except Exception as e:
                log.error(f"FAILED: {station} h{horizon} — {e}")
                failed.append((station, horizon, str(e)))

    # ── Summary ────────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("BATCH AUGMENTATION SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Completed: {len(results_summary)}/{total}")
    for r in results_summary:
        log.info(f"  {r['station']:15} h{r['horizon']} | "
                  f"{r['n_rows']:,} rows x {r['n_features']} features "
                  f"-> {r['output']}")

    if failed:
        log.info(f"\nFAILED ({len(failed)}):")
        for station, horizon, err in failed:
            log.info(f"  {station} h{horizon}: {err}")
    else:
        log.info("\nAll 21 combinations completed successfully.")

    return results_summary, failed


if __name__ == "__main__":
    run_all()