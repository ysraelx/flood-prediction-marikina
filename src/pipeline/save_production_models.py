# src/pipeline/save_production_models.py

import os
import json
import logging
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from src.models.rf_comparison import (
    load_station_horizon, STATIONS, HORIZONS,
    RF_PARAMS, N_PCA_COMPONENTS, RANDOM_STATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

OUTPUT_DIR = 'models/rf/production'


def train_and_save(station: str, horizon: int, output_dir: str = OUTPUT_DIR):
    """
    Trains the FINAL M4 model (PCA transformer + Random Forest) on the full
    dataset for one station-horizon — not a CV fold, this is the model that
    actually goes into production / the FastAPI backend.

    Saves two files:
      {station}_h{horizon}_pca.joblib  — fitted PCA transformer (64 -> 10)
      {station}_h{horizon}_rf.joblib   — fitted RandomForestClassifier
    """
    safe_station = station.replace(' ', '_').replace('-', '_')
    os.makedirs(output_dir, exist_ok=True)

    d = load_station_horizon(station, horizon)
    y = d['y_labels']
    features_m4 = d['features_m4']   # (N, 70)

    h_aug = features_m4[:, :64]
    scalars = features_m4[:, 64:]

    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    h_aug_pca = pca.fit_transform(h_aug)

    X_final = np.concatenate([h_aug_pca, scalars], axis=1)

    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_final, y)

    pca_path = os.path.join(output_dir, f"{safe_station}_h{horizon}_pca.joblib")
    rf_path = os.path.join(output_dir, f"{safe_station}_h{horizon}_rf.joblib")
    joblib.dump(pca, pca_path)
    joblib.dump(clf, rf_path)

    # Training-set class distribution, for traceability
    unique, counts = np.unique(y, return_counts=True)
    dist = {str(u): int(c) for u, c in zip(unique, counts)}

    log.info(f"  {station} h{horizon}: trained on {len(y):,} rows | "
             f"class dist {dist}")
    log.info(f"  Saved: {pca_path}")
    log.info(f"  Saved: {rf_path}")

    return {
        'station': station,
        'horizon': horizon,
        'n_samples': len(y),
        'class_distribution': dist,
        'pca_path': pca_path,
        'rf_path': rf_path,
    }


def run_all(output_dir: str = OUTPUT_DIR):
    manifest = []
    total = len(STATIONS) * len(HORIZONS)
    count = 0

    for station in STATIONS:
        for horizon in HORIZONS:
            count += 1
            log.info(f"\n[{count}/{total}] {station} h{horizon}")
            try:
                result = train_and_save(station, horizon, output_dir)
                manifest.append(result)
            except Exception as e:
                log.error(f"  FAILED: {station} h{horizon} — {e}")

    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"Production models saved: {len(manifest)}/{total}")
    log.info(f"Manifest: {manifest_path}")
    log.info(f"{'='*60}")

    return manifest


if __name__ == "__main__":
    run_all()