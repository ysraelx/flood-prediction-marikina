# src/pipeline/train_all_rf.py

import os
import json
import logging
import numpy as np

from src.models.rf_comparison import run_cv_comparison, STATIONS, HORIZONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

OUTPUT_PATH = 'models/rf/comparison_results.json'


def run_all():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_results = []
    skipped = []
    total = len(STATIONS) * len(HORIZONS)
    count = 0

    for station in STATIONS:
        for horizon in HORIZONS:
            count += 1
            log.info(f"\n{'='*70}")
            log.info(f"[{count}/{total}] {station} h{horizon}")
            log.info(f"{'='*70}")

            result = run_cv_comparison(station, horizon)
            if result is None:
                skipped.append((station, horizon))
            else:
                all_results.append(result)

    # ── Save full results ──────────────────────────────────────────────────
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nFull results saved: {OUTPUT_PATH}")

    # ── Thesis-level summary ───────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("PHASE 6 — THESIS-LEVEL SUMMARY")
    log.info(f"{'='*70}")

    n_success = sum(1 for r in all_results
                     if r['comparison_M3_vs_M4']['meets_success_criterion'])
    n_total = len(all_results)

    log.info(f"Success criterion met: {n_success}/{n_total} "
             f"station-horizon combinations")
    log.info(f"(p<0.05 on both Macro-F1 and Critical-Recall, ΔMacro-F1 >= 0.05)")

    log.info(f"\n{'Station':<15}{'Horizon':<9}{'M1 F1':<9}{'M2 F1':<9}"
             f"{'M3 F1':<9}{'M4 F1':<9}{'ΔF1':<9}{'p-val':<9}{'Result'}")
    log.info("-" * 90)
    for r in all_results:
        c = r['comparison_M3_vs_M4']
        result_str = "SUCCESS" if c['meets_success_criterion'] else "not met"
        log.info(
            f"{r['station']:<15}{'h'+str(r['horizon']):<9}"
            f"{r['M1']['macro_f1_mean']:<9.4f}{r['M2']['macro_f1_mean']:<9.4f}"
            f"{r['M3']['macro_f1_mean']:<9.4f}{r['M4']['macro_f1_mean']:<9.4f}"
            f"{c['delta_macro_f1']:<+9.4f}{c['p_value_macro_f1']:<9.4f}{result_str}"
        )

    if skipped:
        log.info(f"\nSkipped (insufficient class samples for CV): {skipped}")

    # Overall averages across all combos, for a headline thesis number
    avg_m1 = np.mean([r['M1']['macro_f1_mean'] for r in all_results])
    avg_m2 = np.mean([r['M2']['macro_f1_mean'] for r in all_results])
    avg_m3 = np.mean([r['M3']['macro_f1_mean'] for r in all_results])
    avg_m4 = np.mean([r['M4']['macro_f1_mean'] for r in all_results])

    log.info(f"\nOverall mean Macro-F1 across all combos:")
    log.info(f"  M1 (LSTM direct)    : {avg_m1:.4f}")
    log.info(f"  M2 (Raw RF)         : {avg_m2:.4f}")
    log.info(f"  M3 (Non-aug hybrid) : {avg_m3:.4f}")
    log.info(f"  M4 (Augmented)      : {avg_m4:.4f}")

    return all_results, skipped


if __name__ == "__main__":
    run_all()