# scripts/analysis/aggregate_analysis.py
# Run from anywhere: python scripts/analysis/aggregate_analysis.py

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import json
import numpy as np
from scipy import stats as sstats

RESULTS_PATH = 'models/rf/comparison_results.json'


def run_aggregate_test(results, metric_key, label):
    m3 = np.array([r['M3'][metric_key] for r in results])
    m4 = np.array([r['M4'][metric_key] for r in results])
    deltas = m4 - m3

    t, p = sstats.ttest_rel(m4, m3)
    p_one_sided = p / 2 if t > 0 else 1 - p / 2
    w_stat, w_p = sstats.wilcoxon(m4, m3)

    print(f"\n{'='*60}")
    print(f"AGGREGATE TEST — {label}")
    print(f"{'='*60}")
    print(f"n = {len(results)} station-horizon combinations")
    print(f"Mean M3: {m3.mean():.4f}")
    print(f"Mean M4: {m4.mean():.4f}")
    print(f"Mean delta: {deltas.mean():+.4f} (std {deltas.std():.4f})")
    print(f"Combos where M4 > M3: {(deltas > 0).sum()}/{len(results)}")
    print(f"Paired t-test      : t={t:.4f}, two-sided p={p:.6f}, "
          f"one-sided p={p_one_sided:.6f}")
    print(f"Wilcoxon (cross-check): stat={w_stat:.4f}, p={w_p:.6f}")

    return {
        'metric': label,
        'mean_m3': float(m3.mean()),
        'mean_m4': float(m4.mean()),
        'mean_delta': float(deltas.mean()),
        'n_positive': int((deltas > 0).sum()),
        'n_total': len(results),
        't_stat': float(t),
        'p_two_sided': float(p),
        'p_one_sided': float(p_one_sided),
        'wilcoxon_p': float(w_p),
    }


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} station-horizon combinations from {RESULTS_PATH}")

    fold_counts = {r['n_folds_used'] for r in results if 'n_folds_used' in r}
    if fold_counts:
        print(f"Fold counts used across combos: {sorted(fold_counts)}")

    f1_result = run_aggregate_test(results, 'macro_f1_mean', 'Macro-F1')
    recall_result = run_aggregate_test(results, 'critical_recall_mean', 'Critical-Recall')

    print(f"\n{'='*60}")
    print("PER-COMBO DELTAS (Macro-F1), sorted ascending")
    print(f"{'='*60}")
    combo_deltas = [
        (r['station'], r['horizon'], r['M4']['macro_f1_mean'] - r['M3']['macro_f1_mean'])
        for r in results
    ]
    combo_deltas.sort(key=lambda x: x[2])
    for station, horizon, delta in combo_deltas:
        flag = "  <-- outlier" if delta < 0 else ""
        print(f"  {station:15} h{horizon}  ΔF1={delta:+.4f}{flag}")

    out = {
        'macro_f1_aggregate': f1_result,
        'critical_recall_aggregate': recall_result,
        'per_combo_f1_deltas': [
            {'station': s, 'horizon': h, 'delta_f1': d} for s, h, d in combo_deltas
        ],
    }
    with open('models/rf/aggregate_analysis.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: models/rf/aggregate_analysis.json")


if __name__ == "__main__":
    main()