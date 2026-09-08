# diagnose_tumana_h6.py — run from project root: python diagnose_tumana_h6.py

import numpy as np
from src.models.rf_comparison import load_station_horizon

d = load_station_horizon('Tumana Bridge', 6)
y = d['y_labels']

unique, counts = np.unique(y, return_counts=True)
print(f"Tumana Bridge h6 — total samples: {len(y):,}")
print("\nClass distribution:")
for cls, cnt in zip(unique, counts):
    print(f"  {cls:10} {cnt:6,} ({cnt/len(y)*100:.2f}%)")

print(f"\nRarest class: {unique[counts.argmin()]} with {counts.min()} samples")
print(f"At 5-fold CV, that's ~{counts.min()/5:.1f} samples per fold for that class")
print("(if this is under ~10, expect high fold-to-fold variance for that class,")
print(" which can easily produce a noisy/negative delta by chance in one run)")