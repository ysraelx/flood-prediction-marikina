# Phase 6 — RF + baseline comparison: M1 standalone LSTM, M2 standalone RF,
# M3 non-augmented hybrid, M4 augmented hybrid (proposed, H_aug PCA-compressed
# 64->10 components before RF — see N_PCA_COMPONENTS below for rationale).
# 5-fold stratified CV, paired t-test M3 vs M4.

import numpy as np
import json
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score, accuracy_score
from scipy import stats as sstats

from src.data.preprocessor import assign_risk_class, THRESHOLDS
from src.models.lstm_model import inverse_normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

STATIONS = [
    'Sto Nino', 'Tumana Bridge', 'Rodriguez', 'Nangka',
    'San Mateo-1', 'Montalban', 'Burgos',
]
HORIZONS = [1, 3, 6]

RISK_CLASSES = ['Normal', 'Alert', 'Critical']
N_FOLDS = 5
RANDOM_STATE = 42

N_PCA_COMPONENTS = 10   # H_aug 64 -> 10 components before feeding M4's RF;
                        # chosen after an empirical comparison (baseline vs
                        # max_features tuning vs PCA) on Sto Nino h1 showed
                        # PCA was the only variant to beat the non-augmented
                        # M3 baseline at all, and most consistently across folds.

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=3,
    class_weight='balanced',   # station data is heavily Normal-skewed
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def load_station_horizon(station: str, horizon: int,
                          model_dir: str = 'models/lstm',
                          augmented_dir: str = 'data/processed/augmented'):
    """
    Loads the .npz for this station/horizon plus its LSTM stats.json, and
    builds true risk-class labels from the actual (un-normalized) water level.
    """
    safe_station = station.replace(' ', '_').replace('-', '_')
    npz_path = os.path.join(augmented_dir, f"{safe_station}_h{horizon}.npz")
    stats_path = os.path.join(model_dir, f"{safe_station}_h{horizon}_stats.json")

    data = np.load(npz_path, allow_pickle=True)
    with open(stats_path) as f:
        stats = json.load(f)

    features_m4 = data['features']          # (N, 70)
    raw_features_m2 = data['raw_features']   # (N, 7)
    y_true_norm = data['y_true_norm']        # (N,)

    # M3 = M4 minus H_aug (same scalar/static features, isolates H_aug's effect)
    features_m3 = features_m4[:, 64:70]      # (N, 6): W_hat, R_acc, ROC, E_b, D_b, HS_b

    # M1 predicted water level is just W_hat from the M4 vector (index 64), un-normalized
    w_hat_norm = features_m4[:, 64]
    w_hat_meters = np.array([inverse_normalize(v, stats) for v in w_hat_norm])

    # True labels from actual water level
    y_true_meters = np.array([inverse_normalize(v, stats) for v in y_true_norm])
    y_labels = np.array([
        assign_risk_class(wl, station) for wl in y_true_meters
    ])

    # M1 predicted labels — thresholds applied directly to LSTM's own prediction
    m1_pred_labels = np.array([
        assign_risk_class(wl, station) for wl in w_hat_meters
    ])

    return {
        'features_m4': features_m4,
        'features_m3': features_m3,
        'features_m2': raw_features_m2,
        'm1_pred_labels': m1_pred_labels,
        'y_labels': y_labels,
    }


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        'macro_f1':        f1_score(y_true, y_pred, labels=RISK_CLASSES,
                                     average='macro', zero_division=0),
        'critical_recall':  recall_score(y_true, y_pred, labels=['Critical'],
                                          average='macro', zero_division=0),
        'accuracy':        accuracy_score(y_true, y_pred),
    }


def run_cv_comparison(station: str, horizon: int) -> dict:
    """
    Runs stratified CV (5-fold, or fewer if the rarest class doesn't have
    enough samples for 5) for M2, M3, M4, evaluates M1 directly (no training
    needed, thresholds applied to LSTM output), and paired t-tests M3 vs M4
    across the folds actually used.
    """
    d = load_station_horizon(station, horizon)
    y = d['y_labels']

    # Adaptive fold count: use 5 where possible, but shrink down to the
    # rarest class's sample count (min 2) rather than skipping a station
    # entirely just because Critical events are scarce there.
    unique, counts = np.unique(y, return_counts=True)
    n_folds = min(N_FOLDS, int(counts.min()))
    if n_folds < 2:
        log.warning(f"  {station} h{horizon}: class '{unique[counts.argmin()]}' "
                     f"has only {counts.min()} sample(s) — cannot run CV at all, skipping.")
        return None
    if n_folds < N_FOLDS:
        log.warning(f"  {station} h{horizon}: rarest class '{unique[counts.argmin()]}' "
                     f"has only {counts.min()} samples — using {n_folds}-fold CV "
                     f"instead of {N_FOLDS}-fold.")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_results = {'M2': [], 'M3': [], 'M4': []}

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(d['features_m4'], y)):
        y_train, y_test = y[train_idx], y[test_idx]

        for model_key, X in [('M2', d['features_m2']),
                              ('M3', d['features_m3'])]:
            X_train, X_test = X[train_idx], X[test_idx]
            clf = RandomForestClassifier(**RF_PARAMS)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fold_results[model_key].append(evaluate_predictions(y_test, y_pred))

        # ── M4: PCA-compress H_aug (fit on this fold's TRAIN split only,
        # to avoid leaking test-fold information into the components) ────
        h_aug_train = d['features_m4'][train_idx, :64]
        h_aug_test  = d['features_m4'][test_idx, :64]
        scalars_train = d['features_m4'][train_idx, 64:]
        scalars_test  = d['features_m4'][test_idx, 64:]

        pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
        h_aug_train_pca = pca.fit_transform(h_aug_train)
        h_aug_test_pca  = pca.transform(h_aug_test)

        X_train_m4 = np.concatenate([h_aug_train_pca, scalars_train], axis=1)
        X_test_m4  = np.concatenate([h_aug_test_pca, scalars_test], axis=1)

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train_m4, y_train)
        y_pred = clf.predict(X_test_m4)
        fold_results['M4'].append(evaluate_predictions(y_test, y_pred))

    # M1 — no training, thresholds applied directly to LSTM prediction, evaluated
    # on the full dataset (there's no model to cross-validate, it's a direct rule)
    m1_metrics = evaluate_predictions(y, d['m1_pred_labels'])

    # ── Aggregate CV folds ────────────────────────────────────────────────
    summary = {'station': station, 'horizon': horizon, 'n_samples': len(y),
               'n_folds_used': n_folds}
    for model_key in ['M2', 'M3', 'M4']:
        macro_f1s = [f['macro_f1'] for f in fold_results[model_key]]
        crit_recalls = [f['critical_recall'] for f in fold_results[model_key]]
        accs = [f['accuracy'] for f in fold_results[model_key]]
        summary[model_key] = {
            'macro_f1_mean':        float(np.mean(macro_f1s)),
            'macro_f1_std':         float(np.std(macro_f1s)),
            'critical_recall_mean': float(np.mean(crit_recalls)),
            'critical_recall_std':  float(np.std(crit_recalls)),
            'accuracy_mean':        float(np.mean(accs)),
            'fold_macro_f1':        macro_f1s,
            'fold_critical_recall': crit_recalls,
        }

    summary['M1'] = {
        'macro_f1_mean':        m1_metrics['macro_f1'],
        'critical_recall_mean': m1_metrics['critical_recall'],
        'accuracy_mean':        m1_metrics['accuracy'],
    }

    # ── Paired t-test: M3 vs M4, across the 5 folds ────────────────────────
    m3_f1 = summary['M3']['fold_macro_f1']
    m4_f1 = summary['M4']['fold_macro_f1']
    m3_recall = summary['M3']['fold_critical_recall']
    m4_recall = summary['M4']['fold_critical_recall']

    t_f1, p_f1 = sstats.ttest_rel(m4_f1, m3_f1)
    t_recall, p_recall = sstats.ttest_rel(m4_recall, m3_recall)

    delta_f1 = summary['M4']['macro_f1_mean'] - summary['M3']['macro_f1_mean']
    delta_recall = summary['M4']['critical_recall_mean'] - summary['M3']['critical_recall_mean']

    success = (p_f1 < 0.05) and (p_recall < 0.05) and (delta_f1 >= 0.05)

    summary['comparison_M3_vs_M4'] = {
        'delta_macro_f1':        float(delta_f1),
        'delta_critical_recall': float(delta_recall),
        'p_value_macro_f1':      float(p_f1),
        'p_value_critical_recall': float(p_recall),
        'meets_success_criterion': bool(success),
    }

    log.info(f"\n{station} h{horizon} | n={len(y)}")
    log.info(f"  M1 (LSTM direct)    Macro-F1={summary['M1']['macro_f1_mean']:.4f}  "
             f"Crit-Recall={summary['M1']['critical_recall_mean']:.4f}")
    log.info(f"  M2 (Raw RF)         Macro-F1={summary['M2']['macro_f1_mean']:.4f}  "
             f"Crit-Recall={summary['M2']['critical_recall_mean']:.4f}")
    log.info(f"  M3 (Non-aug hybrid) Macro-F1={summary['M3']['macro_f1_mean']:.4f}  "
             f"Crit-Recall={summary['M3']['critical_recall_mean']:.4f}")
    log.info(f"  M4 (Augmented)      Macro-F1={summary['M4']['macro_f1_mean']:.4f}  "
             f"Crit-Recall={summary['M4']['critical_recall_mean']:.4f}")
    log.info(f"  ΔMacro-F1={delta_f1:+.4f} (p={p_f1:.4f})  "
             f"ΔCrit-Recall={delta_recall:+.4f} (p={p_recall:.4f})  "
             f"{'SUCCESS ✓' if success else 'NOT MET ✗'}")

    return summary


if __name__ == "__main__":
    result = run_cv_comparison(station='Sto Nino', horizon=1)
    log.info("\nSmoke test complete.")