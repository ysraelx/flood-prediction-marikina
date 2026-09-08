# experiment_m4_fix.py — run from project root: python experiment_m4_fix.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score
from scipy import stats as sstats

from src.models.rf_comparison import load_station_horizon, RISK_CLASSES, N_FOLDS, RANDOM_STATE

STATION = 'Sto Nino'
HORIZON = 1
N_PCA_COMPONENTS = 10


def evaluate(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, labels=RISK_CLASSES, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, labels=['Critical'], average='macro', zero_division=0)
    return f1, recall


def run_variant(name, X4_builder, rf_params, X3, y, skf):
    """
    X4_builder(X_train_raw, X_test_raw) -> (X_train_final, X_test_final)
    Lets each variant do its own per-fold transform (e.g. PCA fit on train only).
    """
    fold_f1, fold_recall = [], []
    for train_idx, test_idx in skf.split(X3, y):
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test = X4_builder(train_idx, test_idx)

        clf = RandomForestClassifier(**rf_params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1, recall = evaluate(y_test, y_pred)
        fold_f1.append(f1)
        fold_recall.append(recall)

    print(f"\n{name}")
    print(f"  Macro-F1       : {np.mean(fold_f1):.4f} (folds: {[round(x,4) for x in fold_f1]})")
    print(f"  Critical-Recall: {np.mean(fold_recall):.4f} (folds: {[round(x,4) for x in fold_recall]})")
    return fold_f1, fold_recall


def main():
    d = load_station_horizon(STATION, HORIZON)
    X3 = d['features_m3']       # 6-dim, no H_aug — our comparison baseline
    X4 = d['features_m4']       # 70-dim, full H_aug
    y = d['y_labels']

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    base_params = dict(
        n_estimators=200, max_depth=None, min_samples_leaf=3,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
    )

    # M3 baseline (for reference — no H_aug)
    m3_f1, m3_recall = run_variant(
        "M3 (no H_aug, 6 features, reference)",
        lambda tr, te: (X3[tr], X3[te]),
        base_params, X3, y, skf,
    )

    # M4 baseline (current default — max_features='sqrt')
    m4_base_f1, m4_base_recall = run_variant(
        "M4 baseline (70 features, max_features='sqrt' default)",
        lambda tr, te: (X4[tr], X4[te]),
        base_params, X3, y, skf,
    )

    # M4 with tuned max_features (raise candidate pool so W_hat gets picked more)
    tuned_params = dict(base_params, max_features=0.5)
    m4_tuned_f1, m4_tuned_recall = run_variant(
        "M4 tuned (70 features, max_features=0.5)",
        lambda tr, te: (X4[tr], X4[te]),
        tuned_params, X3, y, skf,
    )

    # M4 with PCA-compressed H_aug (fit PCA on train fold only, no leakage)
    def pca_builder(train_idx, test_idx):
        h_aug_train, h_aug_test = X4[train_idx, :64], X4[test_idx, :64]
        scalars_train, scalars_test = X4[train_idx, 64:], X4[test_idx, 64:]

        pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
        h_aug_train_pca = pca.fit_transform(h_aug_train)
        h_aug_test_pca = pca.transform(h_aug_test)

        X_train = np.concatenate([h_aug_train_pca, scalars_train], axis=1)
        X_test = np.concatenate([h_aug_test_pca, scalars_test], axis=1)
        return X_train, X_test

    m4_pca_f1, m4_pca_recall = run_variant(
        f"M4 PCA (H_aug 64->{N_PCA_COMPONENTS} components + 6 scalars = {N_PCA_COMPONENTS+6} features)",
        pca_builder,
        base_params, X3, y, skf,
    )

    # ── Compare each variant against M3 ────────────────────────────────────
    print(f"\n{'='*60}")
    print("PAIRED T-TESTS vs M3 (baseline)")
    print(f"{'='*60}")
    for name, f1s, recalls in [
        ("M4 baseline", m4_base_f1, m4_base_recall),
        ("M4 tuned (max_features=0.5)", m4_tuned_f1, m4_tuned_recall),
        (f"M4 PCA ({N_PCA_COMPONENTS} comp)", m4_pca_f1, m4_pca_recall),
    ]:
        t_f1, p_f1 = sstats.ttest_rel(f1s, m3_f1)
        t_r, p_r = sstats.ttest_rel(recalls, m3_recall)
        delta_f1 = np.mean(f1s) - np.mean(m3_f1)
        delta_r = np.mean(recalls) - np.mean(m3_recall)
        print(f"{name:35} ΔF1={delta_f1:+.4f} (p={p_f1:.4f})  "
              f"ΔRecall={delta_r:+.4f} (p={p_r:.4f})")


if __name__ == "__main__":
    main()