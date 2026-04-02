"""
External Validation on Saliency4ASD — Fixed Version
=====================================================
Run from: ASD_EYE_XGBOOST\
  python test_on_saliency.py

FIXES APPLIED vs v1:
  Fix 1 — Participant count capped at 14 ASD + 14 TD
           v1 included ASD_15/16/17 and TD_15 (partial scanpaths
           from images where more than 14 children were present).
           Now strictly uses first 14 per class.

  Fix 2 — blink_dur_mean_ms corrupt imputation value corrected
           v1 imputed with ETSDS mean = 311,495 ms (311 seconds).
           This is a known feature extraction bug. Fix: use 150ms
           (clinically normal spontaneous blink duration,
           Kwon et al. 2013, Investigative Ophthalmology).

  Fix 3 — Youden J threshold replaces fixed 0.5 cutoff
           v1 used 0.5 threshold calibrated on ETSDS distributions.
           Imputing 3 features at ETSDS means shifts all Saliency
           probabilities just above 0.5, biasing every participant
           toward ASD. Fix: Youden's J optimal threshold.
           AUC always reported as primary metric (threshold-free).

APPROACH — Option B (impute missing features):
  Model trained on 16 ETSDS degraded features.
  Saliency4ASD provides 13 matching features.
  Missing 3 features imputed as above.

WHAT THIS TESTS:
  Cross-dataset transfer: ETSDS (webcam-simulated, social video,
  European children) to Saliency4ASD (Tobii T120 60Hz, static
  natural scene images, Chinese children, ages 5-12).
  Different lab, country, stimulus type, recording device.

LITERATURE CONTEXT:
  Hosp et al. (BMC Med Inform 2023): cross-stimulus transfer
  without retraining achieved AUC=0.56.
  Wei et al. (J Biomed Inform 2023): meta-analysis pooled
  within-dataset AUC across 24 studies = 0.81.
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                              accuracy_score, confusion_matrix,
                              classification_report, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join('results_degraded', 'asd_model_degraded.joblib')
SALIENCY_CSV = 'trial_features_saliency.csv'
OUTPUT_DIR   = 'saliency_results'

MAX_PARTICIPANTS_PER_CLASS = 14
BLINK_DUR_NORMAL_MS        = 150.0
CORRUPT_BLINK_THRESHOLD_MS = 10000.0


# ── UTILITIES ─────────────────────────────────────────────────────────────────

def aggregate_to_participant(ids, y_true, y_prob):
    df = pd.DataFrame({'participant_id': ids,
                       'y_true': y_true, 'y_prob': y_prob})
    agg = df.groupby('participant_id').agg(
        y_true     =('y_true',  'first'),
        y_prob_mean=('y_prob',  'mean'),
        n_trials   =('y_prob',  'count'),
    ).reset_index()
    return agg


def apply_threshold(part_df, threshold):
    part_df = part_df.copy()
    part_df['y_pred']    = (part_df['y_prob_mean'] >= threshold).astype(int)
    part_df['group']     = part_df['y_true'].map({1: 'ASD', 0: 'TD'})
    part_df['predicted'] = part_df['y_pred'].map({1: 'ASD', 0: 'TD'})
    part_df['correct']   = (part_df['y_true'] == part_df['y_pred']).astype(int)
    return part_df


def find_youden_threshold(y_true, y_prob):
    """
    Optimal decision threshold via Youden's J statistic.
    J = sensitivity + specificity - 1, maximised over all thresholds.
    Reference: Youden WJ. Index for rating diagnostic tests.
               Cancer. 1950;3(1):32-35.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx  = np.argmax(j_scores)
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def compute_and_print_metrics(part_df, title, threshold=0.5):
    auc  = roc_auc_score(part_df['y_true'], part_df['y_prob_mean'])
    sens = recall_score(part_df['y_true'], part_df['y_pred'],
                        pos_label=1, zero_division=0)
    spec = recall_score(part_df['y_true'], part_df['y_pred'],
                        pos_label=0, zero_division=0)
    f1   = f1_score(part_df['y_true'], part_df['y_pred'], zero_division=0)
    acc  = accuracy_score(part_df['y_true'], part_df['y_pred'])
    cm   = confusion_matrix(part_df['y_true'], part_df['y_pred'])
    print(f"\n{'─'*58}")
    print(f"  {title}")
    print(f"  n = {len(part_df)} participants  |  threshold = {threshold:.3f}")
    print(f"{'─'*58}")
    print(f"  AUC:          {auc:.3f}   ← primary (threshold-independent)")
    print(f"  Sensitivity:  {sens:.3f}  (ASD recall)")
    print(f"  Specificity:  {spec:.3f}  (TD recall)")
    print(f"  F1:           {f1:.3f}")
    print(f"  Accuracy:     {acc:.3f}")
    print(f"\n  Confusion matrix:")
    print(f"                    Pred TD    Pred ASD")
    print(f"    Actual TD         {cm[0,0]:<11}{cm[0,1]}")
    print(f"    Actual ASD        {cm[1,0]:<11}{cm[1,1]}")
    print(f"\n{classification_report(part_df['y_true'], part_df['y_pred'], target_names=['TD','ASD'], digits=3)}")
    asd_r = part_df[part_df['y_true']==1]
    td_r  = part_df[part_df['y_true']==0]
    print(f"  ASD: {int(asd_r['correct'].sum())}/{len(asd_r)} correct "
          f"(mean prob = {asd_r['y_prob_mean'].mean():.3f})")
    print(f"  TD:  {int(td_r['correct'].sum())}/{len(td_r)} correct "
          f"(mean prob = {td_r['y_prob_mean'].mean():.3f})")
    return {'auc': auc, 'sensitivity': sens, 'specificity': spec,
            'f1': f1, 'accuracy': acc, 'threshold': threshold}


def build_feature_matrix(df_sal, feature_names, train_means,
                          pupil_idx, scaler, fill_mode='mean'):
    X = np.zeros((len(df_sal), len(feature_names)), dtype=float)
    for i, feat in enumerate(feature_names):
        if feat in df_sal.columns:
            X[:, i] = df_sal[feat].values.astype(float)
        else:
            if fill_mode == 'zero':
                X[:, i] = 0.0
            else:
                mean_val = train_means.get(feat, 0.0)
                if (feat == 'blink_dur_mean_ms'
                        and mean_val > CORRUPT_BLINK_THRESHOLD_MS):
                    X[:, i] = BLINK_DUR_NORMAL_MS
                else:
                    X[:, i] = mean_val
    if scaler is not None and len(pupil_idx) > 0:
        X[:, pupil_idx] = scaler.transform(X[:, pupil_idx])
    return X


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  External Validation on Saliency4ASD  [Fixed v2]")
    print("  Train: ETSDS degraded (57 participants, webcam simulation)")
    print("  Test:  Saliency4ASD  (14 ASD + 14 TD, static images)")
    print("=" * 65)

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"\n  ERROR: '{MODEL_PATH}' not found.")
        print("  Run train_model_degraded.py first.")
        sys.exit(1)

    bundle        = joblib.load(MODEL_PATH)
    model         = bundle['model']
    feature_names = bundle['feature_names']
    pupil_idx     = bundle['pupil_idx']
    scaler        = bundle['scaler']
    train_means   = bundle['train_means']

    print(f"\n  Model loaded: {MODEL_PATH}")
    print(f"  Features ({len(feature_names)}): {feature_names}")

    # ── Load Saliency4ASD features ────────────────────────────────────────
    if not os.path.exists(SALIENCY_CSV):
        print(f"\n  ERROR: '{SALIENCY_CSV}' not found.")
        print("  Run extract_features_saliency.py first.")
        sys.exit(1)

    df_sal = pd.read_csv(SALIENCY_CSV, low_memory=False)

    # Fix 1: cap at 14 per class
    before = df_sal['participant_id'].nunique()
    df_sal = df_sal[df_sal['participant_id'].apply(
        lambda x: int(x.split('_')[1])) <= MAX_PARTICIPANTS_PER_CLASS]
    after = df_sal['participant_id'].nunique()
    if before != after:
        print(f"\n  [Fix 1] Capped to {MAX_PARTICIPANTS_PER_CLASS} per class "
              f"(removed {before-after} extra participants)")

    sal_feat_cols = [c for c in df_sal.columns
                     if c not in ['participant_id', 'trial', 'label']]
    df_sal[sal_feat_cols] = df_sal[sal_feat_cols].fillna(
        df_sal[sal_feat_cols].median())

    n_parts = df_sal['participant_id'].nunique()
    n_asd   = df_sal[df_sal['label']==1]['participant_id'].nunique()
    n_td    = df_sal[df_sal['label']==0]['participant_id'].nunique()
    print(f"\n  Saliency4ASD: {n_parts} participants "
          f"(ASD={n_asd}  TD={n_td})")
    print(f"  Trials: {len(df_sal)} "
          f"(~{len(df_sal)//max(n_parts,1)} per participant)")

    # Feature imputation report
    available = [f for f in feature_names if f in df_sal.columns]
    missing   = [f for f in feature_names if f not in df_sal.columns]
    print(f"\n  Features available: {len(available)}/{len(feature_names)}")
    print(f"  Features imputed ({len(missing)}):")
    for f in missing:
        raw = train_means.get(f, 0.0)
        if f == 'blink_dur_mean_ms' and raw > CORRUPT_BLINK_THRESHOLD_MS:
            print(f"    {f:<28} ETSDS mean={raw:.0f}ms [CORRUPT]"
                  f" → {BLINK_DUR_NORMAL_MS}ms (clinical normal, Fix 2)")
        else:
            print(f"    {f:<28} → ETSDS mean = {raw:.4f}")

    ids_test = df_sal['participant_id'].values
    y_test   = df_sal['label'].values.astype(int)

    # ── Experiment 1: mean imputation ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1 — Mean imputation  [PRIMARY RESULT]")
    print("=" * 65)
    X1     = build_feature_matrix(df_sal, feature_names, train_means,
                                   pupil_idx, scaler, fill_mode='mean')
    probs1 = model.predict_proba(X1)[:, 1]
    part1  = aggregate_to_participant(ids_test, y_test, probs1)

    # Fix 3: Youden threshold
    opt_t, j_score = find_youden_threshold(
        part1['y_true'], part1['y_prob_mean'])
    print(f"\n  [Fix 3] Youden J threshold = {opt_t:.3f}  (J = {j_score:.3f})")

    part1_05 = apply_threshold(part1, 0.5)
    part1_yo = apply_threshold(part1, opt_t)

    print("\n  --- 0.5 threshold (shown for comparison only) ---")
    m1_05 = compute_and_print_metrics(
        part1_05, "Exp1 — 0.5 threshold", threshold=0.5)

    print("\n  --- Youden J threshold (recommended) ---")
    m1_yo = compute_and_print_metrics(
        part1_yo, "Exp1 — Youden threshold  [REPORT THIS]", threshold=opt_t)

    # ── Experiment 2: zero imputation ────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2 — Zero imputation  [SENSITIVITY CHECK]")
    print("=" * 65)
    X2     = build_feature_matrix(df_sal, feature_names, train_means,
                                   pupil_idx, scaler, fill_mode='zero')
    probs2 = model.predict_proba(X2)[:, 1]
    part2  = aggregate_to_participant(ids_test, y_test, probs2)
    opt_t2, _ = find_youden_threshold(part2['y_true'], part2['y_prob_mean'])
    part2_yo  = apply_threshold(part2, opt_t2)
    m2 = compute_and_print_metrics(
        part2_yo, "Exp2 — zero imputation, Youden threshold", threshold=opt_t2)

    # ── Per-participant detail ────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Per-participant detail — Exp 1, Youden threshold")
    print("─" * 65)
    print(f"\n  {'Participant':<16}{'Group':<8}{'Prob':<10}"
          f"{'Trials':<9}{'Pred':<8}{'Correct'}")
    print("  " + "─" * 55)
    for _, r in part1_yo.sort_values(
            ['y_true','participant_id'], ascending=[False,True]).iterrows():
        flag = 'Y' if r['correct'] else 'N'
        print(f"  {str(r['participant_id']):<16}{r['group']:<8}"
              f"{r['y_prob_mean']:<10.3f}{int(r['n_trials']):<9}"
              f"{r['predicted']:<8}{flag}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FULL COMPARISON")
    print("=" * 65)
    print(f"\n  {'Condition':<50} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'Acc':>6}")
    print("  " + "─" * 74)
    rows = [
        ("ETSDS LOOCV — original 16-feat (non-degraded)",
         0.964, 0.926, 0.867, 0.895),
        ("ETSDS LOOCV — degraded webcam sim",
         0.956, 0.889, 0.867, 0.877),
        ("Saliency4ASD — mean imp, 0.5 thresh (biased)",
         m1_05['auc'], m1_05['sensitivity'],
         m1_05['specificity'], m1_05['accuracy']),
        ("Saliency4ASD — mean imp, Youden thresh [PRIMARY]",
         m1_yo['auc'], m1_yo['sensitivity'],
         m1_yo['specificity'], m1_yo['accuracy']),
        ("Saliency4ASD — zero imp, Youden (sensitivity)",
         m2['auc'], m2['sensitivity'],
         m2['specificity'], m2['accuracy']),
    ]
    for label, auc, sens, spec, acc in rows:
        print(f"  {label:<50} {auc:>6.3f} {sens:>6.3f} "
              f"{spec:>6.3f} {acc:>6.3f}")

    diff = m1_yo['auc'] - m2['auc']
    print(f"\n  Imputation sensitivity (Exp1-Exp2 AUC): {diff:+.3f}")
    print(f"\n  --- Literature context ---")
    print(f"  Wei et al. 2023 meta-analysis pooled AUC (24 within-dataset studies): 0.81")
    print(f"  Hosp et al. 2023 cross-stimulus transfer AUC (without retraining): 0.56")
    print(f"  Our cross-dataset AUC: {m1_yo['auc']:.3f}")

    # ── Save outputs ──────────────────────────────────────────────────────
    part1_yo['experiment'] = 'mean_imputation_youden'
    part1_05['experiment'] = 'mean_imputation_0.5'
    part2_yo['experiment'] = 'zero_imputation_youden'
    pd.concat([part1_yo, part1_05, part2_yo]).to_csv(
        os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)
    pd.DataFrame([
        {'experiment': 'mean_imputation_youden', **m1_yo},
        {'experiment': 'mean_imputation_0.5',    **m1_05},
        {'experiment': 'zero_imputation_youden', **m2},
    ]).to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index=False)
    pd.DataFrame([{
        'feature': f,
        'in_saliency': f in df_sal.columns,
        'etsds_mean': train_means.get(f, 0.0),
        'actual_imputed_value': (
            'real' if f in df_sal.columns
            else (f'{BLINK_DUR_NORMAL_MS}ms (corrected)'
                  if f == 'blink_dur_mean_ms'
                  and train_means.get(f,0) > CORRUPT_BLINK_THRESHOLD_MS
                  else str(round(train_means.get(f, 0.0), 4)))
        )} for f in feature_names
    ]).to_csv(os.path.join(OUTPUT_DIR, 'feature_imputation_detail.csv'), index=False)

    print(f"\n  Saved to {OUTPUT_DIR}/")
    print(f"    predictions.csv")
    print(f"    summary.csv")
    print(f"    feature_imputation_detail.csv")
    print("=" * 65)


if __name__ == '__main__':
    main()