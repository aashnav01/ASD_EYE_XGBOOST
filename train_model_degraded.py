"""
ASD Classification — LOOCV on Degraded ETSDS + Save Model
===========================================================
Run from: ASD_EYE_XGBOOST\
  python train_model_degraded.py

Reads:   trial_features_degraded.csv
Outputs: results_degraded\
           loocv_fold_results.csv
           loocv_participant_predictions.csv
           loocv_summary.csv
           statistical_validation.csv
           feature_importance.csv
           importance_permutation.png
           importance_xgb_gain.png
           asd_model_degraded.joblib   <-- NEW: saved model bundle
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats         import shapiro, mannwhitneyu, ttest_ind
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (roc_auc_score, f1_score, recall_score,
                                     confusion_matrix, classification_report,
                                     accuracy_score)
from sklearn.inspection      import permutation_importance
from imblearn.combine        import SMOTEENN
from xgboost                 import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
RANDOM_STATE       = 42
SIGNIFICANCE_ALPHA = 0.05
IMBALANCE_THRESH   = 0.65

PUPIL_COLS = ['pupil_r_mean', 'pupil_r_std', 'pupil_l_mean', 'pupil_l_std',
              'pupil_asymmetry_mean', 'pupil_asymmetry_std',
              'pupil_mean', 'pupil_std', 'pupil_slope']

XGB_PARAMS = {
    'n_estimators':     300,
    'max_depth':        3,
    'learning_rate':    0.03,
    'subsample':        0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'reg_alpha':        1.0,
    'reg_lambda':       2.0,
    'random_state':     RANDOM_STATE,
    'verbosity':        0,
}

FEATURE_FILE = 'trial_features_degraded.csv'
OUTPUT_DIR   = 'results_degraded'
META_COLS    = ['participant_id', 'trial', 'stimulus_type', 'label']


# ── UTILITIES ─────────────────────────────────────────────────────────────────

def load_data(path=FEATURE_FILE):
    if not os.path.exists(path):
        print(f"ERROR: '{path}' not found.")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(axis=1, how='all')
    df = df[[c for c in df.columns if 'peak_vel' not in c]]
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    return df


def validate_features(X_part_means, y_part, alpha=SIGNIFICANCE_ALPHA):
    significant, effect_sizes, p_values = [], {}, {}
    for col in X_part_means.columns:
        asd = X_part_means[col][y_part == 1].dropna().values
        td  = X_part_means[col][y_part == 0].dropna().values
        if len(asd) < 3 or len(td) < 3:
            continue
        _, p_asd = shapiro(asd)
        _, p_td  = shapiro(td)
        if p_asd < 0.05 or p_td < 0.05:
            stat, p = mannwhitneyu(asd, td, alternative='two-sided')
            r       = float(1 - (2 * stat) / (len(asd) * len(td)))
        else:
            _, p       = ttest_ind(asd, td, equal_var=False)
            pooled_std = np.sqrt((asd.std()**2 + td.std()**2) / 2.0)
            r          = float((asd.mean() - td.mean()) / (pooled_std + 1e-8))
        p_values[col] = float(p)
        if p < alpha:
            significant.append(col)
            effect_sizes[col] = r
    return significant, effect_sizes, p_values


def aggregate_to_participant(participant_ids, y_true, y_prob):
    df_agg = pd.DataFrame({'participant_id': participant_ids,
                           'y_true': y_true, 'y_prob': y_prob})
    agg = df_agg.groupby('participant_id').agg(
        y_true     =('y_true', 'first'),
        y_prob_mean=('y_prob', 'mean'),
        n_trials   =('y_prob', 'count'),
    ).reset_index()
    agg['y_pred']    = (agg['y_prob_mean'] >= 0.5).astype(int)
    agg['group']     = agg['y_true'].map({1: 'ASD', 0: 'TD'})
    agg['predicted'] = agg['y_pred'].map({1: 'ASD', 0: 'TD'})
    agg['correct']   = (agg['y_true'] == agg['y_pred']).astype(int)
    return agg


def print_participant_metrics(part_df, title):
    auc  = roc_auc_score(part_df['y_true'], part_df['y_prob_mean'])
    sens = recall_score(part_df['y_true'], part_df['y_pred'], pos_label=1, zero_division=0)
    spec = recall_score(part_df['y_true'], part_df['y_pred'], pos_label=0, zero_division=0)
    f1   = f1_score(part_df['y_true'], part_df['y_pred'], zero_division=0)
    acc  = accuracy_score(part_df['y_true'], part_df['y_pred'])
    cm   = confusion_matrix(part_df['y_true'], part_df['y_pred'])
    print(f"\n{'─'*52}")
    print(f"  {title}")
    print(f"  n = {len(part_df)} participants")
    print(f"{'─'*52}")
    print(f"  AUC:          {auc:.3f}")
    print(f"  Sensitivity:  {sens:.3f}  (ASD recall)")
    print(f"  Specificity:  {spec:.3f}  (TD recall)")
    print(f"  F1:           {f1:.3f}")
    print(f"  Accuracy:     {acc:.3f}")
    print(f"\n  Confusion matrix:")
    print(f"                    Pred TD    Pred ASD")
    print(f"    Actual TD         {cm[0,0]:<11}{cm[0,1]}")
    print(f"    Actual ASD        {cm[1,0]:<11}{cm[1,1]}")
    print(f"\n  Classification report:")
    print(classification_report(part_df['y_true'], part_df['y_pred'],
                                target_names=['TD', 'ASD'], digits=3))
    return {'auc': auc, 'sensitivity': sens, 'specificity': spec,
            'f1': f1, 'accuracy': acc}


# ── STEP 0 — GLOBAL FEATURE VALIDATION ───────────────────────────────────────

def global_feature_validation(df, feat_cols):
    pm = df.groupby('participant_id')[feat_cols].mean()
    pl = df.groupby('participant_id')['label'].first()
    return validate_features(pm, pl.values)


# ── STEP 1 — LOOCV ───────────────────────────────────────────────────────────

def run_loocv(df, significant_features):
    feat_cols = significant_features
    logo      = LeaveOneGroupOut()
    groups    = df['participant_id'].values
    y         = df['label'].values.astype(int)
    X         = df[feat_cols].values

    all_ids, all_true, all_probs = [], [], []
    fold_rows = []
    n_parts   = df['participant_id'].nunique()

    print(f"\nRunning LOOCV ({n_parts} folds)...")
    print(f"\n{'Fold':<6}{'Left-out':<14}{'Group':<8}{'Prob(ASD)':<12}{'Correct'}")
    print("─" * 48)

    for fold_i, (tr_idx, te_idx) in enumerate(logo.split(X, y, groups)):
        left_out_id    = groups[te_idx[0]]
        left_out_label = y[te_idx[0]]
        group_name     = 'ASD' if left_out_label == 1 else 'TD'

        df_train = df.iloc[tr_idx].copy()
        df_test  = df.iloc[te_idx].copy()
        y_test   = y[te_idx].astype(int)
        ids_test = groups[te_idx]

        pm  = df_train.groupby('participant_id')[feat_cols].mean()
        pl  = df_train.groupby('participant_id')['label'].first()
        sig, _, _ = validate_features(pm, pl.values)
        if len(sig) == 0:
            sig = feat_cols

        Xt   = df_train[sig].values.astype(float)
        y_tr = df_train['label'].values.astype(int)
        Xv   = df_test[sig].values.astype(float)

        pupil_idx = [i for i, f in enumerate(sig)
                     if any(p in f for p in PUPIL_COLS)]
        if len(pupil_idx) > 0:
            sc = StandardScaler()
            sc.fit(df_train[sig].values[:, pupil_idx])
            Xt[:, pupil_idx] = sc.transform(Xt[:, pupil_idx])
            Xv[:, pupil_idx] = sc.transform(Xv[:, pupil_idx])

        n_asd  = int(np.sum(y_tr == 1))
        n_td   = int(np.sum(y_tr == 0))
        ratio  = min(n_asd, n_td) / (max(n_asd, n_td) + 1e-8)
        y_tr_s = y_tr.copy()
        if ratio < IMBALANCE_THRESH and min(n_asd, n_td) >= 6:
            try:
                sm         = SMOTEENN(random_state=RANDOM_STATE)
                Xt, y_tr_s = sm.fit_resample(Xt, y_tr)
            except Exception:
                pass

        counts    = np.bincount(y_tr_s.astype(int))
        scale_pos = float(counts[0]) / float(counts[1] + 1e-8)
        model     = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos)
        model.fit(Xt, y_tr_s, verbose=False)
        probs = model.predict_proba(Xv)[:, 1]

        all_ids.extend(ids_test.tolist())
        all_true.extend(y_test.tolist())
        all_probs.extend(probs.tolist())

        prob_mean = float(np.mean(probs))
        pred      = int(prob_mean >= 0.5)
        correct   = 'Y' if pred == left_out_label else 'N'

        fold_rows.append({
            'fold':           fold_i + 1,
            'participant_id': left_out_id,
            'true_label':     left_out_label,
            'group':          group_name,
            'n_train_trials': len(tr_idx),
            'n_test_trials':  len(y_test),
            'n_sig_feats':    len(sig),
            'prob_asd':       round(prob_mean, 4),
            'pred':           pred,
            'correct':        int(pred == left_out_label),
        })

        print(f"{fold_i+1:<6}{str(left_out_id):<14}{group_name:<8}"
              f"{prob_mean:<12.3f}{correct}")

    loocv_df   = pd.DataFrame(fold_rows)
    part_loocv = aggregate_to_participant(
        np.array(all_ids), np.array(all_true), np.array(all_probs))
    return loocv_df, part_loocv


# ── STEP 2 — FINAL MODEL + SAVE ──────────────────────────────────────────────

def train_final_model(df, significant_features):
    feat_cols = significant_features
    y = df['label'].values.astype(int)
    X = df[feat_cols].values.copy()

    pupil_idx = [i for i, f in enumerate(feat_cols)
                 if any(p in f for p in PUPIL_COLS)]
    scaler = None
    if len(pupil_idx) > 0:
        scaler = StandardScaler()
        X[:, pupil_idx] = scaler.fit_transform(X[:, pupil_idx])

    counts    = np.bincount(y.astype(int))
    scale_pos = float(counts[0]) / float(counts[1] + 1e-8)
    model     = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos)
    model.fit(X, y, verbose=False)
    cal = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    cal.fit(X, y)

    print(f"\n  Final model: {df['participant_id'].nunique()} participants, "
          f"{len(df)} trials, {len(feat_cols)} features.")

    # ── Save model bundle ─────────────────────────────────────────────────
    # Everything test_on_saliency.py needs is packed here:
    #   model         — calibrated classifier (predict_proba)
    #   raw_model     — raw XGBoost (for feature importance)
    #   scaler        — pupil StandardScaler fitted on ETSDS degraded data
    #   feature_names — exact list of features in correct order
    #   pupil_idx     — indices of pupil columns in feature_names
    #   train_means   — ETSDS training mean per feature (used to impute
    #                   missing Saliency4ASD features: sac_rate_per_min,
    #                   blink_rate_per_min, blink_dur_mean_ms)
    #   train_stds    — ETSDS training std per feature
    bundle = {
        'model':         cal,
        'raw_model':     model,
        'scaler':        scaler,
        'feature_names': feat_cols,
        'pupil_idx':     pupil_idx,
        'train_means':   dict(zip(feat_cols, df[feat_cols].mean().values)),
        'train_stds':    dict(zip(feat_cols, df[feat_cols].std().values)),
    }
    bundle_path = os.path.join(OUTPUT_DIR, 'asd_model_degraded.joblib')
    joblib.dump(bundle, bundle_path)
    print(f"  Model bundle saved: {bundle_path}")

    return cal, model, scaler


# ── STEP 3 — FEATURE IMPORTANCE ──────────────────────────────────────────────

def feature_importance(raw_model, df, significant_features, scaler):
    feat_cols = significant_features
    X = df[feat_cols].values.copy()
    y = df['label'].values.astype(int)

    if scaler is not None:
        pupil_idx = [i for i, f in enumerate(feat_cols)
                     if any(p in f for p in PUPIL_COLS)]
        if len(pupil_idx) > 0:
            X[:, pupil_idx] = scaler.transform(X[:, pupil_idx])

    gain_scores = raw_model.get_booster().get_score(importance_type='gain')
    gain_df = pd.DataFrame([
        {'feature': f, 'gain': gain_scores.get(f'f{i}', 0.0)}
        for i, f in enumerate(feat_cols)
    ]).sort_values('gain', ascending=False)

    perm = permutation_importance(raw_model, X, y, n_repeats=30,
                                  scoring='roc_auc', random_state=RANDOM_STATE)
    perm_df = pd.DataFrame({
        'feature':       feat_cols,
        'perm_imp_mean': perm.importances_mean.round(4),
        'perm_imp_std':  perm.importances_std.round(4),
    }).sort_values('perm_imp_mean', ascending=False)

    for top_df, xlabel, title, fname in [
        (perm_df.head(15), 'Permutation importance (AUC drop)',
         'Feature importance — permutation (top 15)', 'importance_permutation.png'),
        (gain_df.head(15), 'Mean gain (XGBoost)',
         'Feature importance — XGBoost gain (top 15)', 'importance_xgb_gain.png'),
    ]:
        col = 'perm_imp_mean' if 'perm' in fname else 'gain'
        err = top_df['perm_imp_std'] if 'perm' in fname else None
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_df['feature'][::-1], top_df[col][::-1],
                xerr=err[::-1] if err is not None else None,
                color='#1f77b4', capsize=3)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.tick_params(axis='y', labelsize=10)
        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {p}")

    combined = gain_df.merge(perm_df, on='feature', how='outer').fillna(0)
    combined = combined.sort_values('perm_imp_mean', ascending=False)
    combined.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'feature_importance.csv')}")

    try:
        import shap, tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp = f.name
        raw_model.save_model(tmp)
        m2 = XGBClassifier(); m2.load_model(tmp); os.unlink(tmp)
        expl      = shap.TreeExplainer(m2)
        shap_vals = expl.shap_values(X)
        if isinstance(shap_vals, list): shap_vals = shap_vals[1]
        if shap_vals.ndim == 3:        shap_vals = shap_vals[:, :, 1]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X, feature_names=feat_cols, show=False)
        plt.tight_layout()
        p3 = os.path.join(OUTPUT_DIR, 'shap_summary.png')
        plt.savefig(p3, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {p3}")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

    return combined


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  ASD Classification — LOOCV on Degraded ETSDS (57 participants)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df        = load_data()
    feat_cols = [c for c in df.columns if c not in META_COLS]
    n_parts   = df['participant_id'].nunique()
    n_asd     = df[df['label'] == 1]['participant_id'].nunique()
    n_td      = df[df['label'] == 0]['participant_id'].nunique()

    print(f"\nLoaded:  {n_parts} participants  (ASD={n_asd}, TD={n_td})")
    print(f"         {len(df)} trials  (~{len(df)//max(n_parts,1)} per participant)")
    print(f"         {len(feat_cols)} features")

    print("\n" + "─" * 70)
    print("  STEP 0 — Global Statistical Validation")
    print("─" * 70)
    significant, effect_sizes, p_values = global_feature_validation(df, feat_cols)
    print(f"\n  Features tested:      {len(feat_cols)}")
    print(f"  Features significant: {len(significant)}  (α={SIGNIFICANCE_ALPHA})")
    if len(significant) == 0:
        print("  No significant features. Check trial_features_degraded.csv.")
        sys.exit(1)
    effect_df = pd.DataFrame([
        {'Feature': f, 'p_value': round(p_values[f], 4),
         'effect_r': round(effect_sizes.get(f, 0.0) or 0.0, 3)}
        for f in significant
    ]).sort_values('effect_r', key=lambda x: x.abs(), ascending=False)
    print("\n  Significant features (sorted by |effect size|):")
    print(effect_df.to_string(index=False))
    stat_path = os.path.join(OUTPUT_DIR, 'statistical_validation.csv')
    effect_df.to_csv(stat_path, index=False)
    print(f"\n  Saved: {stat_path}")

    print("\n" + "─" * 70)
    print("  STEP 1 — Leave-One-Participant-Out CV")
    print("─" * 70)
    loocv_df, part_loocv = run_loocv(df, significant)
    loocv_metrics = print_participant_metrics(
        part_loocv, "LOOCV — pooled over all participants  [PRIMARY RESULT]")
    asd_r = part_loocv[part_loocv['y_true'] == 1]
    td_r  = part_loocv[part_loocv['y_true'] == 0]
    print(f"\n  ASD: {int(asd_r['correct'].sum())}/{len(asd_r)} correct  "
          f"(mean prob = {asd_r['y_prob_mean'].mean():.3f})")
    print(f"  TD:  {int(td_r['correct'].sum())}/{len(td_r)} correct  "
          f"(mean prob = {td_r['y_prob_mean'].mean():.3f})")
    loocv_df.to_csv(os.path.join(OUTPUT_DIR, 'loocv_fold_results.csv'), index=False)
    part_loocv.to_csv(os.path.join(OUTPUT_DIR, 'loocv_participant_predictions.csv'), index=False)
    pd.DataFrame([loocv_metrics]).to_csv(
        os.path.join(OUTPUT_DIR, 'loocv_summary.csv'), index=False)

    print("\n" + "─" * 70)
    print("  STEP 2 — Train Final Model (all participants) + Save")
    print("─" * 70)
    cal_model, raw_model, scaler = train_final_model(df, significant)

    print("\n" + "─" * 70)
    print("  STEP 3 — Feature Importance")
    print("─" * 70)
    imp_df = feature_importance(raw_model, df, significant, scaler)
    print("\n  Top 10 features by permutation importance:")
    print(imp_df[['feature', 'perm_imp_mean', 'perm_imp_std']].head(10)
          .to_string(index=False))

    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Dataset:   {n_parts} participants  (ASD={n_asd}, TD={n_td})")
    print(f"  Degraded:  60Hz SMI RED-m → 27.8Hz GazeTrack webcam profile")
    print(f"  Features:  {len(significant)} significant (from {len(feat_cols)})")
    print(f"\n  LOOCV [{n_parts} folds, each participant left out once]:")
    for k, v in loocv_metrics.items():
        print(f"    {k:<14}  {v:.3f}")
    print(f"\n  Original baseline (non-degraded): AUC=0.964  Sens=0.926  Spec=0.867")
    print(f"\n  Output files in: {os.path.abspath(OUTPUT_DIR)}/")
    for fname in ['statistical_validation.csv', 'loocv_fold_results.csv',
                  'loocv_participant_predictions.csv', 'loocv_summary.csv',
                  'feature_importance.csv', 'importance_permutation.png',
                  'importance_xgb_gain.png', 'asd_model_degraded.joblib']:
        print(f"    {fname}")
    print("=" * 70)


if __name__ == '__main__':
    main()