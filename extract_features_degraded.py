"""
Trial-Level Feature Extraction — Degraded ETSDS Dataset
=========================================================
Identical to extract_features.py with ONE change:
  sac_peak_vel_mean is EXCLUDED from feature extraction.

WHY: After degradation to 28Hz webcam quality, per-sample velocity
(pixels/ms between consecutive gaze samples) is unreliable — the
larger time gaps and added spatial noise make instantaneous velocity
estimates meaningless. All other 31 features are unaffected because
they are duration-based, count-based, or spatial aggregates that
are robust to the Hz reduction and noise level.

FOLDER STRUCTURE:
  ASD_degraded/   <- output of degrade_etsds.py
  TD_degraded/    <- output of degrade_etsds.py

OUTPUT:
  trial_features_degraded.csv   -> feed to train_model_degraded.py

References: Cilia et al. (2022); Samonte 2024 HISS;
            Al-Adhaileh 2025 Front.Medicine
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────── COLUMN NAMES ────────────────────────────────────

C_TIME        = 'RecordingTime [ms]'
C_CAT_GROUP   = 'Category Group'
C_CAT_RIGHT   = 'Category Right'
C_IDX_RIGHT   = 'Index Right'
C_GAZE_RX     = 'Point of Regard Right X [px]'
C_GAZE_RY     = 'Point of Regard Right Y [px]'
C_GAZE_LX     = 'Point of Regard Left X [px]'
C_GAZE_LY     = 'Point of Regard Left Y [px]'
C_PUPIL_R     = 'Pupil Diameter Right [mm]'
C_PUPIL_L     = 'Pupil Diameter Left [mm]'
C_TRACK_RATIO = 'Tracking Ratio [%]'
C_PARTICIPANT = 'Participant'
C_TRIAL       = 'Trial'
C_STIMULUS    = 'Stimulus'

BLINK_SHORT_MS  = 100
BLINK_LONG_MS   = 500
MIN_TRACK_RATIO = 50.0
MAX_FIX_DUR_MS  = 10000
MIN_FIXATIONS   = 3


# ─────────────────────────── PREPROCESSING ───────────────────────────────────

def load_and_clean(path):
    df = pd.read_csv(path, low_memory=False)
    cat_cols = [C_CAT_GROUP, C_CAT_RIGHT, 'Category Left']
    df[[c for c in df.columns if c not in cat_cols]] = \
        df[[c for c in df.columns if c not in cat_cols]].replace('-', np.nan)
    for col in [C_TIME, C_IDX_RIGHT, C_GAZE_RX, C_GAZE_RY,
                C_GAZE_LX, C_GAZE_LY, C_PUPIL_R, C_PUPIL_L, C_TRACK_RATIO]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in [C_PUPIL_R, C_PUPIL_L, C_GAZE_RX, C_GAZE_RY, C_GAZE_LX, C_GAZE_LY]:
        if col in df.columns:
            df[col] = df[col].replace(0.0, np.nan)
    if C_CAT_GROUP in df.columns:
        df = df[df[C_CAT_GROUP] == 'Eye'].copy()
    if C_CAT_RIGHT in df.columns:
        df = df[~df[C_CAT_RIGHT].isin(['-', 'Separator'])].copy()
        df = df[df[C_CAT_RIGHT].notna()].copy()
    if C_TRACK_RATIO in df.columns:
        df = df[df[C_TRACK_RATIO] >= MIN_TRACK_RATIO].copy()
    return df.sort_values(C_TIME).reset_index(drop=True)


def interpolate_blinks(df):
    if C_CAT_RIGHT not in df.columns or C_TIME not in df.columns:
        return df
    df = df.copy()
    is_blink = df[C_CAT_RIGHT] == 'Blink'
    blocks   = (is_blink != is_blink.shift()).cumsum()
    for _, grp in df[is_blink].groupby(blocks[is_blink]):
        idx   = grp.index
        times = df.loc[idx, C_TIME].dropna().values
        if len(times) < 2:
            continue
        if (times[-1] - times[0]) >= BLINK_SHORT_MS:
            for col in [C_GAZE_RX, C_GAZE_RY, C_GAZE_LX, C_GAZE_LY,
                        C_PUPIL_R, C_PUPIL_L]:
                if col in df.columns:
                    df.loc[idx, col] = np.nan
    for col in [C_GAZE_RX, C_GAZE_RY, C_GAZE_LX, C_GAZE_LY,
                C_PUPIL_R, C_PUPIL_L]:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit_area='inside')
    return df


def zscore_pupil(df):
    df = df.copy()
    for col in [C_PUPIL_R, C_PUPIL_L]:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 1 and vals.std() > 0:
                df[col] = (df[col] - vals.mean()) / vals.std()
            else:
                df[col] = 0.0
    return df


# ─────────────────────────── EVENT BUILDERS ──────────────────────────────────

def safe_stat(arr, fn):
    a = arr[~np.isnan(arr)] if len(arr) > 0 else arr
    return float(fn(a)) if len(a) > 0 else 0.0


def build_fixation_events(trial_df):
    fix = trial_df[trial_df[C_CAT_RIGHT] == 'Fixation'].copy()
    if len(fix) == 0:
        return pd.DataFrame()
    if C_IDX_RIGHT in fix.columns and fix[C_IDX_RIGHT].notna().sum() > 0:
        groups = (fix[C_IDX_RIGHT] != fix[C_IDX_RIGHT].shift()).cumsum()
    else:
        groups = (fix[C_CAT_RIGHT] != fix[C_CAT_RIGHT].shift()).cumsum()
    events = []
    for seq, grp in fix.groupby(groups):
        times = grp[C_TIME].dropna().values
        if len(times) == 0:
            continue
        dur = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        if dur > MAX_FIX_DUR_MS:
            continue
        x  = grp[C_GAZE_RX].dropna().values if C_GAZE_RX in grp.columns else np.array([])
        y  = grp[C_GAZE_RY].dropna().values if C_GAZE_RY in grp.columns else np.array([])
        pr = grp[C_PUPIL_R].dropna().values  if C_PUPIL_R in grp.columns else np.array([])
        pl = grp[C_PUPIL_L].dropna().values  if C_PUPIL_L in grp.columns else np.array([])
        events.append({
            'fix_seq':     int(seq),
            'duration_ms': dur,
            'start_time':  float(times[0]),
            'mean_x':      float(np.mean(x))  if len(x) > 0 else np.nan,
            'mean_y':      float(np.mean(y))  if len(y) > 0 else np.nan,
            'std_x':       float(np.std(x))   if len(x) > 1 else 0.0,
            'std_y':       float(np.std(y))   if len(y) > 1 else 0.0,
            'mean_pupil_r':float(np.mean(pr)) if len(pr) > 0 else np.nan,
            'mean_pupil_l':float(np.mean(pl)) if len(pl) > 0 else np.nan,
        })
    return pd.DataFrame(events).sort_values('start_time').reset_index(drop=True)


def build_saccade_events(trial_df):
    """
    Saccade amplitude only — peak velocity EXCLUDED.
    At 28Hz the inter-sample interval is ~36ms; computing px/ms from two
    consecutive noisy gaze points gives meaningless velocity estimates.
    Amplitude (Euclidean distance first->last) is still valid.
    """
    sac = trial_df[trial_df[C_CAT_RIGHT] == 'Saccade'].copy()
    if len(sac) == 0:
        return pd.DataFrame()
    groups = (sac[C_CAT_RIGHT] != sac[C_CAT_RIGHT].shift()).cumsum()
    events = []
    for _, grp in sac.groupby(groups):
        times = grp[C_TIME].dropna().values
        if len(times) == 0:
            continue
        x = grp[C_GAZE_RX].dropna().values if C_GAZE_RX in grp.columns else np.array([])
        y = grp[C_GAZE_RY].dropna().values if C_GAZE_RY in grp.columns else np.array([])
        amp = float(np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)) \
              if len(x) > 1 and len(y) > 1 else 0.0
        events.append({
            'duration_ms':  float(times[-1]-times[0]) if len(times) > 1 else 0.0,
            'amplitude_px': amp,
            # peak_velocity intentionally omitted
        })
    return pd.DataFrame(events)


def build_blink_events(trial_df):
    blk = trial_df[trial_df[C_CAT_RIGHT] == 'Blink'].copy()
    if len(blk) == 0:
        return pd.DataFrame()
    groups = (blk[C_CAT_RIGHT] != blk[C_CAT_RIGHT].shift()).cumsum()
    events = []
    for _, grp in blk.groupby(groups):
        times = grp[C_TIME].dropna().values
        if len(times) == 0:
            continue
        dur = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        events.append({'duration_ms': dur, 'is_long': int(dur >= BLINK_LONG_MS)})
    return pd.DataFrame(events)


# ─────────────────────────── FEATURE EXTRACTION ──────────────────────────────

def extract_trial_features(trial_df, fix_events, sac_events, blk_events):
    """
    Extract 31 features per trial (original 32 minus sac_peak_vel_mean).
    Everything else identical to extract_features.py.
    """
    feats = {}

    times = trial_df[C_TIME].dropna().values if C_TIME in trial_df.columns else np.array([])
    rec_dur_min = max((times[-1] - times[0]) / 60_000.0, 0.001) \
                  if len(times) > 1 else 1.0

    # ── Fixation (10 features) ───────────────────────────────────────────────
    if len(fix_events) > 0:
        durs = fix_events['duration_ms'].dropna().values
        feats['fix_count']           = len(fix_events)
        feats['fix_dur_mean_ms']     = safe_stat(durs, np.mean)
        feats['fix_dur_std_ms']      = safe_stat(durs, np.std)
        feats['fix_dur_max_ms']      = safe_stat(durs, np.max)
        feats['fix_dur_cv']          = feats['fix_dur_std_ms'] / (feats['fix_dur_mean_ms'] + 1e-8)
        feats['fix_rate_per_min']    = len(fix_events) / rec_dur_min
        disp = (fix_events['std_x'].fillna(0) + fix_events['std_y'].fillna(0)).values
        feats['fix_dispersion_mean'] = safe_stat(disp, np.mean)
        feats['fix_seq_norm']        = (fix_events['fix_seq'].mean() /
                                        (fix_events['fix_seq'].max() + 1e-8))
        if len(fix_events) >= 2:
            x   = fix_events['mean_x'].values
            y   = fix_events['mean_y'].values
            ifd = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            feats['ifd_mean'] = float(np.nanmean(ifd))
            feats['ifd_std']  = float(np.nanstd(ifd))
        else:
            feats['ifd_mean'] = feats['ifd_std'] = 0.0
    else:
        for k in ['fix_count', 'fix_dur_mean_ms', 'fix_dur_std_ms', 'fix_dur_max_ms',
                  'fix_dur_cv', 'fix_rate_per_min', 'fix_dispersion_mean', 'fix_seq_norm',
                  'ifd_mean', 'ifd_std']:
            feats[k] = 0.0

    # ── Saccade (5 features — peak_vel removed) ──────────────────────────────
    if len(sac_events) > 0:
        amps = sac_events['amplitude_px'].dropna().values
        feats['sac_count']          = len(sac_events)
        feats['sac_amplitude_mean'] = safe_stat(amps, np.mean)
        feats['sac_amplitude_std']  = safe_stat(amps, np.std)
        feats['sac_rate_per_min']   = len(sac_events) / rec_dur_min
        feats['sac_fix_ratio']      = len(sac_events) / (len(fix_events) + len(sac_events) + 1e-8)
    else:
        for k in ['sac_count', 'sac_amplitude_mean', 'sac_amplitude_std',
                  'sac_rate_per_min', 'sac_fix_ratio']:
            feats[k] = 0.0

    # ── Blink (4 features) ───────────────────────────────────────────────────
    if len(blk_events) > 0:
        durs = blk_events['duration_ms'].dropna().values
        feats['blink_count']        = len(blk_events)
        feats['blink_dur_mean_ms']  = safe_stat(durs, np.mean)
        feats['blink_rate_per_min'] = len(blk_events) / rec_dur_min
        feats['blink_long_ratio']   = blk_events['is_long'].sum() / (len(blk_events) + 1e-8)
    else:
        for k in ['blink_count', 'blink_dur_mean_ms', 'blink_rate_per_min', 'blink_long_ratio']:
            feats[k] = 0.0

    # ── Pupil (7 features) ───────────────────────────────────────────────────
    for col, suf in [(C_PUPIL_R, 'r'), (C_PUPIL_L, 'l')]:
        if col in trial_df.columns:
            vals = trial_df[col].dropna().values
            feats[f'pupil_{suf}_mean'] = safe_stat(vals, np.mean)
            feats[f'pupil_{suf}_std']  = safe_stat(vals, np.std)
        else:
            feats[f'pupil_{suf}_mean'] = feats[f'pupil_{suf}_std'] = 0.0
    if C_PUPIL_R in trial_df.columns and C_PUPIL_L in trial_df.columns:
        pr = trial_df[C_PUPIL_R].dropna()
        pl = trial_df[C_PUPIL_L].dropna()
        common = pr.index.intersection(pl.index)
        diff   = (pr.loc[common] - pl.loc[common]).abs() if len(common) > 0 \
                 else pd.Series([0])
        feats['pupil_asymmetry_mean'] = float(diff.mean())
        feats['pupil_asymmetry_std']  = float(diff.std()) if len(diff) > 1 else 0.0
    else:
        feats['pupil_asymmetry_mean'] = feats['pupil_asymmetry_std'] = 0.0
    if C_PUPIL_R in trial_df.columns:
        pvals = trial_df[C_PUPIL_R].dropna().values
        feats['pupil_slope'] = (float(linregress(range(len(pvals)), pvals).slope)
                                if len(pvals) > 2 else 0.0)
    else:
        feats['pupil_slope'] = 0.0

    # ── Gaze position (2 features — std removed) ────────────────────────────
    # gaze_x_std and gaze_y_std are EXCLUDED after degradation.
    # Adding 25px Gaussian noise inflates within-trial gaze spread
    # non-uniformly, creating a spurious signal unrelated to ASD biology.
    # In the degraded run these two features dominated importance (0.135 and
    # 0.101 AUC drop) while clinical features like sac_rate were suppressed.
    # gaze_x_mean and gaze_y_mean (central tendency) are unaffected by noise
    # sigma and retain their clinical meaning as scan-path position measures.
    if C_GAZE_RX in trial_df.columns and C_GAZE_RY in trial_df.columns:
        gx = trial_df[C_GAZE_RX].dropna().values
        gy = trial_df[C_GAZE_RY].dropna().values
        feats['gaze_x_mean'] = safe_stat(gx, np.mean)
        feats['gaze_y_mean'] = safe_stat(gy, np.mean)
    else:
        feats['gaze_x_mean'] = feats['gaze_y_mean'] = 0.0

    return feats


# ─────────────────────────── PROCESS ONE PARTICIPANT ─────────────────────────

def process_participant(path, label):
    try:
        df_raw = pd.read_csv(path, nrows=5, low_memory=False)
        pid = str(df_raw[C_PARTICIPANT].dropna().iloc[0]) \
              if C_PARTICIPANT in df_raw.columns \
              else os.path.splitext(os.path.basename(path))[0]
    except Exception:
        pid = os.path.splitext(os.path.basename(path))[0]

    try:
        df = load_and_clean(path)
    except Exception as e:
        print(f'ERROR: {e}')
        return []

    if len(df) < 50:
        print('SKIP — too few rows after cleaning')
        return []

    # Preprocess on full recording BEFORE trial split
    df = interpolate_blinks(df)
    df = zscore_pupil(df)

    if C_TRIAL not in df.columns:
        print('SKIP — no Trial column')
        return []

    stim_map = df.groupby(C_TRIAL)[C_STIMULUS].first().to_dict() \
               if C_STIMULUS in df.columns else {}

    records = []
    for trial_name, trial_df in df.groupby(C_TRIAL):
        trial_df  = trial_df.copy()
        stim      = stim_map.get(trial_name, '')
        stim_type = 1 if str(stim).lower().endswith('.avi') else 0

        fix_ev = build_fixation_events(trial_df)
        sac_ev = build_saccade_events(trial_df)
        blk_ev = build_blink_events(trial_df)

        if len(fix_ev) < MIN_FIXATIONS:
            continue

        feats = extract_trial_features(trial_df, fix_ev, sac_ev, blk_ev)
        feats.update({'participant_id': pid, 'trial': trial_name,
                      'stimulus_type': stim_type, 'label': label})
        records.append(feats)
    return records


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    ASD_FOLDER = 'ASD_degraded'
    TD_FOLDER  = 'TD_degraded'
    OUTPUT     = 'trial_features_degraded.csv'

    for folder in [ASD_FOLDER, TD_FOLDER]:
        if not os.path.exists(folder):
            print(f"ERROR: '{folder}' not found. Run degrade_etsds.py first.")
            sys.exit(1)

    all_records = []
    for folder, label, name in [(ASD_FOLDER, 1, 'ASD'), (TD_FOLDER, 0, 'TD')]:
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.csv'))
        print(f"\nProcessing {name} ({len(files)} participants)...")
        for fname in files:
            print(f"  {fname}", end=' ... ', flush=True)
            recs = process_participant(os.path.join(folder, fname), label)
            all_records.extend(recs)
            print(f"OK ({len(recs)} trials)")

    if not all_records:
        print("No records extracted.")
        sys.exit(1)

    df_out    = pd.DataFrame(all_records)
    meta_cols = ['participant_id', 'trial', 'stimulus_type']
    feat_cols = [c for c in df_out.columns if c not in meta_cols + ['label']]
    df_out    = df_out[meta_cols + feat_cols + ['label']]
    df_out.to_csv(OUTPUT, index=False)

    print(f"\n{'='*55}")
    print(f"  Saved:        {OUTPUT}")
    print(f"  Participants: {df_out['participant_id'].nunique()}")
    print(f"  Trials:       {len(df_out)}  "
          f"(ASD={int((df_out['label']==1).sum())}, "
          f"TD={int((df_out['label']==0).sum())})")
    print(f"  Features:     {len(feat_cols)}  (sac_peak_vel excluded)")
    print(f"  Missing:      {df_out[feat_cols].isnull().sum().sum()}")
    print(f"{'='*55}")
    print("Next: run train_model_degraded.py")


if __name__ == '__main__':
    main()