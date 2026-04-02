"""
Degrade All ETSDS Participants to GazeTrack Webcam Quality
===========================================================
Reference device : GazeTrack v21 webcam (TEST_21)
Source data      : ETSDS 57 participants — SMI RED-m, 60 Hz
Output           : ASD_degraded/ and TD_degraded/

DEGRADATION PARAMETERS (measured from TEST_21):
  Downsample    : 60 Hz -> 27.8 Hz  (keep rows >= 36 ms apart)
  Spatial noise : Gaussian sigma = 25 px on gaze X/Y (both eyes)
  Tracking loss : 6.8% of Eye rows get gaze/pupil nulled (blocks 1-4)

CRITICAL DESIGN RULE — GAZE-ONLY LOSS:
  Tracking loss nulls ONLY gaze and pupil columns.
  Category Right and Tracking Ratio [%] are NOT modified.

  Bug in v1: setting Category Right to '-' caused load_and_clean()
  to drop those rows. Short image trials (2-3s, 3-8 fixation events)
  lost entire fixation events, pushing below MIN_FIXATIONS=3.
  Result: 68% of trials dropped (593 vs ~1881), 4 participants lost,
  only 5 significant features (vs 16), AUC collapsed to 0.633.

  Fix: null gaze values only. The event label and tracking ratio stay
  intact, so load_and_clean keeps the row. Blink interpolation then
  bridges the missing gaze values exactly as it does for real data.
"""

import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

ASD_IN  = 'ASD'
TD_IN   = 'TD'
ASD_OUT = 'ASD_degraded'
TD_OUT  = 'TD_degraded'

TARGET_INTERVAL_MS = 36.0
SPATIAL_SIGMA      = 25.0
LOSS_RATE          = 0.068
LOSS_MIN_BLOCK     = 1
LOSS_MAX_BLOCK     = 4
RANDOM_SEED        = 42

C_TIME      = 'RecordingTime [ms]'
C_CAT_GROUP = 'Category Group'
C_GAZE_RX   = 'Point of Regard Right X [px]'
C_GAZE_RY   = 'Point of Regard Right Y [px]'
C_GAZE_LX   = 'Point of Regard Left X [px]'
C_GAZE_LY   = 'Point of Regard Left Y [px]'
C_PUPIL_R   = 'Pupil Diameter Right [mm]'
C_PUPIL_L   = 'Pupil Diameter Left [mm]'
C_TRACK     = 'Tracking Ratio [%]'

GAZE_COLS = [C_GAZE_RX, C_GAZE_RY, C_GAZE_LX, C_GAZE_LY]
# GAZE + PUPIL nulled during loss — Category Right and Tracking Ratio intentionally excluded
LOSS_COLS = GAZE_COLS + [C_PUPIL_R, C_PUPIL_L]


def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df.replace('-', np.nan, inplace=True)
    for col in [C_TIME] + LOSS_COLS + [C_TRACK]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def downsample(df):
    """
    Downsample Eye rows to TARGET_INTERVAL_MS, operating per trial.

    WHY PER TRIAL:
    ETSDS CSVs are not globally sorted by RecordingTime — trials can be
    stored out of order (noted in the original extract_features.py header).
    Global downsampling collapses because a backward timestamp jump between
    trials (e.g. -414,924 ms) means t_new - last_t < 36 ms for every row
    in that trial, so only 1-2 rows survive. Some participants went from
    32,851 -> 1,093 rows (3.3%) instead of the expected ~50%.

    Fix: sort each trial's Eye rows by RecordingTime independently, then
    apply the time threshold within that trial. Non-Eye rows are kept as-is.
    """
    if C_TIME not in df.columns or C_CAT_GROUP not in df.columns:
        return df, 0, 0

    df = df.copy()
    has_trial = 'Trial' in df.columns

    eye_mask    = df[C_CAT_GROUP] == 'Eye'
    non_eye_idx = df.index[~eye_mask].tolist()
    eye_df      = df[eye_mask].copy()

    n_eye_orig = len(eye_df)
    if n_eye_orig == 0:
        return df, 0, 0

    keep_eye_idx = []

    if has_trial:
        # Downsample within each trial separately (ETSDS: timestamps not globally sorted)
        for _, grp in eye_df.groupby('Trial', sort=False):
            grp_sorted = grp.sort_values(C_TIME)
            times = grp_sorted[C_TIME].values
            idxs  = grp_sorted.index.tolist()
            if len(idxs) == 0:
                continue
            keep_eye_idx.append(idxs[0])
            last_t = times[0]
            for i in range(1, len(idxs)):
                t = times[i]
                if not np.isnan(t) and (t - last_t) >= TARGET_INTERVAL_MS:
                    keep_eye_idx.append(idxs[i])
                    last_t = t
    else:
        # No trial column: sort globally and downsample
        eye_sorted = eye_df.sort_values(C_TIME)
        times = eye_sorted[C_TIME].values
        idxs  = eye_sorted.index.tolist()
        keep_eye_idx.append(idxs[0])
        last_t = times[0]
        for i in range(1, len(idxs)):
            t = times[i]
            if not np.isnan(t) and (t - last_t) >= TARGET_INTERVAL_MS:
                keep_eye_idx.append(idxs[i])
                last_t = t

    keep_all = sorted(set(keep_eye_idx) | set(non_eye_idx))
    df_out   = df.loc[keep_all].reset_index(drop=True)
    return df_out, n_eye_orig, len(keep_eye_idx)


def add_spatial_noise(df, rng):
    df = df.copy()
    eye_mask = df[C_CAT_GROUP] == 'Eye' if C_CAT_GROUP in df.columns \
               else pd.Series(True, index=df.index)
    for col in GAZE_COLS:
        if col not in df.columns:
            continue
        vals  = df.loc[eye_mask, col].values.copy().astype(float)
        valid = ~np.isnan(vals)
        vals[valid] += rng.normal(0.0, SPATIAL_SIGMA, size=valid.sum())
        vals = np.clip(vals, 0, 1920) if 'X' in col else np.clip(vals, 0, 1080)
        df.loc[eye_mask, col] = vals
    return df


def inject_loss(df, rng):
    """
    Null gaze + pupil values only. Category Right and Tracking Ratio
    are intentionally left unchanged so event structure is preserved.
    """
    df = df.copy()
    eye_mask = df[C_CAT_GROUP] == 'Eye' if C_CAT_GROUP in df.columns \
               else pd.Series(True, index=df.index)
    eye_idx = df.index[eye_mask].tolist()
    n = len(eye_idx)
    if n == 0:
        return df, 0, 0
    target = int(round(n * LOSS_RATE))
    n_lost = 0
    for _ in range(n * 20):
        if n_lost >= target:
            break
        block = int(rng.integers(LOSS_MIN_BLOCK, LOSS_MAX_BLOCK + 1))
        start = int(rng.integers(0, max(1, n - block)))
        rows  = eye_idx[start:min(start + block, n)]
        for col in LOSS_COLS:
            if col in df.columns:
                df.loc[rows, col] = np.nan   # gaze/pupil only — event labels intact
        n_lost += len(rows)
    return df, n_lost, n


def degrade_file(path, out_path, rng_noise, rng_loss):
    df = load_csv(path)
    df, n_orig, n_kept = downsample(df)
    df = add_spatial_noise(df, rng_noise)
    df, n_lost, n_eye = inject_loss(df, rng_loss)
    df.to_csv(out_path, index=False)
    loss_pct = round(100.0 * n_lost / max(n_eye, 1), 1)
    print(f'  {os.path.basename(path):<30}  {n_orig:>5} -> {n_kept:>5} rows'
          f'  ({1000/TARGET_INTERVAL_MS:.0f} Hz)   gaze loss: {loss_pct}%')


def main():
    print('=' * 72)
    print('  ETSDS Degradation v2 — SMI RED-m 60Hz -> GazeTrack 27.8Hz')
    print('  Fix: loss injection nulls gaze only, preserves event labels')
    print('=' * 72)
    print(f'\n  sigma={SPATIAL_SIGMA}px  loss={LOSS_RATE*100:.1f}%  '
          f'interval={TARGET_INTERVAL_MS}ms  blocks={LOSS_MIN_BLOCK}-{LOSS_MAX_BLOCK}')

    for in_dir, out_dir in [(ASD_IN, ASD_OUT), (TD_IN, TD_OUT)]:
        if not os.path.isdir(in_dir):
            print(f'\nERROR: "{in_dir}" not found.'); sys.exit(1)
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith('.csv'))
        label = 'ASD' if in_dir == ASD_IN else 'TD'
        print(f'\n  {label} ({len(files)} files)  {in_dir}/ -> {out_dir}/')
        print(f'  {"File":<30}  {"Eye rows":>22}   Gaze loss')
        print('  ' + '-' * 64)
        rng_noise = np.random.default_rng(RANDOM_SEED)
        rng_loss  = np.random.default_rng(RANDOM_SEED + 1)
        for fname in files:
            try:
                degrade_file(os.path.join(in_dir, fname),
                             os.path.join(out_dir, fname),
                             rng_noise, rng_loss)
            except Exception as e:
                print(f'  ERROR {fname}: {e}')

    print(f'\n{"=" * 72}')
    print(f'  Done. Next: python extract_features_degraded.py')
    print('=' * 72)


if __name__ == '__main__':
    main()
