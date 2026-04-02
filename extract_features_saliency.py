"""
Saliency4ASD Feature Extractor
================================
Reads TrainingData/TrainingData/ASD/ and TrainingData/TrainingData/TD/
Produces trial_features_saliency.csv in the same folder as this script.

Run from: ASD_EYE_XGBOOST\
  python extract_features_saliency.py

FORMAT:
  Each file e.g. ASD_scanpath_1.txt = ALL 14 children for image 1.
  Columns: Idx, x, y, duration  (duration in milliseconds)
  A new child starts when Idx resets to 0 after the first row.

OUTPUT:
  trial_features_saliency.csv
  28 participants (ASD_01..ASD_14, TD_01..TD_14) x ~300 trials each
  13 features per trial
"""

import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── PATHS — matches ASD_EYE_XGBOOST folder structure ─────────────────────────
ASD_DIR      = os.path.join('TrainingData', 'TrainingData', 'ASD')
TD_DIR       = os.path.join('TrainingData', 'TrainingData', 'TD')
OUTPUT_CSV   = 'trial_features_saliency.csv'
MIN_FIXATIONS = 3


def parse_scanpath_file(path):
    """
    Parse one scanpath TXT file.
    Returns list of scanpaths — one per child in that file.
    Each scanpath = list of (x, y, duration_ms).
    Split rule: new child when Idx resets to 0 after first row.
    """
    try:
        df = pd.read_csv(path, header=0,
                         names=['Idx', 'x', 'y', 'duration'],
                         skipinitialspace=True)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df = df.astype(int)
    except Exception as e:
        print(f"  WARNING: could not parse {path}: {e}")
        return []

    if len(df) == 0:
        return []

    scanpaths, current = [], []
    for _, row in df.iterrows():
        if int(row['Idx']) == 0 and current:
            scanpaths.append(current)
            current = []
        current.append((int(row['x']), int(row['y']), int(row['duration'])))
    if current:
        scanpaths.append(current)
    return scanpaths


def compute_features(scanpath):
    """
    Compute 13 features from one scanpath (list of (x,y,dur_ms)).
    Returns dict or None if too few fixations.
    """
    if len(scanpath) < MIN_FIXATIONS:
        return None

    xs   = np.array([f[0] for f in scanpath], dtype=float)
    ys   = np.array([f[1] for f in scanpath], dtype=float)
    durs = np.array([f[2] for f in scanpath], dtype=float)

    fix_count       = len(durs)
    fix_dur_mean_ms = float(np.mean(durs))
    fix_dur_std_ms  = float(np.std(durs))
    fix_dur_max_ms  = float(np.max(durs))
    fix_dur_cv      = fix_dur_std_ms / (fix_dur_mean_ms + 1e-8)

    total_ms         = float(np.sum(durs))
    rec_dur_min      = max(total_ms / 60000.0, 1e-6)
    fix_rate_per_min = fix_count / rec_dur_min

    indices      = np.arange(fix_count, dtype=float)
    fix_seq_norm = float(np.mean(indices) / (np.max(indices) + 1e-8)) \
                   if fix_count > 1 else 0.0

    if fix_count >= 2:
        ifd      = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        ifd_mean = float(np.mean(ifd))
        ifd_std  = float(np.std(ifd))
    else:
        ifd_mean = ifd_std = 0.0

    return {
        'fix_count':        fix_count,
        'fix_dur_mean_ms':  round(fix_dur_mean_ms,  3),
        'fix_dur_std_ms':   round(fix_dur_std_ms,   3),
        'fix_dur_max_ms':   round(fix_dur_max_ms,   3),
        'fix_dur_cv':       round(fix_dur_cv,        4),
        'fix_rate_per_min': round(fix_rate_per_min,  4),
        'fix_seq_norm':     round(fix_seq_norm,       4),
        'ifd_mean':         round(ifd_mean,           3),
        'ifd_std':          round(ifd_std,            3),
        'gaze_x_mean':      round(float(np.mean(xs)), 3),
        'gaze_y_mean':      round(float(np.mean(ys)), 3),
        'gaze_x_std':       round(float(np.std(xs)),  3),
        'gaze_y_std':       round(float(np.std(ys)),  3),
    }


def process_class(scan_dir, label, class_name):
    files = sorted(
        [f for f in os.listdir(scan_dir) if f.endswith('.txt')],
        key=lambda f: int(''.join(filter(str.isdigit, f)))
    )
    print(f"\n  Processing {class_name}: {len(files)} image files...")

    all_scanpaths, n_per_image = [], []
    for fname in files:
        img_id    = int(''.join(filter(str.isdigit, fname)))
        scanpaths = parse_scanpath_file(os.path.join(scan_dir, fname))
        all_scanpaths.append((img_id, scanpaths))
        n_per_image.append(len(scanpaths))

    max_subj = max(n_per_image) if n_per_image else 0
    print(f"    Children per image: min={min(n_per_image)} "
          f"max={max_subj} mean={np.mean(n_per_image):.1f}")

    records, n_skip = [], 0
    for subj_idx in range(max_subj):
        pid = f"{class_name}_{subj_idx+1:02d}"
        for img_id, scanpaths in all_scanpaths:
            if subj_idx >= len(scanpaths):
                n_skip += 1
                continue
            feat = compute_features(scanpaths[subj_idx])
            if feat is None:
                n_skip += 1
                continue
            feat['participant_id'] = pid
            feat['trial']          = f"img_{img_id:03d}"
            feat['label']          = label
            records.append(feat)

    print(f"    Participants: {max_subj}  |  "
          f"Trials kept: {len(records)}  |  Skipped: {n_skip}")
    return records


def main():
    print("=" * 60)
    print("  Saliency4ASD Feature Extractor")
    print("=" * 60)

    for d, name in [(ASD_DIR, 'ASD'), (TD_DIR, 'TD')]:
        if not os.path.isdir(d):
            print(f"\n  ERROR: '{d}' not found.")
            print("  Run from inside ASD_EYE_XGBOOST\\")
            sys.exit(1)

    records  = process_class(ASD_DIR, label=1, class_name='ASD')
    records += process_class(TD_DIR,  label=0, class_name='TD')

    if not records:
        print("\n  ERROR: no records extracted.")
        sys.exit(1)

    df       = pd.DataFrame(records)
    feat_cols = [c for c in df.columns
                 if c not in ['participant_id', 'trial', 'label']]
    df = df[['participant_id', 'trial'] + feat_cols + ['label']]
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"  Saved: {OUTPUT_CSV}")
    print(f"  Participants: {df['participant_id'].nunique()}  "
          f"(ASD={df[df['label']==1]['participant_id'].nunique()}  "
          f"TD={df[df['label']==0]['participant_id'].nunique()})")
    print(f"  Total rows:   {len(df)}")
    print(f"  Features:     {len(feat_cols)}")
    print(f"\n  Next: python train_model_degraded.py")
    print(f"  Then: python test_on_saliency.py")
    print("=" * 60)


if __name__ == '__main__':
    main()