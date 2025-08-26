#!/usr/bin/env python3
# clean_arm_data.py
#
# USAGE:
#     python clean_arm_data.py  raw_arm_dataset.csv  cleaned_arm_dataset.csv

import sys, json, math
import numpy as np
import pandas as pd
from sklearn.utils import resample

RAW_CSV  = sys.argv[1]          # input file
CLEAN_CSV = sys.argv[2]         # output file

# -------------------------------------------------------------
# PARAMETERS you may want to tweak
# -------------------------------------------------------------
PIX_X_MAX  = 640.0
PIX_Y_MAX  = 480.0
ANGLE_MAX  =  1.6            # slightly larger than ±1.5, acts as guard
PIX_TOL    =  2              # consider rows identical if every pixel
                             # value differs ≤ PIX_TOL
MAX_ZSCORE =  3.5            # for statistical outlier removal
BALANCE_BINS = 8             # how many equal-width bins per joint angle
                             # (set 0 to skip balancing)
RANDOM_STATE = 42

# -------------------------------------------------------------
# COLUMN names
# -------------------------------------------------------------
PIX_COLS = ['cam1_tip_x','cam1_tip_y',
            'cam2_tip_x','cam2_tip_y',
            'cam3_tip_x','cam3_tip_y']

ANGLE_COLS = ['shoulder_yaw','shoulder_pitch',
              'elbow_pitch','wrist_roll_1','wrist_yaw']

ALL_COLS = ANGLE_COLS + PIX_COLS

# -------------------------------------------------------------
# 1. Read + coerce to numeric
# -------------------------------------------------------------
na_tokens = ['', ' ', 'NA', 'N/A', 'nan', 'NaN', 'None']
df = pd.read_csv(RAW_CSV, sep=';', na_values=na_tokens)

for c in ALL_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce')

print(f'Loaded {len(df):,} rows')

# -------------------------------------------------------------
# 2. Drop rows with any NaN
# -------------------------------------------------------------
df = df.dropna(subset=ALL_COLS)
print(f'After NaN removal          : {len(df):,}')

# -------------------------------------------------------------
# 3. Enforce legal numeric ranges
# -------------------------------------------------------------
mask_pix  = (
    (df['cam1_tip_x'].between(0, PIX_X_MAX)) &
    (df['cam2_tip_x'].between(0, PIX_X_MAX)) &
    (df['cam3_tip_x'].between(0, PIX_X_MAX)) &
    (df['cam1_tip_y'].between(0, PIX_Y_MAX)) &
    (df['cam2_tip_y'].between(0, PIX_Y_MAX)) &
    (df['cam3_tip_y'].between(0, PIX_Y_MAX))
)

mask_ang  = df[ANGLE_COLS].abs().le(ANGLE_MAX).all(axis=1)
before = len(df)
df = df[mask_pix & mask_ang].reset_index(drop=True)
print(f'After range filter         : {len(df):,} '
      f'({before-len(df):,} removed)')

# -------------------------------------------------------------
# 4. Statistical outlier removal (z-score over the whole table)
# -------------------------------------------------------------
z = ((df[ALL_COLS] - df[ALL_COLS].mean()) / df[ALL_COLS].std()).abs()
mask_z = (z < MAX_ZSCORE).all(axis=1)
before = len(df)
df = df[mask_z].reset_index(drop=True)
print(f'After z-score filter       : {len(df):,} '
      f'({before-len(df):,} removed)')

# -------------------------------------------------------------
# 5. Remove exact duplicates
# -------------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset=ALL_COLS).reset_index(drop=True)
print(f'After exact duplicate drop : {len(df):,} '
      f'({before-len(df):,} removed)')

# -------------------------------------------------------------
# 6. Remove near-duplicates (within PIX_TOL pixels every column)
# -------------------------------------------------------------
# Approach: round pixel columns to PIX_TOL grid, then drop duplicates
def _round_series(s, tol):
    return (s / tol).round().astype(int)

rounded = df.copy()
for col in PIX_COLS:
    rounded[col] = _round_series(rounded[col], PIX_TOL)

before = len(df)
dupes = rounded.duplicated(subset=PIX_COLS, keep='first')
df = df.loc[~dupes].reset_index(drop=True)
print(f'After near-duplicate drop  : {len(df):,} '
      f'({before-len(df):,} removed)')

# -------------------------------------------------------------
# 7. OPTIONAL – balance the joint-angle distribution
# -------------------------------------------------------------
if BALANCE_BINS > 0:
    print('\nBalancing dataset …')
    bins = np.linspace(-ANGLE_MAX, ANGLE_MAX, BALANCE_BINS + 1)
    frames = []
    for joint in ANGLE_COLS:
        # put each row into one bin for this joint
        df['bin'] = np.digitize(df[joint], bins)
        # find the minimum bin size
        counts = df['bin'].value_counts()
        target_n = counts.min()
        # downsample each bin to the same size
        parts = []
        for b in counts.index:
            subset = df[df['bin'] == b]
            subset_down = resample(subset,
                                   replace=False,
                                   n_samples=target_n,
                                   random_state=RANDOM_STATE)
            parts.append(subset_down)
        frames.append(pd.concat(parts, ignore_index=True))
    # union over joints, then drop duplicates again
    df_balanced = pd.concat(frames, ignore_index=True)
    df_balanced = df_balanced.drop(columns=['bin'])
    before = len(df)
    df = df_balanced.drop_duplicates(subset=ALL_COLS).reset_index(drop=True)
    print(f'After balancing            : {len(df):,} '
          f'(was {before:,})')

# -------------------------------------------------------------
# 8. Save
# -------------------------------------------------------------
df.to_csv(CLEAN_CSV, sep=';', index=False)
print(f'\nSaved cleaned file: {CLEAN_CSV}')