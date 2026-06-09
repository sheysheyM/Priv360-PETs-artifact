from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import re
import numpy as np
import pandas as pd


# Dataset and experiment settings

DS1 = {
    'default_root': '/mimer/NOBACKUP/groups/naiss2024-22-977/Formated_Data',
    'base_cols': [
        'UnitQuaternion.x', 'UnitQuaternion.y',
        'UnitQuaternion.z', 'UnitQuaternion.w',
        'HmdPosition.x',    'HmdPosition.y',    'HmdPosition.z',
    ],
    'label_col': 'UserID',
    'video_col': 'video_id2',
    'epochs':    5,
    'multi_gpu': False,
}

DS2 = {
    'default_root': '/mimer/NOBACKUP/groups/naiss2024-22-977/DataPrive360/DataPrive360',
    'base_cols': ['qx', 'qy', 'qz', 'qw', 'pos_x', 'pos_y', 'pos_z'],
    'label_col': 'participant',
    'video_col': 'video',
    'epochs':    30,
    'multi_gpu': True,
}

CONFIGS = {'ds1': DS1, 'ds2': DS2}

SEQ_LEN             = 10
SEED                = 42
ALPHA               = 0.5
SIGMA               = 5.0
BATCH               = 32768
STEPS_PER_EPOCH     = 300
PATIENCE            = 3
LR_PATIENCE         = 2
SHUFFLE_BUFFER      = 20_000
PARTIAL_CLEAN_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
RF_TREES            = 100
RF_DEPTH            = 20
RF_JOBS             = -1
SIGMAS_NOISY        = [1.0, 5.0, 20.0]


# Dataset loading helpers

def load_ds1(root_dir, base_cols, label_col, video_col, seq_len=10):
    records = []
    for experiment_id in sorted(os.listdir(root_dir)):
        ep = os.path.join(root_dir, experiment_id)
        if not os.path.isdir(ep):
            continue
        for user_id in sorted(os.listdir(ep)):
            up = os.path.join(ep, user_id)
            if not os.path.isdir(up):
                continue
            for fn in sorted(os.listdir(up)):
                if not fn.endswith('.csv'):
                    continue
                fp = os.path.join(up, fn)
                try:
                    df_tmp = pd.read_csv(fp, engine='c', low_memory=False)
                except Exception as exc:
                    print(f'  [WARN] {fp}: {exc}')
                    continue
                if any(c not in df_tmp.columns for c in base_cols):
                    continue
                df_tmp[label_col] = str(user_id).strip()
                df_tmp[video_col] = str(experiment_id).strip() + '_' + os.path.splitext(fn)[0].strip()
                for c in base_cols:
                    df_tmp[c] = pd.to_numeric(df_tmp[c], errors='coerce').astype('float32')
                df_tmp = df_tmp.dropna(subset=base_cols)
                if len(df_tmp) < seq_len:
                    continue
                records.append(df_tmp[[label_col, video_col] + base_cols])
    if not records:
        raise SystemExit(f'No valid data in {root_dir}')
    df = pd.concat(records, ignore_index=True)
    df[label_col] = df[label_col].astype(str).str.strip()
    df[video_col] = df[video_col].astype(str).str.strip()
    print(f'[INFO] DS1 {len(df):,} rows | {df[video_col].nunique()} videos | {df[label_col].nunique()} users')
    return df


def load_ds2(root_dir, base_cols, label_col, video_col):
    data_path = os.path.join(root_dir, '_qoe_batch_runs')
    if os.path.isdir(data_path):
        csvs = sorted(f for f in os.listdir(data_path)
                      if f.startswith('dataprive360_ar2') and f.endswith('.csv'))
        if csvs:
            latest = os.path.join(data_path, csvs[-1])
            print(f'[INFO] Loading preprocessed: {latest}')
            df = pd.read_csv(latest, engine='c', low_memory=False)
            for c in base_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
            return df.dropna(subset=base_cols)

    print(f'[INFO] Loading raw CSVs from {root_dir}')
    records = []
    for fn in sorted(os.listdir(root_dir)):
        if not fn.endswith('.csv') or not re.match(r'^P\d+_S\d+_V\d+_', fn):
            continue
        fp = os.path.join(root_dir, fn)
        try:
            df_tmp = pd.read_csv(fp, sep=',', engine='c', low_memory=False)
        except Exception as exc:
            print(f'  [WARN] {fn}: {exc}')
            continue
        if any(c not in df_tmp.columns for c in base_cols + [label_col, video_col]):
            continue
        df_tmp[video_col] = (df_tmp[video_col].astype(str)
                             .str.replace(r'\.mp4$', '', regex=True).str.strip())
        df_tmp[label_col] = df_tmp[label_col].astype(str).str.strip()
        for c in base_cols:
            df_tmp[c] = pd.to_numeric(df_tmp[c], errors='coerce').astype('float32')
        records.append(df_tmp.dropna(subset=base_cols))
    if not records:
        raise SystemExit(f'No valid CSVs in {root_dir}')
    df = pd.concat(records, ignore_index=True)
    print(f'[INFO] DS2 {len(df):,} rows | participants: {sorted(df[label_col].unique())}')
    return df


def load(dataset, root_dir, base_cols, label_col, video_col, seq_len=10):
    if dataset == 'ds1':
        return load_ds1(root_dir, base_cols, label_col, video_col, seq_len)
    return load_ds2(root_dir, base_cols, label_col, video_col)
