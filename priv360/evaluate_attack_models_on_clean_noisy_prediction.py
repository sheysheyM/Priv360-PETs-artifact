from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import time
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from priv360.apply_noise_and_prediction_defenses import add_noise_df, defend
from priv360.dataset_settings_and_loader import (
    CONFIGS, SEQ_LEN, SEED, SIGMA, BATCH, STEPS_PER_EPOCH,
    PATIENCE, LR_PATIENCE, PARTIAL_CLEAN_FRACS,
    RF_TREES, RF_DEPTH, RF_JOBS, SIGMAS_NOISY, load
)

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

try:
    from cuml.ensemble import RandomForestClassifier as _RF
    RF_BACKEND = 'cuML'
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as _RF
    RF_BACKEND = 'sklearn-CPU'


def parse_args():
    p = argparse.ArgumentParser(description='Priv360 attack-model evaluation runner')
    p.add_argument('--dataset', choices=['ds1', 'ds2'], required=True)
    p.add_argument('--model',
                   choices=['lstm', 'transformer', 'rf',
                            'clean_lstm', 'clean_transformer', 'clean_rf',
                            'defense_lstm', 'defense_transformer', 'defense_rf',
                            'noisy_lstm', 'noisy_transformer', 'noisy_rf'],
                   required=True)
    p.add_argument('--data-root', default=None)
    p.add_argument('--output',    default=None)
    return p.parse_args()


# ── shared helpers ────────────────────────────────────────────────────

def make_partial_clean(X_clean, X_noisy, p_clean):
    if p_clean <= 0.0:
        return X_noisy.copy()
    if p_clean >= 1.0:
        return X_clean.copy()
    mask      = np.random.default_rng(SEED).random(len(X_clean)) < p_clean
    out       = X_noisy.copy()
    out[mask] = X_clean[mask]
    return out


def make_windows(x, y, seq_len):
    nf    = x.shape[1]
    n_win = x.shape[0] - seq_len + 1
    X_win = np.lib.stride_tricks.sliding_window_view(x, (seq_len, nf))
    return X_win.reshape(n_win, seq_len * nf).astype('float32'), y[seq_len - 1:]


def oversample_balanced(X, Y):
    counts  = np.bincount(Y)
    max_cnt = counts.max()
    rng     = np.random.default_rng(SEED)
    idx_bal = []
    for cls_id, cnt in enumerate(counts):
        idx_cls = np.where(Y == cls_id)[0]
        extra   = (rng.choice(idx_cls, size=max_cnt - cnt, replace=True)
                   if cnt < max_cnt else np.array([], dtype=int))
        idx_bal.append(np.concatenate([idx_cls, extra]))
    return (X[rng.permutation(np.concatenate(idx_bal))],
            Y[rng.permutation(np.concatenate(idx_bal))])


def get_class_weights(y):
    classes = np.unique(y)
    counts  = np.bincount(y)
    total   = float(len(y))
    return {c: total / (len(classes) * counts[c]) for c in classes}


def make_recording_id(df, label_col, video_col):
    return (df[label_col].astype(str) + '_' + df[video_col].astype(str)).values


def make_ts_ds(x, y, seq_len, batch, shuffle):
    tgt = y[seq_len - 1:]
    return (tf.keras.utils.timeseries_dataset_from_array(
                data=x, targets=tgt, sequence_length=seq_len,
                sequence_stride=1, shuffle=shuffle, batch_size=batch)
            .prefetch(tf.data.AUTOTUNE))


def make_recording_windows(x, y, recording_ids, seq_len):
    Xs, Ys = [], []
    for rec_id in np.unique(recording_ids):
        mask = recording_ids == rec_id
        xr, yr = x[mask], y[mask]
        if len(xr) < seq_len:
            continue
        for i in range(len(xr) - seq_len + 1):
            Xs.append(xr[i:i+seq_len])
            Ys.append(yr[i + seq_len - 1])
    return np.array(Xs, dtype='float32'), np.array(Ys, dtype='int32')


# ── model builders ───────────────────────────────────────────────────

def build_lstm(n_cls, n_feat, seq_len):
    inp = tf.keras.Input(shape=(seq_len, n_feat))
    x   = LSTM(128, return_sequences=True)(inp)
    x   = Dropout(0.3)(x)
    x   = LSTM(64)(x)
    x   = Dropout(0.3)(x)
    x   = Dense(64, activation='relu')(x)
    out = Dense(n_cls, activation='softmax', dtype='float32')(x)
    m   = tf.keras.Model(inp, out)
    m.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
              metrics=['accuracy'])
    return m


def build_transformer(n_cls, n_feat, seq_len):
    d   = 64
    inp = tf.keras.Input(shape=(seq_len, n_feat))
    x   = tf.keras.layers.Dense(d, dtype='float32')(inp)
    pos = tf.keras.layers.Embedding(seq_len, d, dtype='float32')(tf.range(seq_len))
    pos = tf.keras.layers.Lambda(
        lambda inputs: tf.tile(tf.expand_dims(inputs[0], 0), [tf.shape(inputs[1])[0], 1, 1]))([pos, x])
    x   = tf.keras.layers.Add()([x, pos])
    for _ in range(2):
        a = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d//4, dtype='float32')(x, x)
        x = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([x, a]))
        f = tf.keras.layers.Dense(128, activation='relu', dtype='float32')(x)
        f = tf.keras.layers.Dense(d, dtype='float32')(f)
        x = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([x, f]))
    x   = tf.keras.layers.GlobalAveragePooling1D(dtype='float32')(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = tf.keras.layers.Dense(64, activation='relu', dtype='float32')(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_cls, activation='softmax', dtype='float32')(x)
    m   = tf.keras.Model(inp, out)
    m.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
              metrics=['accuracy'])
    return m


# ── callbacks ─────────────────────────────────────────────────────────

class LivePrint(Callback):
    def __init__(self, label=''):
        super().__init__()
        self.label = label
        self.t0    = time.time()
        self.best  = 0.0

    def on_train_begin(self, logs=None):
        print(f"\n  {'Ep':>4} | {'TrainAcc':>9} {'Loss':>8} | {'ValAcc':>8} {'ValLoss':>8} | {'Time':>5}")
        print(f"  {'─'*55}")

    def on_epoch_end(self, epoch, logs=None):
        logs   = logs or {}
        va_acc = logs.get('val_accuracy', 0) * 100
        mark   = ' ✅' if va_acc > self.best else ''
        if va_acc > self.best:
            self.best = va_acc
        print(f"  {epoch+1:>4} | {logs.get('accuracy',0)*100:>8.2f}% {logs.get('loss',0):>8.4f} | "
              f"{va_acc:>7.2f}% {logs.get('val_loss',0):>8.4f} | "
              f"{time.time()-self.t0:>4.0f}s{mark}", flush=True)

    def on_train_end(self, logs=None):
        print(f"  {'─'*55}\n  [{self.label}] Best val_acc={self.best:.2f}%")


# ── train helpers ─────────────────────────────────────────────────────

def train_eval_nn(x_tr, y_tr, x_te, y_te, n_cls, label, model_fn, epochs, seq_len):
    sc    = StandardScaler()
    x_tr  = sc.fit_transform(x_tr).astype('float32')
    x_te  = sc.transform(x_te).astype('float32')
    tr_ds = make_ts_ds(x_tr, y_tr, seq_len, BATCH, True)
    te_ds = make_ts_ds(x_te, y_te, seq_len, BATCH, False)
    model = model_fn(n_cls, x_tr.shape[1], seq_len)
    model.fit(tr_ds, validation_data=te_ds, epochs=epochs,
              steps_per_epoch=STEPS_PER_EPOCH,
              callbacks=[
                  EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                restore_best_weights=True, verbose=0),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=LR_PATIENCE, min_lr=1e-6, verbose=0),
                  LivePrint(label),
              ], verbose=0)
    y_true, y_pred = [], []
    for bx, by in te_ds:
        y_pred.extend(np.argmax(model.predict(bx, verbose=0), axis=1))
        y_true.extend(by.numpy())
    acc = float(accuracy_score(y_true, y_pred))
    tf.keras.backend.clear_session()
    return acc


def train_eval_nn_recording(x_tr, y_tr, x_te, y_te, rec_tr, rec_te,
                             n_cls, model_fn, epochs, seq_len):
    sc    = StandardScaler()
    x_tr  = sc.fit_transform(x_tr).astype('float32')
    x_te  = sc.transform(x_te).astype('float32')
    X_tr_w, Y_tr_w = make_recording_windows(x_tr, y_tr, rec_tr, seq_len)
    X_te_w, Y_te_w = make_recording_windows(x_te, y_te, rec_te, seq_len)
    cw    = compute_class_weight('balanced', classes=np.unique(Y_tr_w), y=Y_tr_w)
    model = model_fn(n_cls, x_tr.shape[1], seq_len)
    tr_ds = (tf.data.Dataset.from_tensor_slices((X_tr_w, Y_tr_w))
             .shuffle(20_000).batch(BATCH).prefetch(tf.data.AUTOTUNE))
    te_ds = (tf.data.Dataset.from_tensor_slices((X_te_w, Y_te_w))
             .batch(BATCH).prefetch(tf.data.AUTOTUNE))
    model.fit(tr_ds, validation_data=te_ds, epochs=epochs,
              steps_per_epoch=STEPS_PER_EPOCH,
              class_weight=dict(enumerate(cw)),
              callbacks=[
                  EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                restore_best_weights=True, verbose=0),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=LR_PATIENCE, min_lr=1e-6, verbose=0),
                  LivePrint(),
              ], verbose=0)
    y_true, y_pred = [], []
    for bx, by in te_ds:
        y_pred.extend(np.argmax(model.predict(bx, verbose=0), axis=1))
        y_true.extend(by.numpy())
    acc = float(accuracy_score(y_true, y_pred))
    tf.keras.backend.clear_session()
    return acc


def train_eval_rf_partial(x_tr, y_tr_e, x_te, y_te_e, label):
    t0   = time.time()
    sc   = StandardScaler()
    x_tr = sc.fit_transform(x_tr).astype('float32')
    x_te = sc.transform(x_te).astype('float32')
    X_tr_w, Y_tr = make_windows(x_tr, y_tr_e, SEQ_LEN)
    X_te_w, Y_te = make_windows(x_te, y_te_e, SEQ_LEN)
    if RF_BACKEND == 'cuML':
        X_tr_w, Y_tr = oversample_balanced(X_tr_w, Y_tr)
        clf = _RF(n_estimators=RF_TREES, max_depth=RF_DEPTH,
                  n_streams=4, random_state=SEED)
        clf.fit(X_tr_w, Y_tr)
        y_pred = np.asarray(clf.predict(X_te_w))
    else:
        weights = get_class_weights(Y_tr)
        clf     = _RF(n_estimators=RF_TREES, max_depth=None,
                      n_jobs=RF_JOBS, class_weight=weights, random_state=SEED)
        clf.fit(X_tr_w, Y_tr)
        y_pred  = clf.predict(X_te_w)
    acc = float(accuracy_score(np.asarray(Y_te), y_pred))
    print(f'  [{label}] acc={acc*100:.2f}%  t={time.time()-t0:.1f}s  backend={RF_BACKEND}')
    return acc


def train_eval_rf_clean(x_tr, y_tr, x_te, y_te, seq_len):
    sc    = StandardScaler()
    x_tr  = sc.fit_transform(x_tr).astype('float32')
    x_te  = sc.transform(x_te).astype('float32')
    nf    = x_tr.shape[1]
    X_tr_w = np.lib.stride_tricks.sliding_window_view(
        x_tr, (seq_len, nf)).reshape(x_tr.shape[0]-seq_len+1, seq_len*nf)
    Y_tr_w = y_tr[seq_len - 1:]
    X_te_w = np.lib.stride_tricks.sliding_window_view(
        x_te, (seq_len, nf)).reshape(x_te.shape[0]-seq_len+1, seq_len*nf)
    Y_te_w = y_te[seq_len - 1:]
    cw     = compute_class_weight('balanced', classes=np.unique(Y_tr_w), y=Y_tr_w)
    if RF_BACKEND == 'cuML':
        clf = _RF(n_estimators=RF_TREES, max_depth=20, n_streams=4, random_state=SEED)
    else:
        clf = _RF(n_estimators=RF_TREES, max_depth=None, n_jobs=RF_JOBS,
                  class_weight=dict(enumerate(cw)), random_state=SEED)
    clf.fit(X_tr_w.astype('float32'), Y_tr_w)
    y_pred = np.asarray(clf.predict(X_te_w.astype('float32')))
    return float(accuracy_score(Y_te_w, y_pred))


# ── attack runners ────────────────────────────────────────────────────

def run_partial_clean_nn(data, cfg, model_fn, model_name, output):
    mixed_precision.set_global_policy('mixed_float16')
    tf.config.optimizer.set_jit(True)
    for _g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(_g, True)
        except Exception:
            pass
    if not tf.config.list_physical_devices('GPU'):
        raise SystemError('No GPU found')

    base_cols = cfg['base_cols']
    label_col = cfg['label_col']
    video_col = cfg['video_col']
    epochs    = cfg['epochs']
    all_videos = sorted(data[video_col].unique())
    results    = []
    t_global   = time.time()

    for fold_i, vid in enumerate(all_videos):
        t0 = time.time()
        print(f'\n{"█"*65}\nFold {fold_i+1}/{len(all_videos)} | test={vid}')

        df_te = data[data[video_col] == vid].reset_index(drop=True)
        df_tr = data[data[video_col] != vid].reset_index(drop=True)
        le    = LabelEncoder()
        le.fit(np.concatenate([df_tr[label_col].values, df_te[label_col].values]))
        y_tr_e = le.transform(df_tr[label_col].values).astype('int32')
        y_te_e = le.transform(df_te[label_col].values).astype('int32')
        n_cls  = len(le.classes_)

        X_clean_tr = df_tr[base_cols].values.astype('float32')
        X_clean_te = df_te[base_cols].values.astype('float32')
        X_noisy_tr = add_noise_df(df_tr, base_cols, label_col, video_col, SIGMA, salt=SEED)
        X_noisy_te = add_noise_df(df_te, base_cols, label_col, video_col, SIGMA, salt=SEED)

        row = dict(fold=fold_i+1, test_video=vid, n_cls=n_cls, n_train=len(df_tr), n_test=len(df_te))

        for p_clean in PARTIAL_CLEAN_FRACS:
            pct  = int(p_clean * 100)
            X_tr = make_partial_clean(X_clean_tr, X_noisy_tr, p_clean)
            X_te = make_partial_clean(X_clean_te, X_noisy_te, p_clean)
            X_tr_pred = defend(X_tr, SIGMA)
            X_te_pred = defend(X_te, SIGMA)
            for cond, xtr, xte in [('noisy', X_tr, X_te),
                                   ('pred',  X_tr_pred, X_te_pred)]:
                key      = f'orientation_p{pct}_{cond}_{model_name}'
                row[key] = train_eval_nn(xtr, y_tr_e, xte, y_te_e, n_cls,
                                         f'{model_name} p={p_clean} {cond}',
                                         model_fn, epochs, SEQ_LEN)
                print(f'  {key}: {row[key]*100:.1f}%')

        row['time_min'] = round((time.time() - t0) / 60, 1)
        results.append(row)
        pd.DataFrame(results).to_csv(output, index=False)

    print(f'\nTotal: {(time.time()-t_global)/60:.1f} min  →  {output}')


def run_partial_clean_rf(data, cfg, output):
    print(f'RF backend: {RF_BACKEND}')

    base_cols  = cfg['base_cols']
    label_col  = cfg['label_col']
    video_col  = cfg['video_col']
    all_videos = sorted(data[video_col].unique())
    results    = []

    for fold_i, vid in enumerate(all_videos):
        t0 = time.time()
        print(f'\nFold {fold_i+1}/{len(all_videos)} | test={vid}')

        df_te  = data[data[video_col] == vid].reset_index(drop=True)
        df_tr  = data[data[video_col] != vid].reset_index(drop=True)
        le     = LabelEncoder()
        le.fit(np.concatenate([df_tr[label_col].values, df_te[label_col].values]))
        y_tr_e = le.transform(df_tr[label_col].values).astype('int32')
        y_te_e = le.transform(df_te[label_col].values).astype('int32')

        X_clean_tr = df_tr[base_cols].values.astype('float32')
        X_clean_te = df_te[base_cols].values.astype('float32')
        X_noisy_tr = add_noise_df(df_tr, base_cols, label_col, video_col, SIGMA, salt=SEED)
        X_noisy_te = add_noise_df(df_te, base_cols, label_col, video_col, SIGMA, salt=SEED)

        row = dict(fold=fold_i+1, test_video=vid, n_cls=len(le.classes_),
                   n_train=len(df_tr), n_test=len(df_te))

        for p_clean in PARTIAL_CLEAN_FRACS:
            pct  = int(p_clean * 100)
            X_tr = make_partial_clean(X_clean_tr, X_noisy_tr, p_clean)
            X_te = make_partial_clean(X_clean_te, X_noisy_te, p_clean)
            for cond, xtr, xte in [('noisy', X_tr, X_te),
                                   ('pred',  defend(X_tr, SIGMA), defend(X_te, SIGMA))]:
                key      = f'orientation_p{pct}_{cond}_rf'
                row[key] = train_eval_rf_partial(xtr, y_tr_e, xte, y_te_e,
                                                 f'p={p_clean} {cond}')

        row['time_min'] = round((time.time() - t0) / 60, 1)
        results.append(row)
        pd.DataFrame(results).to_csv(output, index=False)

    print(f'\nSaved → {output}')


def run_noisy_sigma_sweep_nn(data, cfg, model_fn, model_name, output):
    for _g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(_g, True)
        except Exception:
            pass
    if not tf.config.list_physical_devices('GPU'):
        raise SystemError('No GPU found')

    base_cols  = cfg['base_cols']
    label_col  = cfg['label_col']
    video_col  = cfg['video_col']
    epochs     = cfg['epochs']
    all_videos = sorted(data[video_col].unique())
    results    = []
    t_global   = time.time()

    print(f'\n{"="*65}\nLOVO — {len(all_videos)} folds | {model_name} | sigma sweep {SIGMAS_NOISY}')
    print(f'Train: noisy(salt=0)  Test: noisy(salt=1) + AR2+Kalman\n{"="*65}')

    for fold_i, vid in enumerate(all_videos):
        t0 = time.time()
        print(f'\n{"█"*65}\nFold {fold_i+1}/{len(all_videos)} | test={vid}')

        df_te  = data[data[video_col] == vid].reset_index(drop=True)
        df_tr  = data[data[video_col] != vid].reset_index(drop=True)
        le     = LabelEncoder()
        le.fit(np.concatenate([df_tr[label_col].values, df_te[label_col].values]))
        y_tr_e = le.transform(df_tr[label_col].values).astype('int32')
        y_te_e = le.transform(df_te[label_col].values).astype('int32')
        n_cls  = len(le.classes_)

        row = dict(fold=fold_i+1, test_video=vid, n_cls=n_cls, n_train=len(df_tr), n_test=len(df_te))

        for sigma in SIGMAS_NOISY:
            print(f'\n  ── σ={sigma} ──')
            X_tr_n = add_noise_df(df_tr, base_cols, label_col, video_col, sigma, salt=0)
            X_te_n = add_noise_df(df_te, base_cols, label_col, video_col, sigma, salt=1)
            X_tr_p = defend(X_tr_n, sigma)
            X_te_p = defend(X_te_n, sigma)

            acc_n = train_eval_nn(X_tr_n, y_tr_e, X_te_n, y_te_e, n_cls,
                                  f'noisy σ={sigma}', model_fn, epochs, SEQ_LEN)
            acc_p = train_eval_nn(X_tr_p, y_tr_e, X_te_p, y_te_e, n_cls,
                                  f'pred  σ={sigma}', model_fn, epochs, SEQ_LEN)

            row[f'acc_noisy_s{sigma}'] = acc_n
            row[f'acc_pred_s{sigma}']  = acc_p
            print(f'  noisy={acc_n*100:.1f}%  pred={acc_p*100:.1f}%  drop={(acc_n-acc_p)*100:.1f}%')

        row['time_min'] = round((time.time() - t0) / 60, 1)
        results.append(row)
        pd.DataFrame(results).to_csv(output, index=False)

    df_res = pd.DataFrame(results)
    print(f'\n{"═"*65}\nFINAL — Noisy {model_name}')
    for sigma in SIGMAS_NOISY:
        nc = df_res.get(f'acc_noisy_s{sigma}', pd.Series([0])).mean() * 100
        pc = df_res.get(f'acc_pred_s{sigma}',  pd.Series([0])).mean() * 100
        print(f'  σ={sigma:<5} noisy={nc:.1f}%  pred={pc:.1f}%  drop={nc-pc:.1f}%')
    print(f'Total: {(time.time()-t_global)/60:.1f} min  →  {output}')


def run_noisy_sigma_sweep_rf(data, cfg, output):
    print(f'RF backend: {RF_BACKEND}')

    base_cols  = cfg['base_cols']
    label_col  = cfg['label_col']
    video_col  = cfg['video_col']
    all_videos = sorted(data[video_col].unique())
    results    = []
    t_global   = time.time()

    print(f'\n{"="*65}\nLOVO RF — {len(all_videos)} folds | sigma sweep {SIGMAS_NOISY}')
    print(f'Train: noisy(salt=0)  Test: noisy(salt=1) + AR2+Kalman\n{"="*65}')

    for fold_i, vid in enumerate(all_videos):
        t0 = time.time()
        print(f'\nFold {fold_i+1}/{len(all_videos)} | test={vid}')

        df_te  = data[data[video_col] == vid].reset_index(drop=True)
        df_tr  = data[data[video_col] != vid].reset_index(drop=True)
        le     = LabelEncoder()
        le.fit(np.concatenate([df_tr[label_col].values, df_te[label_col].values]))
        y_tr_e = le.transform(df_tr[label_col].values).astype('int32')
        y_te_e = le.transform(df_te[label_col].values).astype('int32')
        n_cls  = len(le.classes_)

        row = dict(fold=fold_i+1, test_video=vid, n_cls=n_cls, n_train=len(df_tr), n_test=len(df_te))

        for sigma in SIGMAS_NOISY:
            print(f'\n  ── σ={sigma} ──')
            X_tr_n = add_noise_df(df_tr, base_cols, label_col, video_col, sigma, salt=0)
            X_te_n = add_noise_df(df_te, base_cols, label_col, video_col, sigma, salt=1)
            X_tr_p = defend(X_tr_n, sigma)
            X_te_p = defend(X_te_n, sigma)

            acc_n = train_eval_rf_partial(X_tr_n, y_tr_e, X_te_n, y_te_e, f'noisy σ={sigma}')
            acc_p = train_eval_rf_partial(X_tr_p, y_tr_e, X_te_p, y_te_e, f'pred  σ={sigma}')
            row[f'acc_noisy_s{sigma}'] = acc_n
            row[f'acc_pred_s{sigma}']  = acc_p
            print(f'  noisy={acc_n*100:.2f}%  pred={acc_p*100:.2f}%  drop={(acc_n-acc_p)*100:.2f}%')

        row['time_min'] = round((time.time() - t0) / 60, 1)
        results.append(row)
        pd.DataFrame(results).to_csv(output, index=False)

        df_sf = pd.DataFrame(results)
        print(f'  Running mean after {fold_i+1} fold(s):')
        for sigma in SIGMAS_NOISY:
            nc = df_sf.get(f'acc_noisy_s{sigma}', pd.Series([0])).mean() * 100
            pc = df_sf.get(f'acc_pred_s{sigma}',  pd.Series([0])).mean() * 100
            print(f'    σ={sigma}  noisy={nc:.2f}%  pred={pc:.2f}%')

    df_res = pd.DataFrame(results)
    print(f'\n{"="*65}\nFINAL — Noisy RF  backend={RF_BACKEND}')
    for sigma in SIGMAS_NOISY:
        nc = df_res.get(f'acc_noisy_s{sigma}', pd.Series([0])).mean() * 100
        pc = df_res.get(f'acc_pred_s{sigma}',  pd.Series([0])).mean() * 100
        print(f'  σ={sigma:<5} noisy={nc:.2f}%  pred={pc:.2f}%  drop={nc-pc:.2f}%')
    print(f'Total: {(time.time()-t_global)/60:.1f} min  →  {output}')


def run_clean_baseline(data, cfg, model_name, output):
    for _g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(_g, True)
        except Exception:
            pass

    base_cols  = cfg['base_cols']
    label_col  = cfg['label_col']
    video_col  = cfg['video_col']
    epochs     = cfg['epochs']
    all_videos = sorted(data[video_col].unique())
    results    = []
    t_global   = time.time()

    for fold_i, vid in enumerate(all_videos):
        t0 = time.time()
        print(f'\nFold {fold_i+1}/{len(all_videos)} | test={vid} | model={model_name}')

        df_te  = data[data[video_col] == vid].reset_index(drop=True)
        df_tr  = data[data[video_col] != vid].reset_index(drop=True)
        le     = LabelEncoder()
        le.fit(np.concatenate([df_tr[label_col].values, df_te[label_col].values]))
        y_tr_e = le.transform(df_tr[label_col].values).astype('int32')
        y_te_e = le.transform(df_te[label_col].values).astype('int32')
        n_cls  = len(le.classes_)
        x_tr   = df_tr[base_cols].values.astype('float32')
        x_te   = df_te[base_cols].values.astype('float32')
        rec_tr = make_recording_id(df_tr, label_col, video_col)
        rec_te = make_recording_id(df_te, label_col, video_col)

        if model_name == 'clean_lstm':
            acc = train_eval_nn_recording(x_tr, y_tr_e, x_te, y_te_e, rec_tr, rec_te,
                                          n_cls, build_lstm, epochs, SEQ_LEN)
        elif model_name == 'clean_transformer':
            acc = train_eval_nn_recording(x_tr, y_tr_e, x_te, y_te_e, rec_tr, rec_te,
                                          n_cls, build_transformer, epochs, SEQ_LEN)
        else:
            acc = train_eval_rf_clean(x_tr, y_tr_e, x_te, y_te_e, SEQ_LEN)

        row = dict(fold=fold_i+1, test_video=vid, n_cls=n_cls, n_train=len(df_tr), n_test=len(df_te),
                   acc_clean=acc, time_min=round((time.time()-t0)/60, 1))
        results.append(row)
        print(f'  acc_clean={acc*100:.2f}%')
        pd.DataFrame(results).to_csv(output, index=False)

    mean_acc = pd.DataFrame(results)['acc_clean'].mean()
    print(f'\nMean clean acc ({model_name}): {mean_acc*100:.2f}%')
    print(f'Total: {(time.time()-t_global)/60:.1f} min  →  {output}')


# ── main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = CONFIGS[args.dataset]
    root = args.data_root or cfg['default_root']
    output = args.output or f'{args.dataset}_{args.model}_results.csv'

    data = load(args.dataset, root, cfg['base_cols'],
                cfg['label_col'], cfg['video_col'], SEQ_LEN)

    if args.model == 'lstm':
        run_partial_clean_nn(data, cfg, build_lstm, 'lstm', output)
    elif args.model == 'transformer':
        run_partial_clean_nn(data, cfg, build_transformer, 'transformer', output)
    elif args.model == 'rf':
        run_partial_clean_rf(data, cfg, output)
    elif args.model in ('defense_lstm', 'noisy_lstm'):
        run_noisy_sigma_sweep_nn(data, cfg, build_lstm, 'lstm', output)
    elif args.model in ('defense_transformer', 'noisy_transformer'):
        run_noisy_sigma_sweep_nn(data, cfg, build_transformer, 'transformer', output)
    elif args.model in ('defense_rf', 'noisy_rf'):
        run_noisy_sigma_sweep_rf(data, cfg, output)
    elif args.model in ('clean_lstm', 'clean_transformer', 'clean_rf'):
        run_clean_baseline(data, cfg, args.model, output)


if __name__ == '__main__':
    main()
