import hashlib
import numpy as np

AR_WIN    = 128
AR_STRIDE = 8
L2        = 1e-3
DT        = 1 / 90.0
KF_Q_POS  = 0.02
KF_Q_VEL  = 0.10
KF_R_PS   = 0.09
ALPHA_DEFAULT = 0.5
SEED_DEFAULT  = 42

def session_seed(participant, video, salt, global_seed=SEED_DEFAULT):
    key = f'{participant}|{video}|{salt}|{global_seed}'
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def gen_ewma_noise(n, sigma, alpha=ALPHA_DEFAULT, rng=None):
    a    = alpha / 100.0 if alpha > 1 else float(alpha)
    b    = 1.0 - a
    e    = (rng.normal(0.0, sigma, n) if rng is not None
            else np.random.normal(0.0, sigma, n)).astype('float32')
    s    = np.empty_like(e)
    prev = np.float32(0.0)
    for i in range(n):
        prev = a * e[i] + b * prev
        s[i] = prev
    return s


def add_noise_df(df, base_cols, label_col, video_col, sigma,
                 salt=0, alpha=ALPHA_DEFAULT, global_seed=SEED_DEFAULT):
    df = df.reset_index(drop=True)
    X  = df[base_cols].values.astype('float32').copy()
    for (pid, vid), idx in df.groupby([label_col, video_col]).groups.items():
        idx = idx.values
        rng = np.random.default_rng(session_seed(str(pid), str(vid), salt, global_seed))
        for d in range(X.shape[1]):
            X[idx, d] += gen_ewma_noise(len(idx), sigma, alpha=alpha, rng=rng)
    return X


def _ridge(y, l2=L2):
    Y = y[2:]
    if Y.size < 2:
        return 1.0, 0.0, 0.0
    X  = np.stack([y[1:-1], y[:-2], np.ones_like(Y)], axis=1)
    I  = np.eye(3)
    I[-1, -1] = 0.0
    b  = np.linalg.solve(X.T @ X + l2 * I, X.T @ Y)
    return float(b[0]), float(b[1]), float(b[2])


def ar2(X_in, win=AR_WIN, stride=AR_STRIDE, l2=L2):
    T, F = X_in.shape
    out  = np.full((T, F), np.nan, dtype=np.float32)
    win  = max(16, min(win, T))
    for d in range(F):
        lb = None
        lp = np.nan
        for t in range(2, T):
            if lb is None or t % stride == 0:
                start = max(0, t - win)
                w     = X_in[start:t, d].astype(np.float64)
                if t - start < 4 or np.std(w) < 1e-8:
                    lb = None
                    lp = float(X_in[t - 1, d])
                else:
                    mu = float(np.mean(w))
                    sd = float(np.std(w) + 1e-8)
                    wz = (w - mu) / sd
                    a1, a2, b = _ridge(wz, l2)
                    val = a1 * wz[-1] + a2 * wz[-2] + b
                    lb  = (a1, a2, b, mu, sd)
                    lp  = float(val * sd + mu)
            elif lb is None:
                continue
            out[t, d] = np.float32(lp)
    return out


def ar2_predict(X_in, seq_len, horizon, win=AR_WIN, stride=AR_STRIDE, l2=L2):
    T, F   = X_in.shape
    pseudo = np.full((T, F), np.nan, dtype=np.float32)
    win    = max(16, min(win, T))
    for d in range(F):
        lb = lws = None
        rp1 = rp2 = np.nan
        for t in range(max(seq_len, 2), T):
            if lb is None or t % stride == 0:
                start = max(0, t - win)
                w     = X_in[start:t, d].astype(np.float64)
                if t - start < 4 or np.std(w) < 1e-8:
                    lb = None
                    pseudo[t, d] = np.float32(X_in[t - 1, d])
                    continue
                mu = float(np.mean(w))
                sd = float(np.std(w) + 1e-8)
                wz = (w - mu) / sd
                a1, a2, b = _ridge(wz, l2)
                v1, v2    = wz[-1], wz[-2]
                val       = v1
                for _ in range(horizon):
                    val = a1 * v1 + a2 * v2 + b
                    v2, v1 = v1, val
                lb  = (a1, a2, b)
                lws = (mu, sd)
                rp1, rp2     = val, wz[-1]
                pseudo[t, d] = np.float32(val * sd + mu)
            else:
                if lb is None:
                    continue
                a1, a2, b    = lb
                mu, sd       = lws
                val          = a1 * rp1 + a2 * rp2 + b
                rp2, rp1     = rp1, val
                pseudo[t, d] = np.float32(val * sd + mu)
    return pseudo


class Kalman1D:
    __slots__ = ('x0', 'x1', 'P00', 'P01', 'P10', 'P11',
                 'dt', 'q11', 'q12', 'q22', 'r')

    def __init__(self, x0=0.0, dt=DT, q_pos=KF_Q_POS, q_vel=KF_Q_VEL, r=1.0):
        self.x0  = float(x0)
        self.x1  = 0.0
        self.dt  = float(dt)
        self.r   = float(r)
        self.P00 = self.P11 = 1e2
        self.P01 = self.P10 = 0.0
        self.q11 = (dt ** 4) / 4 * q_pos + 1e-9
        self.q12 = (dt ** 3) / 2 * q_pos
        self.q22 = dt ** 2 * q_pos + q_vel * 1e-3

    def predict(self):
        self.x0 += self.dt * self.x1
        P00, P01, P10, P11 = self.P00, self.P01, self.P10, self.P11
        dt = self.dt
        self.P00 = P00 + dt * (P10 + P01) + dt * dt * P11 + self.q11
        self.P01 = P01 + dt * P11 + self.q12
        self.P10 = P10 + dt * P11 + self.q12
        self.P11 = P11 + self.q22

    def update(self, z, r=None):
        R = self.r if r is None else float(r)
        y = float(z) - self.x0
        S = self.P00 + R
        if S <= 0:
            return
        K0 = self.P00 / S
        K1 = self.P10 / S
        self.x0 += K0 * y
        self.x1 += K1 * y
        P00 = self.P00
        self.P00 -= K0 * P00
        self.P01 -= K0 * self.P01
        self.P10 -= K1 * P00
        self.P11 -= K1 * self.P01


class KalmanND:
    def __init__(self, x0_vec, dt=DT, q_pos=KF_Q_POS, q_vel=KF_Q_VEL, r=1.0):
        self.filters = [Kalman1D(float(x), dt=dt, q_pos=q_pos, q_vel=q_vel, r=r)
                        for x in x0_vec]

    def predict(self):
        for f in self.filters:
            f.predict()

    def update(self, z_vec, r=None):
        for f, z in zip(self.filters, z_vec):
            f.update(float(z), r=r)

    def state(self):
        return np.array([f.x0 for f in self.filters], dtype=np.float32)


def kalman_nd(noisy_X, pseudo_X, r_meas, r_pseudo=KF_R_PS,
              dt=DT, q_pos=KF_Q_POS, q_vel=KF_Q_VEL):
    T, D = noisy_X.shape
    out  = np.empty_like(noisy_X, dtype=np.float32)
    kf   = KalmanND(noisy_X[0], dt=dt, q_pos=q_pos, q_vel=q_vel, r=r_meas)
    for t in range(T):
        kf.predict()
        kf.update(noisy_X[t], r=r_meas)
        p = pseudo_X[t]
        if np.all(np.isfinite(p)):
            kf.update(p, r=r_pseudo)
        out[t] = kf.state()
    return out


def defend(Xn, sigma, r_pseudo=KF_R_PS, dt=DT, q_pos=KF_Q_POS, q_vel=KF_Q_VEL):
    r_meas = float(sigma) ** 2
    return kalman_nd(Xn, ar2(Xn), r_meas=r_meas, r_pseudo=r_pseudo,
                     dt=dt, q_pos=q_pos, q_vel=q_vel)

