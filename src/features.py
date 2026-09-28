"""Regime clustering, condition-specific normalisation, rolling features, 3D tensors.

Why this matters on FD002: the six flight conditions (altitude, Mach, throttle
resolver angle) shift every sensor by far more than degradation does. A global
scaler leaves those jumps in the signal, so the model spends its capacity on
"which condition am I in" instead of "how worn is the engine". Normalising
each sensor *within its operating regime* removes that offset and exposes the
slow mechanical-degradation trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import config as C


# ---------------------------------------------------------------------------
# 1. Operating-regime segmentation
# ---------------------------------------------------------------------------
class RegimeClusterer:
    """K-Means (k=6) on the three operational settings."""

    def __init__(self, k: int = C.N_REGIMES, seed: int = C.SEED):
        self.scaler = StandardScaler()
        self.km = KMeans(n_clusters=k, n_init=10, random_state=seed)

    def fit(self, df: pd.DataFrame):
        self.km.fit(self.scaler.fit_transform(df[C.SETTING_COLS]))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.km.predict(self.scaler.transform(df[C.SETTING_COLS]))


# ---------------------------------------------------------------------------
# 2. Normalisers
# ---------------------------------------------------------------------------
class GlobalNormalizer:
    """Baseline: one StandardScaler over all rows (ignores flight condition)."""

    def fit(self, df, sensors):
        self.sensors = sensors
        self.scaler = StandardScaler().fit(df[sensors])
        return self

    def transform(self, df):
        out = df.copy()
        out[self.sensors] = self.scaler.transform(df[self.sensors])
        return out


class RegimeNormalizer:
    """Condition-specific z-scoring: mean/std learned per regime on train only."""

    def fit(self, df, sensors):
        self.sensors = sensors
        g = df.groupby("regime")[sensors]
        self.mu = g.mean()
        self.sd = g.std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, df):
        out = df.copy()
        mu = self.mu.loc[df["regime"]].to_numpy()
        sd = self.sd.loc[df["regime"]].to_numpy()
        out[self.sensors] = (df[self.sensors].to_numpy() - mu) / sd
        return out


def informative_sensors(df: pd.DataFrame, by_regime: bool, tol: float = 1e-3) -> list[str]:
    """Drop sensors that carry no information.

    With regimes, a sensor that is constant *inside every regime* (e.g. s_1,
    s_5, s_18, s_19 on FD002) only encodes the flight condition, not wear.
    """
    if by_regime:
        spread = df.groupby("regime")[C.SENSOR_COLS].std().max()
    else:
        spread = df[C.SENSOR_COLS].std()
    return [s for s in C.SENSOR_COLS if spread[s] > tol]


# ---------------------------------------------------------------------------
# 3. Rolling features (computed per engine, causal — no look-ahead)
# ---------------------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame, sensors: list[str], w: int = C.ROLL_WINDOW):
    g = df.groupby("unit")[sensors]
    mean = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    std = g.rolling(w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    mean.columns = [f"{s}_rmean{w}" for s in sensors]
    std.columns = [f"{s}_rstd{w}" for s in sensors]
    return pd.concat([df, mean, std], axis=1), list(mean.columns) + list(std.columns)


# ---------------------------------------------------------------------------
# 4. 3D sequence tensors  (samples, WINDOW, features)
# ---------------------------------------------------------------------------
def _pad(arr: np.ndarray, w: int) -> np.ndarray:
    """Left-pad short engines by repeating the first row."""
    if len(arr) >= w:
        return arr
    return np.vstack([np.repeat(arr[:1], w - len(arr), axis=0), arr])


def make_train_sequences(df, feats, w=C.WINDOW):
    X, y, units = [], [], []
    for u, g in df.groupby("unit"):
        a, r = _pad(g[feats].to_numpy(np.float32), w), g["RUL"].to_numpy(np.float32)
        r = np.concatenate([np.repeat(r[:1], len(a) - len(r)), r])
        for end in range(w, len(a) + 1):
            X.append(a[end - w:end])
            y.append(r[end - 1])
            units.append(u)
    return np.stack(X), np.array(y), np.array(units)


def make_test_sequences(df, feats, w=C.WINDOW):
    """One window per engine: the last W cycles observed."""
    X = [_pad(g[feats].to_numpy(np.float32), w)[-w:] for _, g in df.groupby("unit")]
    return np.stack(X)


# ---------------------------------------------------------------------------
# 5. End-to-end preparation for one experiment
# ---------------------------------------------------------------------------
def prepare(train, test, *, regime_norm: bool, rolling: bool, regime_onehot: bool = True):
    """Return X_train, y_train, units, X_test, feature_names, fitted clusterer."""
    train, test = train.copy(), test.copy()

    clusterer = RegimeClusterer().fit(train)
    train["regime"] = clusterer.predict(train)
    test["regime"] = clusterer.predict(test)

    sensors = informative_sensors(train, by_regime=regime_norm)
    norm = (RegimeNormalizer() if regime_norm else GlobalNormalizer()).fit(train, sensors)
    train, test = norm.transform(train), norm.transform(test)

    feats = list(sensors)
    if rolling:
        train, roll = add_rolling_features(train, sensors)
        test, _ = add_rolling_features(test, sensors)
        feats += roll

    if regime_norm and regime_onehot:
        # Tell the model which condition each cycle was flown in.
        for k in range(C.N_REGIMES):
            train[f"regime_{k}"] = (train["regime"] == k).astype(np.float32)
            test[f"regime_{k}"] = (test["regime"] == k).astype(np.float32)
            feats.append(f"regime_{k}")

    X_tr, y_tr, units = make_train_sequences(train, feats)
    X_te = make_test_sequences(test, feats)
    return X_tr, y_tr, units, X_te, feats, clusterer
