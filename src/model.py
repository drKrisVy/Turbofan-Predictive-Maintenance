"""Bi-LSTM regressor and evaluation metrics."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

from . import config as C


def build_bilstm(window: int, n_features: int) -> tf.keras.Model:
    inp = layers.Input(shape=(window, n_features))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out, name="bilstm_rul")
    model.compile(optimizer=optimizers.Adam(C.LEARNING_RATE), loss="mse",
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def train_model(X, y, units, epochs=C.EPOCHS, verbose=2):
    """Train with an engine-level validation split (no window leakage)."""
    rng = np.random.default_rng(C.SEED)
    uniq = np.unique(units)
    val_units = rng.choice(uniq, int(len(uniq) * C.VAL_FRACTION), replace=False)
    val = np.isin(units, val_units)

    model = build_bilstm(X.shape[1], X.shape[2])
    cbs = [
        callbacks.EarlyStopping(monitor="val_rmse", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_rmse", factor=0.5, patience=4, min_lr=1e-5),
    ]
    hist = model.fit(X[~val], y[~val], validation_data=(X[val], y[val]),
                     epochs=epochs, batch_size=C.BATCH_SIZE, callbacks=cbs, verbose=verbose)
    return model, hist.history


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric score: late predictions are penalised harder than early ones."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
