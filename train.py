"""Run the full FD002 study: diagnosis -> regime clustering -> baseline vs. Bi-LSTM.

    python train.py                  # all three experiments
    python train.py --epochs 5       # quick smoke test
    python train.py --only final     # just the full pipeline
"""
from __future__ import annotations

import argparse
import json
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from src import config as C
from src.data import load_fd002
from src import features as F
from src import diagnostics as D
from src.model import train_model, rmse, nasa_score

EXPERIMENTS = {
    # name: (regime_norm, rolling)
    "baseline":   (False, False),   # global scaler, raw sensors  -> the failure case
    "regime":     (True,  False),   # + K-Means regimes & per-regime normalisation
    "final":      (True,  True),    # + 30-cycle rolling features
}
LABELS = {
    "baseline": "Baseline Bi-LSTM (global norm)",
    "regime": "+ Regime normalisation (K-Means k=6)",
    "final": "+ 30-cycle rolling features",
}


def set_seeds(seed=C.SEED):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def diagnose(train: pd.DataFrame):
    """Show that flight condition, not wear, dominates sensor variance."""
    tr = train.copy()
    cl = F.RegimeClusterer().fit(tr)
    tr["regime"] = cl.predict(tr)

    print("\n[diagnosis] regime sizes:", tr["regime"].value_counts().sort_index().tolist())
    sensors = F.informative_sensors(tr, by_regime=False)
    share = D.regime_variance_share(tr, sensors)
    print("[diagnosis] share of sensor variance explained by flight regime alone:")
    print(share.round(3).to_string())

    reg_sensors = F.informative_sensors(tr, by_regime=True)
    glob = F.GlobalNormalizer().fit(tr, sensors).transform(tr)
    reg = F.RegimeNormalizer().fit(tr, reg_sensors).transform(tr)
    corr_g = glob[reg_sensors].corrwith(tr["RUL"]).abs()
    corr_r = reg[reg_sensors].corrwith(tr["RUL"]).abs()
    corr = pd.DataFrame({"|corr| global": corr_g, "|corr| regime": corr_r}).round(3)
    print("[diagnosis] |correlation with RUL| before vs after regime normalisation:")
    print(corr.sort_values("|corr| regime", ascending=False).to_string())

    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    D.plot_regimes(tr, C.FIG_DIR / "regimes.png")
    D.plot_normalisation_effect(tr, glob, reg, sensor="s_11", unit=1,
                                path=C.FIG_DIR / "normalisation_effect.png")
    return {"regime_variance_share": share.round(4).to_dict(),
            "rul_correlation": corr.to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--only", choices=list(EXPERIMENTS), nargs="*")
    args = ap.parse_args()

    train, test, true_rul = load_fd002()
    print(f"[data] train {train.shape}, test {test.shape}, "
          f"{train.unit.nunique()} train / {test.unit.nunique()} test engines")

    report = {"diagnosis": diagnose(train), "experiments": {}}
    y_true = np.minimum(true_rul, C.RUL_CAP)   # standard: evaluate against capped RUL
    preds = {}
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.only or EXPERIMENTS:
        regime_norm, rolling = EXPERIMENTS[name]
        set_seeds()
        X, y, units, X_te, feats, _ = F.prepare(train, test, regime_norm=regime_norm, rolling=rolling)
        print(f"\n=== {name}: X_train {X.shape}, X_test {X_te.shape} ({len(feats)} features) ===")
        model, hist = train_model(X, y, units, epochs=args.epochs)
        p = np.clip(model.predict(X_te, verbose=0).ravel(), 0, None)
        preds[LABELS[name]] = p
        r = {"test_rmse": rmse(y_true, p), "nasa_score": nasa_score(y_true, p),
             "best_val_rmse": float(min(hist["val_rmse"])), "epochs_run": len(hist["loss"]),
             "n_features": len(feats)}
        report["experiments"][name] = r
        model.save(C.MODEL_DIR / f"bilstm_{name}.keras")
        print(f"[{name}] test RMSE {r['test_rmse']:.2f} | NASA score {r['nasa_score']:.0f}")

    ex = report["experiments"]
    if "baseline" in ex and "final" in ex:
        b, f = ex["baseline"]["test_rmse"], ex["final"]["test_rmse"]
        report["rmse_improvement_pct"] = round(100 * (b - f) / b, 1)
        print(f"\nRMSE {b:.2f} -> {f:.2f}  ({report['rmse_improvement_pct']}% lower)")

    D.plot_predictions(y_true, preds, C.FIG_DIR / "predictions.png")
    D.plot_rmse_bars({LABELS[k]: v["test_rmse"] for k, v in ex.items()}, C.FIG_DIR / "rmse_comparison.png")
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS_DIR / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nSaved metrics to {C.RESULTS_DIR / 'metrics.json'} and figures to {C.FIG_DIR}")


if __name__ == "__main__":
    main()
