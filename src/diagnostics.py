"""Diagnostics & plots: why a global model fails on FD002 and what regimes fix."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

# Reference palette (validated categorical order) + recessive ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10, "lines.linewidth": 2,
    "legend.frameon": False,
})


def regime_variance_share(df: pd.DataFrame, sensors: list[str]) -> pd.Series:
    """Share of each sensor's variance explained by the flight regime alone.

    Values near 1.0 mean the signal is dominated by altitude/throttle, not wear.
    """
    total = df[sensors].var()
    within = df.groupby("regime")[sensors].var().mul(
        df.groupby("regime").size(), axis=0).sum() / len(df)
    return (1 - within / total).sort_values(ascending=False)


def plot_regimes(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k in sorted(df["regime"].unique()):
        d = df[df["regime"] == k]
        ax.scatter(d["setting_1"], d["setting_2"], s=70, color=SERIES[k], alpha=0.9,
                   edgecolors="none", label=f"Regime {k}")
        cx, cy = d["setting_1"].mean(), d["setting_2"].mean()
        ax.annotate(f"R{k}", (cx, cy), xytext=(6, 6), textcoords="offset points",
                    color=INK, fontsize=9, weight="bold")
    ax.set_xlabel("setting_1 (altitude, kft)")
    ax.set_ylabel("setting_2 (Mach)")
    ax.set_title("K-Means (k=6) recovers the six FD002 flight conditions", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9, markerscale=0.6)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_normalisation_effect(raw: pd.DataFrame, glob: pd.DataFrame, reg: pd.DataFrame,
                              sensor: str, unit: int, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for ax, d, title in [(axes[0], glob, "Global z-score: regime jumps swamp the trend"),
                         (axes[1], reg, "Regime z-score: degradation trend emerges")]:
        e = d[d["unit"] == unit]
        ax.plot(e["cycle"], e[sensor], color=SERIES[0], linewidth=1, alpha=0.45)
        ax.plot(e["cycle"], e[sensor].rolling(C.ROLL_WINDOW, min_periods=1).mean(),
                color=SERIES[1], linewidth=2, label=f"{C.ROLL_WINDOW}-cycle rolling mean")
        ax.set_title(title, loc="left"); ax.set_xlabel("cycle")
    axes[0].set_ylabel(f"{sensor} (z)")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Engine {unit}, sensor {sensor}", x=0.01, ha="left", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_predictions(y_true, preds: dict[str, np.ndarray], path):
    order = np.argsort(y_true)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(y_true)), np.asarray(y_true)[order], color=INK, linewidth=2, label="True RUL")
    for i, (name, p) in enumerate(preds.items()):
        ax.plot(np.arange(len(y_true)), np.asarray(p)[order], color=SERIES[i], linewidth=1.2,
                marker="o", markersize=3, alpha=0.8, label=name)
    ax.set_xlabel("test engines (sorted by true RUL)"); ax.set_ylabel("RUL (cycles)")
    ax.set_title("Test-set predictions vs ground truth", loc="left")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_rmse_bars(results: dict[str, float], path):
    names, vals = list(results), list(results.values())
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(names, vals, color=SERIES[0], height=0.55)
    bars[-1].set_color(SERIES[2])
    for b, v in zip(bars, vals):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", color=INK, fontsize=9)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Test RMSE (cycles, lower is better)")
    ax.set_title("FD002 test RMSE by pipeline", loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
