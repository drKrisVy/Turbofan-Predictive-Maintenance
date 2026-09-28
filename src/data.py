"""Download (from Kaggle) and load the NASA CMAPSS FD002 subset."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

REQUIRED = [f"train_{C.SUBSET}.txt", f"test_{C.SUBSET}.txt", f"RUL_{C.SUBSET}.txt"]


def _have_files(d: Path) -> bool:
    return all((d / f).exists() for f in REQUIRED)


def download_from_kaggle(dest: Path = C.DATA_DIR) -> Path:
    """Fetch the CMAPSS files from Kaggle via kagglehub and copy FD002 into `dest`.

    Public datasets usually download without credentials. If Kaggle asks for
    auth, create an API token at https://www.kaggle.com/settings and place it at
    ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME / KAGGLE_KEY).
    """
    dest.mkdir(parents=True, exist_ok=True)
    if _have_files(dest):
        return dest

    import kagglehub  # imported lazily so the rest of the code works offline

    last_err = None
    for slug in C.KAGGLE_DATASETS:
        try:
            print(f"[data] downloading https://www.kaggle.com/datasets/{slug} ...")
            root = Path(kagglehub.dataset_download(slug))
            for name in REQUIRED:
                hits = list(root.rglob(name))
                if not hits:
                    raise FileNotFoundError(f"{name} not found in {slug}")
                shutil.copy(hits[0], dest / name)
            print(f"[data] FD002 files copied to {dest}")
            return dest
        except Exception as e:  # try the next mirror
            print(f"[data] {slug} failed: {e}")
            last_err = e
    raise RuntimeError(
        "Could not download CMAPSS from Kaggle. Download manually from "
        "https://www.kaggle.com/datasets/behrad3d/nasa-cmaps and place "
        f"{REQUIRED} in {dest}"
    ) from last_err


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(C.COLUMNS)]  # raw files have trailing blank columns
    df.columns = C.COLUMNS
    return df


def load_fd002(data_dir: Path = C.DATA_DIR):
    """Return (train_df, test_df, true_rul) with a capped RUL target on train."""
    if not _have_files(data_dir):
        download_from_kaggle(data_dir)

    train = _read(data_dir / f"train_{C.SUBSET}.txt")
    test = _read(data_dir / f"test_{C.SUBSET}.txt")
    true_rul = np.loadtxt(data_dir / f"RUL_{C.SUBSET}.txt")

    # Train target: cycles remaining until failure, capped (piecewise linear).
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["RUL"] = (max_cycle - train["cycle"]).clip(upper=C.RUL_CAP)
    return train, test, true_rul
