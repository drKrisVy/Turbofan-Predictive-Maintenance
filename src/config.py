"""Central configuration for the FD002 multi-regime RUL project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"

# --- Dataset source (Kaggle) -------------------------------------------------
# Primary:  https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
# Fallback: https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data
KAGGLE_DATASETS = [
    "behrad3d/nasa-cmaps",
    "palbha/cmapss-jet-engine-simulated-data",
]
SUBSET = "FD002"

# --- Column layout (NASA readme) --------------------------------------------
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTING_COLS + SENSOR_COLS

# --- Modelling hyper-parameters ----------------------------------------------
N_REGIMES = 6          # FD002 has six flight conditions (altitude / Mach / TRA)
RUL_CAP = 125          # piecewise-linear RUL target (standard for CMAPSS)
WINDOW = 30            # sequence length fed to the Bi-LSTM
ROLL_WINDOW = 30       # rolling-feature window (cycles)
SEED = 42

EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.2     # fraction of *engines* held out for validation
