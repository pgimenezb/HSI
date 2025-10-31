import numpy as np

# ============================================================
# === GLOBAL CONFIGURATION VARIABLES ===
# ============================================================

variables = {
    "data_folder": "/home/pgimenez/projects/HSI/database",
    "excel_file": "/home/pgimenez/projects/HSI/database/noms_fichiers.csv",

    "data_type": ["vis", "swir"],
    "start_index": 0,
    "num_files": 21,

    "selected_regions": [],
    "selected_subregions": [],

    "savgol_window": 0,
    "savgol_polyorder": 0,
    "num_regions": 1,

    "num_mixture": {
        "Pure": 1,
        "Pigment + White titanium (80:20)": 2,
        "Pigment + White titanium (50:50)": 3,
        "Pigment + White titanium (20:80)": 4,
    },

    "mixture_columns": {
        22: "Pure",
        23: "Pigment + White titanium (80:20)",
        24: "Pigment + White titanium (50:50)",
        25: "Pigment + White titanium (20:80)",
    },

    "mixture_mapping": {
        "1000": "Pure",
        "0100": "Pigment + White titanium (80:20)",
        "0010": "Pigment + White titanium (50:50)",
        "0001": "Pigment + White titanium (20:80)",
    },

    "meta_label_map": {
        1: ("Pure", "1000"),
        2: ("Pigment + White titanium (80:20)", "0100"),
        3: ("Pigment + White titanium (50:50)", "0010"),
        4: ("Pigment + White titanium (20:80)", "0001"),
    },

    "smoothing_method": "Without filter",
    "ai_architecture": "CNN",
    "target_mode": "pigments+mixture",
    "model": "Model 1",
    "seed": 42,

    "outputs_dir": "outputs",
    "models_dir": "outputs",
    "model_list": [
        "cnn_baseline", "cnn_dilated", "cnn_residual",
        "dnn_baseline", "dnn_selu", "dnn_wide", "DNN"
    ],

    "trials": 50,
    "epochs": 50,
    "optuna_n_jobs": 1,
    "n_jobs_models": 1,
    "batch_size": 32,
    "region_row_quota": {1: 30, 2: 10, 3: 10, 4: 10},
    "subregion_row_quota": {},
    "balance_seed": 42,
    "test_per_mixture": 2,
    "balance_test_by_mixture": True,


     # Kubelka–Munk configuration
    "do_km_mixing": True,        # Enable or disable synthetic KM stage
    "km_n_samples": 2000,        # Number of synthetic mixtures to generate
    "km_n_mix": (2, 3),          # Range of components per mixture (binary/ternary)
    "km_method": "minimize",      # "minimize" for fast unmixing or "lstsq" for fast unmixing
    "km_method": "nnls"
}
