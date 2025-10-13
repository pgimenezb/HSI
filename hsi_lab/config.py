variables = {
    "data_folder": "/home/pgimenez/database",
    "excel_file": "/home/pgimenez/database/noms_fichiers.csv",
    "data_type": ["vis", "swir"],
    "start_index": 0,
    "num_files": 3,
    "selected_regions": [1],
    "selected_subregions": [1],
    "savgol_window": 0,
    "savgol_polyorder": 0,
    "num_regions": 4,

    "num_binder": {"Arabic Gum": 1, "Egg Tempera": 2},
    "num_mixture": {
        "Pure": 1,
        "Pigment + White titanium (80:20)": 2,
        "Pigment + White titanium (50:50)": 3,
        "Pigment + White titanium (20:80)": 4
    },

    "binder_columns":  {20: "Arabic Gum", 21: "Egg Tempera"},
    "mixture_columns": {22: "Pure", 23: "Pigment + White titanium (80:20)",
                        24: "Pigment + White titanium (50:50)",
                        25: "Pigment + White titanium (20:80)"},

    "meta_label_map": {
        1: ("Arabic Gum", "10"),
        2: ("Egg Tempera", "01"),
        3: ("Pure", "1000"),
        4: ("Pigment + White titanium (80:20)", "0100"),
        5: ("Pigment + White titanium (50:50)", "0010"),
        6: ("Pigment + White titanium (20:80)", "0001")
    },

    "binder_mapping":  {"10": "Arabic Gum", "01": "Egg Tempera"},
    "mixture_mapping": {"1000": "Pure",
                        "0100": "Pigment + White titanium (80:20)",
                        "0010": "Pigment + White titanium (50:50)",
                        "0001": "Pigment + White titanium (20:80)"},

    "selected_binders": [],

    "smoothing_method": "Without filter",
    "ai_architecture": "CNN",
    "target_mode": "pigments+binders",
    "model": "Model 1",

    "seed": 42,
    "outputs_dir": "outputs",
    "models_dir": "models"
}
