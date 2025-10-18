variables = {
    "data_folder": "/home/pgimenez/database",
    "excel_file": "/home/pgimenez/database/noms_fichiers.csv",

    "data_type": ["vis", "swir"],
    "start_index": 0,
    # nº de pigmentos (bits delante de Multi)
    "num_files": 20,

    # sin filtros por defecto
    "selected_regions": [],
    "selected_subregions": [],

    "savgol_window": 0,
    "savgol_polyorder": 0,
    "num_regions": 1,  # puedes dejarlo así si no lo usas en lógica

    # --- SOLO MIXTURE ---
    "num_mixture": {
        "Pure": 1,
        "Pigment + White titanium (80:20)": 2,
        "Pigment + White titanium (50:50)": 3,
        "Pigment + White titanium (20:80)": 4
    },

    # Si no usas estos índices en ningún sitio, puedes borrarlo;
    # si los usas como metadatos, mantenlos tal cual.
    "mixture_columns": {
        22: "Pure",
        23: "Pigment + White titanium (80:20)",
        24: "Pigment + White titanium (50:50)",
        25: "Pigment + White titanium (20:80)"
    },

    # Map solo de mixture (bits → nombre)
    "mixture_mapping": {
        "1000": "Pure",
        "0100": "Pigment + White titanium (80:20)",
        "0010": "Pigment + White titanium (50:50)",
        "0001": "Pigment + White titanium (20:80)"
    },

    # Etiquetas “meta” solo con mixture
    "meta_label_map": {
        1: ("Pure", "1000"),
        2: ("Pigment + White titanium (80:20)", "0100"),
        3: ("Pigment + White titanium (50:50)", "0010"),
        4: ("Pigment + White titanium (20:80)", "0001")
    },

    # Eliminados: binder_mapping, binder_columns, num_binder, selected_binders
    # "binder_mapping": {},
    # "binder_columns": {},
    # "num_binder": {},
    # "selected_binders": [],

    "smoothing_method": "Without filter",
    "ai_architecture": "CNN",
    "target_mode": "pigments+mixture",  # ← antes decía pigments+binders

    "model": "Model 1",
    "seed": 42,

    "outputs_dir": "outputs",
    "models_dir":  "outputs",
    "model_list": ["cnn_baseline","cnn_dilated","cnn_residual", "dnn_baseline","dnn_selu","dnn_wide"],

    "trials": 50,
    "epochs": 50,

    # Si usas SQLite (optuna.db), mejor 1 para evitar bloqueos
    "optuna_n_jobs": 1,
    "n_jobs_models": 1,
}
