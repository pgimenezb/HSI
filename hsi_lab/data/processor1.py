import re
import os
import sys
import importlib
import subprocess
import inspect
import glob
from copy import deepcopy
import json
import h5py
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown, IntSlider
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
)
from sklearn.decomposition import PCA
import random
from collections import defaultdict
import tensorflow as tf
import optuna
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    BatchNormalization,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit, cross_val_score
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.model_selection import iterative_train_test_split
from scipy.signal import savgol_filter
import matplotlib.patches as patches
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from numpy.random import default_rng
from IPython import get_ipython
from pathlib import Path
import pickle
import hashlib


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_to_jsonable(list(obj)))
    if isinstance(obj, Path):
        return str(obj)
    # numpy -> python
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _stable_hash(payload) -> str:
    data = _to_jsonable(payload)
    raw = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()  # clave corta y estable para caché


class HSIDataProcessor:
    SEED = 42

    def __init__(self, variables):
        self.set_global_seed()

        self.data_folder = variables['data_folder']
        self.excel_file = variables['excel_file']
        self.file_names = pd.read_csv(self.excel_file) if self.excel_file else None
        print(f"active data_folder: {self.data_folder}")
        print(f"active excel_file: {self.excel_file}")

        self.all_data = {}
        self.loaded_files_range = (0, 0)
        self.binary_representations = {}
        self.Pigment_nom = None

        self.start_index = variables['start_index']
        self.num_files = variables['num_files']

        self.selected_regions = variables['selected_regions']
        self.selected_subregions = variables['selected_subregions']
        self.region_conditions = {
            1: lambda v: v[-7] == 1 and v[-4] == 1,
            2: lambda v: v[-6] == 1 and v[-4] == 1,
            3: lambda v: v[-7] == 1 and v[-3] == 1,
            4: lambda v: v[-6] == 1 and v[-3] == 1,
            5: lambda v: v[-7] == 1 and v[-2] == 1,
            6: lambda v: v[-6] == 1 and v[-2] == 1,
            7: lambda v: v[-7] == 1 and v[-1] == 1,
            8: lambda v: v[-6] == 1 and v[-1] == 1,
        }
        self.selected_binders = variables['selected_binders']
        self.data_type = variables['data_type']
        self.points = [
            (21, 9), (81, 9), (81, 32), (47, 47), (20, 60),
            (60, 68), (85, 75), (20, 90), (90, 93),
        ]
        self.savgol_window = variables['savgol_window']
        self.savgol_polyorder = variables['savgol_polyorder']
        self.num_binder = variables['num_binder']
        self.num_mixture = variables['num_mixture']
        self.binder_columns = variables['binder_columns']
        self.mixture_columns = variables['mixture_columns']
        self.all_pigment_columns = pd.read_csv(self.excel_file)['Title'].tolist() if self.excel_file else []
        self.meta_label_map = variables['meta_label_map']
        self.binder_mapping = variables['binder_mapping']
        self.mixture_mapping = variables['mixture_mapping']
        self.pigments_mapping = {}
        self._multi_to_label = None

        # copia jsonable de variables para hashing estable
        self.variables = variables.copy()
        self.variables["binder_columns"] = {
            int(k): v for k, v in self.variables.get("binder_columns", {}).items()
        }
        self.variables["mixture_columns"] = {
            int(k): v for k, v in self.variables.get("mixture_columns", {}).items()
        }
        self.binder_columns = self.variables["binder_columns"]
        self.mixture_columns = self.variables["mixture_columns"]

        self._df_cache = None

    def set_global_seed(self):
        seed = self.SEED
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # LOADING .H5 FILES AND FILTERING DATA BASED ON SELECTED PIGMENTS
    def load_h5_files(self):
        self.loaded_files_range = (self.start_index, self.start_index + self.num_files)
        self.all_data = {}

        for fichier in self.file_names.Title[self.start_index:self.start_index + self.num_files]:
            file_path = os.path.join(self.data_folder, fichier)
            try:
                h5_file = h5py.File(file_path, 'r')

                # Load original arrays (without filters)
                binder = h5_file['labels/labels_binder'][()]
                mixture = h5_file['labels/labels_mixture'][()]
                multilabel = h5_file['labels/vector_multilabel'][()]

                # Save directly into filtered_labels
                h5_filtered = {
                    'labels/labels_binder': binder,
                    'labels/labels_mixture': mixture,
                    'labels/vector_multilabel': multilabel,
                }

                h5_file.filtered_labels = h5_filtered
                self.all_data[fichier] = h5_file

            except Exception as e:
                print(f"Error loading {fichier}: {e}")

        # Keep in file_names only those we actually loaded
        self.file_names = self.file_names[self.file_names['Title'].isin(self.all_data.keys())]

        print(f"{len(self.all_data)} loaded files.")
        if self.all_data:
            last_key = list(self.all_data.keys())[-1]
            print(f"Last file loaded: {last_key}")
        else:
            print("No valid files loaded.")

        return self.all_data

    def dataframe(self):
        import numpy as np
        import pandas as pd

        # =======================
        #   Helpers / Layout
        # =======================
        def _first_one_idx(bitstr: str) -> int:
            idxs = [i for i, c in enumerate(bitstr) if c == '1']
            if len(idxs) != 1:
                raise ValueError(f"Bitstring inválido (un solo '1'): {bitstr}")
            return idxs[0]

        multilabel_rows = []

        # --- 20 pigmentos fijos delante (usa num_files de config por defecto) ---
        NUM_PIGMENTS = int(getattr(self, "num_files", 20))

        # --- Mappings ---
        if not getattr(self, "binder_mapping", None):
            raise ValueError("self.binder_mapping vacío o no definido")
        if not getattr(self, "mixture_mapping", None):
            raise ValueError("self.mixture_mapping vacío o no definido")

        binder_bits = list(self.binder_mapping.keys())
        binder_len = len(binder_bits[0])
        if not all(len(b) == binder_len for b in binder_bits):
            raise ValueError("Todos los bitstrings de binder deben tener la misma longitud")

        mixture_bits = list(self.mixture_mapping.keys())
        mixture_len = len(mixture_bits[0])
        if not all(len(b) == mixture_len for b in mixture_bits):
            raise ValueError("Todos los bitstrings de mixture deben tener la misma longitud")

        # Bloques
        BINDER_BASE = NUM_PIGMENTS
        MIX_BASE = NUM_PIGMENTS + binder_len
        TOTAL_LEN = NUM_PIGMENTS + binder_len + mixture_len

        # nombre -> offset dentro de su bloque (0-based), derivado del bit '1'
        binder_pos_by_name = {name: _first_one_idx(bits) for bits, name in self.binder_mapping.items()}
        mixture_pos_by_name = {name: _first_one_idx(bits) for bits, name in self.mixture_mapping.items()}

        # Orden de mixtures por posición del '1'
        mixture_names_ordered = [
            self.mixture_mapping[bits]
            for bits in sorted(self.mixture_mapping.keys(), key=_first_one_idx)
        ]

        # Para traducir número->nombre (viene del binder_map)
        num_binder = getattr(self, "num_binder", {})
        inv_num_binder = {v: k for k, v in num_binder.items()}
        num_mixture = getattr(self, "num_mixture", {})

        # Filtros opcionales
        sel_regions = set(getattr(self, "selected_regions", []) or [])
        sel_subregs = set(getattr(self, "selected_subregions", []) or [])

        # =======================
        #      Recorrido H5
        # =======================
        for file_name, h5_file in self.all_data.items():
            data = h5_file.filtered_labels["labels/vector_multilabel"]  # (H, W, D_in)
            H, W, D_in = data.shape

            # binder/mixture maps
            try:
                binder_map = h5_file.filtered_labels["labels/labels_binder"]
                mixture_map = h5_file.filtered_labels["labels/labels_mixture"]
                assert binder_map.shape == (H, W) and mixture_map.shape == (H, W)
            except Exception:
                binder_map, mixture_map = None, None

            binder_all_ones = False
            if binder_map is not None:
                u = np.unique(binder_map)
                binder_all_ones = (u.size == 1 and u[0] == 1)

            # Rangos para region/subregion (solo metadatos)
            binder_ranges, mixture_ranges = {}, {}
            if binder_map is not None:
                for bnum in np.unique(binder_map):
                    if bnum == 0:
                        continue
                    cols = np.any(binder_map == bnum, axis=0)
                    xs = np.where(cols)[0]
                    if xs.size:
                        binder_ranges[int(bnum)] = (int(xs[0]), int(xs[-1]) + 1)
            if mixture_map is not None:
                for mnum in np.unique(mixture_map):
                    if mnum == 0:
                        continue
                    rows = np.any(mixture_map == mnum, axis=1)
                    ys = np.where(rows)[0]
                    if ys.size:
                        mixture_ranges[int(mnum)] = (int(ys[0]), int(ys[-1]) + 1)
            n_binders = len(binder_ranges) if binder_ranges else 0

            # píxeles con alguna etiqueta activa
            mask = data.any(axis=2)
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                continue

            for y, x in zip(ys, xs):
                v = data[y, x, :]

                # 1) Pigmento
                if not np.any(v[:NUM_PIGMENTS]):
                    continue
                pigment_idx = int(np.argmax(v[:NUM_PIGMENTS]))

                # 2) Binder forzado
                if binder_all_ones:
                    forced_binder_name = "Arabic Gum"
                    bnum_px = num_binder.get("Arabic Gum", 1)
                else:
                    forced_binder_name = None
                    bnum_px = 0
                    if binder_map is not None:
                        bnum_px = int(binder_map[y, x])
                        if bnum_px > 0:
                            forced_binder_name = inv_num_binder.get(bnum_px, None)
                    if not forced_binder_name:
                        forced_binder_name = "Arabic Gum" if "Arabic Gum" in binder_pos_by_name else next(iter(binder_pos_by_name.keys()))
                        bnum_px = num_binder.get(forced_binder_name, 0)

                # 3) Región/Subregión (metadatos)
                region = 0
                subregion = 0
                if binder_map is not None and mixture_map is not None:
                    mnum_px = int(mixture_map[y, x])
                    if bnum_px > 0 and mnum_px > 0 and n_binders > 0:
                        region = (mnum_px - 1) * n_binders + bnum_px
                        if (bnum_px in binder_ranges) and (mnum_px in mixture_ranges):
                            x0, x1 = binder_ranges[bnum_px]
                            y0, y1 = mixture_ranges[mnum_px]
                            xmid = (x0 + x1) / 2.0
                            ymid = (y0 + y1) / 2.0
                            if x < xmid and y >= ymid:
                                subregion = 1
                            elif x >= xmid and y >= ymid:
                                subregion = 2
                            elif x < xmid and y < ymid:
                                subregion = 3
                            else:
                                subregion = 4

                if sel_regions and region not in sel_regions:
                    continue
                if sel_subregs and subregion not in sel_subregs:
                    continue

                # 4) Generar SIEMPRE 4 mixtures en el orden 1000,0100,0010,0001
                for m_name in mixture_names_ordered:
                    full_vector = [0] * TOTAL_LEN

                    # Pigmento
                    full_vector[pigment_idx] = 1

                    # Binder (bits detrás de pigmentos)
                    b_off = binder_pos_by_name.get(forced_binder_name)
                    if b_off is None:
                        # fallback al primero disponible
                        forced_binder_name, b_off = next(iter(binder_pos_by_name.items()))
                    full_vector[BINDER_BASE + b_off] = 1

                    # Mixture (bits a continuación del binder)
                    m_off = mixture_pos_by_name[m_name]
                    full_vector[MIX_BASE + m_off] = 1

                    multilabel_rows.append({
                        "File": str(file_name).strip(),
                        "X": int(x),
                        "Y": int(y),
                        "Pigment Index": pigment_idx,
                        "Binder": forced_binder_name,
                        "Mixture": m_name,
                        "Num. Binder": num_binder.get(forced_binder_name, 0),
                        "Num. Mixture": num_mixture.get(m_name, 0),
                        "Region": region,
                        "Subregion": subregion,
                        "Multi": full_vector,
                    })

        df_multilabel = pd.DataFrame(multilabel_rows)

        # ====== CUBOS / SPECTRUM ======
        def build_cube_df(cube_type):
            dfs = []
            for file_name, h5_file in self.all_data.items():
                if cube_type not in h5_file:
                    continue
                cube_data = h5_file[f"{cube_type}/data"][()]  # (H, W, bands)
                h, w, bands = cube_data.shape
                ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                ys, xs = ys.ravel(), xs.ravel()
                base = pd.DataFrame({"File": str(file_name).strip(), "X": xs, "Y": ys})
                val_cols = [f"{cube_type}_{i+1}" for i in range(bands)]
                vals = pd.DataFrame(cube_data[ys, xs, :].astype(np.float32, copy=False), columns=val_cols)
                df = pd.concat([base, vals], axis=1)
                dfs.append(df)
            if not dfs:
                return pd.DataFrame()
            df_type = pd.concat(dfs, ignore_index=True)
            df_type = df_type.sort_values(["File", "Y", "X"], kind="mergesort").reset_index(drop=True)
            return df_type

        cube_types = self.data_type
        if isinstance(cube_types, str):
            cube_types = [cube_types]
        elif not isinstance(cube_types, (list, tuple)):
            cube_types = [str(cube_types)]

        cube_frames = {t: build_cube_df(t) for t in cube_types}

        df_cubes = None
        for t, df_t in cube_frames.items():
            if df_t.empty:
                continue
            cols = ["File", "X", "Y"] + [c for c in df_t.columns if c.startswith(f"{t}_")]
            df_t = df_t[cols]
            df_cubes = df_t if df_cubes is None else df_cubes.merge(df_t, on=["File", "X", "Y"], how="inner")

        if not df_multilabel.empty:
            df_multilabel = df_multilabel.sort_values(
                ["File", "Region", "Mixture", "Y", "X"],
                kind="mergesort"
            ).reset_index(drop=True)
            df_multilabel["Spectrum"] = (
                df_multilabel.groupby("File").cumcount() + 1
            ).astype(str).radd("Spectrum_")

        if not df_multilabel.empty and df_cubes is not None and not df_cubes.empty:
            df_merged = df_multilabel.merge(df_cubes, on=["File", "X", "Y"], how="left")
        else:
            df_merged = df_cubes if (df_cubes is not None and not df_cubes.empty) else df_multilabel

        return df_merged
