import re
import os, sys, importlib, subprocess, inspect, glob
from copy import deepcopy
import json
import h5py
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown, IntSlider
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, log_loss, make_scorer, precision_score, recall_score, f1_score, hamming_loss 
from sklearn.decomposition import PCA
import random 
from collections import defaultdict
import tensorflow as tf
import optuna
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from tensorflow.keras.regularizers import l2
import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.model_selection import iterative_train_test_split
from scipy.signal import savgol_filter
import matplotlib.patches as patches
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from numpy.random import default_rng
from IPython import get_ipython

class HSIDataProcessor:

    SEED = 42

    def __init__(self, variables):
        self.set_global_seed()

        self.data_folder = variables['data_folder']   # cadena fija
        self.excel_file  = variables['excel_file']    # cadena fija
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
            8: lambda v: v[-6] == 1 and v[-1] == 1
        }
        self.selected_binders = variables['selected_binders']
        self.data_type = variables['data_type']
        self.points = [(21, 9), (81, 9), (81, 32), (47, 47), (20, 60), (60, 68), (85, 75), (20, 90), (90, 93)]
        self.savgol_window = variables['savgol_window']
        self.savgol_polyorder = variables['savgol_polyorder']
        self.num_binder = variables['num_binder']
        self.num_mixture = variables['num_mixture']
        self.binder_columns = variables['binder_columns']
        self.mixture_columns = variables['mixture_columns']
        self.all_pigment_columns = pd.read_csv(self.excel_file)['Title'].tolist()
        self.meta_label_map = variables['meta_label_map']
        self.binder_mapping = variables['binder_mapping']
        self.mixture_mapping = variables['mixture_mapping']
        self.pigments_mapping = {}
        self._multi_to_label = None
        self.variables = variables.copy()
        self.variables["binder_columns"]  = {int(k): v for k, v in self.variables.get("binder_columns", {}).items()}
        self.variables["mixture_columns"] = {int(k): v for k, v in self.variables.get("mixture_columns", {}).items()}
        self.binder_columns  = self.variables["binder_columns"]
        self.mixture_columns = self.variables["mixture_columns"]



    def set_global_seed(self):
        seed = self.SEED  # Use the class's internal seed
        os.environ['PYTHONHASHSEED'] = str(seed)  # Fix randomness in internal operations
        random.seed(seed)                         # Python random
        np.random.seed(seed)                      # Numpy
        tf.random.set_seed(seed)                  # TensorFlow


# LOADING .H5 FILES AND FILTERING DATA BASED ON SELECTED PIGMENTS
    def load_h5_files(self): 
        self.loaded_files_range = (self.start_index, self.start_index + self.num_files)
        self.all_data = {}

        for fichier in self.file_names.Title[self.start_index:self.start_index + self.num_files]:
            file_path = os.path.join(self.data_folder, fichier)
            try:
                h5_file = h5py.File(file_path, 'r')

                # Load original arrays (without filters)
                binder     = h5_file['labels/labels_binder'][()]
                mixture    = h5_file['labels/labels_mixture'][()]
                multilabel = h5_file['labels/vector_multilabel'][()]

                # Save directly into filtered_labels
                h5_filtered = {
                    'labels/labels_binder': binder,
                    'labels/labels_mixture': mixture,
                    'labels/vector_multilabel': multilabel
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

        # --- 20 pigmentos fijos delante ---
        NUM_PIGMENTS = int(getattr(self, "num_pigments", 20))

        # --- Mappings que defines en self ---
        # binder_mapping:  {"10": "Arabic Gum", "01": "Egg Tempera"}
        # mixture_mapping: {"1000": "Pure", "0100": "...", "0010": "...", "0001": "..."}
        if not getattr(self, "binder_mapping", None):
            raise ValueError("self.binder_mapping vacío o no definido")
        if not getattr(self, "mixture_mapping", None):
            raise ValueError("self.mixture_mapping vacío o no definido")

        binder_bits = list(self.binder_mapping.keys())
        binder_len = len(binder_bits[0])
        if not all(len(b) == binder_len for b in binder_bits):
            raise ValueError("Todos los bitstrings de binder deben tener la misma longitud")
        if binder_len != 2:
            # si un día cambias a 3 binders (len 3) esto seguirá funcionando; no es obligatorio que sea 2
            pass

        mixture_bits = list(self.mixture_mapping.keys())
        mixture_len = len(mixture_bits[0])
        if not all(len(b) == mixture_len for b in mixture_bits):
            raise ValueError("Todos los bitstrings de mixture deben tener la misma longitud")

        # Bloques
        BINDER_BASE = NUM_PIGMENTS
        MIX_BASE    = NUM_PIGMENTS + binder_len
        TOTAL_LEN   = NUM_PIGMENTS + binder_len + mixture_len

        # nombre -> offset dentro de su bloque (0-based), derivado del bit '1'
        binder_pos_by_name = {name: _first_one_idx(bits) for bits, name in self.binder_mapping.items()}
        mixture_pos_by_name = {name: _first_one_idx(bits) for bits, name in self.mixture_mapping.items()}

        # Orden de mixtures por posición del '1' (1000,0100,0010,0001)
        mixture_names_ordered = [
            self.mixture_mapping[bits]
            for bits in sorted(self.mixture_mapping.keys(), key=_first_one_idx)
        ]

        # Para traducir número->nombre (viene del binder_map)
        # self.num_binder p.ej. {"Arabic Gum":1, "Egg Tempera":2}
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

            # Solo usamos binder_map para forzar binder
            try:
                binder_map  = h5_file.filtered_labels["labels/labels_binder"]
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

                # 1) Pigmento (0..19 en el tensor de entrada)
                if not np.any(v[:NUM_PIGMENTS]):
                    continue
                pigment_idx = int(np.argmax(v[:NUM_PIGMENTS]))

                # 2) Binder forzado por binder_map (o all-ones -> Arabic Gum)
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
                        # si no hay nombre, prioriza AG si existe en mapping
                        forced_binder_name = "Arabic Gum" if "Arabic Gum" in binder_pos_by_name else next(iter(binder_pos_by_name.keys()))
                        bnum_px = num_binder.get(forced_binder_name, 0)

                # 3) Región/Subregión (metadatos, no afecta al vector)
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
                            if x < xmid and y >= ymid:   subregion = 1
                            elif x >= xmid and y >= ymid: subregion = 2
                            elif x < xmid and y < ymid:   subregion = 3
                            else:                          subregion = 4

                if sel_regions and region not in sel_regions:
                    continue
                if sel_subregs and subregion not in sel_subregs:
                    continue

                # 4) Generar SIEMPRE 4 mixtures en el orden 1000,0100,0010,0001
                for m_name in mixture_names_ordered:
                    full_vector = [0] * TOTAL_LEN

                    # Pigmento
                    full_vector[pigment_idx] = 1

                    # Binder (2 bits, p. ej. AG=10, ET=01) — bloque justo detrás de pigmentos
                    # Primero limpia ambos bits (ya están en cero), y activa el que corresponda
                    b_off = binder_pos_by_name.get(forced_binder_name)
                    if b_off is None:
                        # si el nombre no está, usa el primero disponible
                        forced_binder_name, b_off = next(iter(binder_pos_by_name.items()))
                        # OJO: b_off aquí sería offset si vienes del dict en forma {name:offset}
                        # el next iter(binder_pos_by_name.items()) devuelve (name, offset)
                        # Corrige:
                        # forced_binder_name, b_off = forced_binder_name, b_off
                    full_vector[BINDER_BASE + b_off] = 1  # <-- aquí garantizamos 2 bits para binder

                    # Mixture (4 bits) — bloque a continuación del binder
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
                        "Multi": full_vector
                    })

        df_multilabel = pd.DataFrame(multilabel_rows)

        # ====== (tu bloque de cubos / Spectrum igual que ya tenías) ======
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
            df_multilabel = df_multilabel.sort_values(["File", "Region", "Mixture", "Y", "X"], kind="mergesort").reset_index(drop=True)
            df_multilabel["Spectrum"] = (df_multilabel.groupby("File").cumcount() + 1).astype(str).radd("Spectrum_")

        if not df_multilabel.empty and df_cubes is not None and not df_cubes.empty:
            df_merged = df_multilabel.merge(df_cubes, on=["File", "X", "Y"], how="left")
        else:
            df_merged = df_cubes if (df_cubes is not None and not df_cubes.empty) else df_multilabel

        return df_merged




    def plot_binder_mixture_panel(self, h5_file, pigment_index=None, fichier_name=None):
            binder = h5_file.filtered_labels['labels/labels_binder']
            mixture = h5_file.filtered_labels['labels/labels_mixture']
            multilabel = h5_file.filtered_labels['labels/vector_multilabel']

            assert binder.shape == mixture.shape, "The dimensions of 'binder' and 'mixture' do not match."
            height, width = binder.shape

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.invert_yaxis()
            cmap = plt.get_cmap('tab20')
            color_idx = 0

            # Binders
            binder_ticks = []
            binder_tick_labels = []
            unique_binders = np.unique(binder)
            for b in unique_binders:
                if b == 0:
                    continue
                mask = (binder == b)
                cols = np.any(mask, axis=0)
                col_indices = np.where(cols)[0]
                if col_indices.size == 0:
                    continue
                x_min, x_max = col_indices[0], col_indices[-1] + 1
                ax.add_patch(patches.Rectangle((x_min, 0), x_max - x_min, height,
                                            facecolor=cmap(color_idx), alpha=0.2, edgecolor='none'))
                center_x = (x_min + x_max) / 2
                binder_ticks.append(center_x)
                binder_tick_labels.append(list(self.num_binder.keys())[int(b) - 1])
                color_idx += 1

            # Mixtures
            color_idx = 10
            mixture_ticks = []
            mixture_tick_labels = []
            unique_mixture = np.unique(mixture)
            for v in unique_mixture:
                if v == 0:
                    continue
                mask = (mixture == v)
                rows = np.any(mask, axis=1)
                row_indices = np.where(rows)[0]
                if row_indices.size == 0:
                    continue
                y_min, y_max = row_indices[0], row_indices[-1] + 1
                ax.add_patch(patches.Rectangle((0, y_min), width, y_max - y_min,
                                            facecolor=cmap(color_idx), alpha=0.2, edgecolor='none'))
                center_y = (y_min + y_max) / 2
                mixture_ticks.append(center_y)
                mixture_tick_labels.append(list(self.num_mixture.keys())[int(v) - 1])
                color_idx += 1

            # Pigment overlay
            if pigment_index is not None:
                if pigment_index < multilabel.shape[2]:  
                    pigment_layer = multilabel[:, :, pigment_index]
                    yx = np.argwhere(pigment_layer == 1)
                    for y, x in yx:
                        ax.add_patch(patches.Rectangle((x, y), 1, 1,
                                                    facecolor='red', edgecolor='black', linewidth=1.0))

            ax.set_xticks(binder_ticks)
            ax.set_xticklabels(binder_tick_labels, rotation=0)
            ax.set_yticks(mixture_ticks)
            ax.set_yticklabels(mixture_tick_labels)
            title = f"Pigment {fichier_name[3:-3]}" if fichier_name else f"Pigment {pigment_index}" if pigment_index is not None else "Binder & Mixture Panel"
            ax.set_title(title)


            # ➕ subregions
            region_centers = {}  
            region_number = 1
            for mixture_y, _ in zip(mixture_ticks, mixture_tick_labels):
                for binder_x, _ in zip(binder_ticks, binder_tick_labels):
                    region_centers[region_number] = (binder_x, mixture_y)  # center
                    if region_number in self.selected_regions:
                        ax.text(binder_x, mixture_y, str(region_number),
                                ha='center', va='center', fontsize=12,
                                color='green', fontweight='bold')
                    else:
                        ax.text(binder_x, mixture_y, str(region_number),
                                ha='center', va='center', fontsize=8,
                                color='black')
                    region_number += 1


            rect_h = 15                # height
            rect_w = rect_h * 3        # width (x3)
            gap_x  = 4                 # gap between columns
            gap_y  = 4                 # gap between rows

            for region_num in self.selected_regions:
                center = region_centers.get(region_num)
                if not center:
                    continue
                x_center, y_center = center

                # centers of each sub-rect, centered relative to the region center
                cx_left  = x_center - (gap_x/2 + rect_w/2)
                cx_right = x_center + (gap_x/2 + rect_w/2)
                cy_bottom = y_center - (gap_y/2 + rect_h/2)   # ↓ lower half
                cy_top    = y_center + (gap_y/2 + rect_h/2)   # ↑ upper half

                centers = {
                    1: (cx_left,  cy_bottom),
                    2: (cx_right, cy_bottom),
                    3: (cx_left,  cy_top),
                    4: (cx_right, cy_top),
                }

                for idx in (1, 2, 3, 4):
                    if idx not in self.selected_subregions:
                        continue
                    cx, cy = centers[idx]
                    x0 = cx - rect_w/2
                    y0 = cy - rect_h/2

                    ax.add_patch(patches.Rectangle(
                        (x0, y0), rect_w, rect_h,
                        edgecolor='grey', facecolor='none', linewidth=1.0, linestyle='-'
                    ))
                    ax.text(cx, cy, f"{idx}", ha='center', va='center', fontsize=6, color='grey')


            plt.tight_layout()
            plt.show()


    def all_data_interactive(self):
        # Visible options
        data_types = ['VIS+SWIR', 'vis', 'swir', 'Maxrf', 'lis 655', 'lis 365']
        default_type = self.data_type if isinstance(self.data_type, str) and self.data_type in data_types else 'VIS+SWIR'

        # Calibration (adjust if your sensors use other values)
        VIS_START, VIS_END, VIS_N = 395.3, 1019.61, 785
        SWIR_START, SWIR_END, SWIR_N = 1098.03, 2577.88, 225
        VIS_STEP  = (VIS_END  - VIS_START)  / (VIS_N  - 1)
        SWIR_STEP = (SWIR_END - SWIR_START) / (SWIR_N - 1)

        def wl_for(kind: str, n_ch: int):
            if kind == 'vis':   
                return VIS_START + np.arange(n_ch, dtype=float) * VIS_STEP
            if kind == 'swir':
                return SWIR_START + np.arange(n_ch, dtype=float) * SWIR_STEP
            return np.arange(n_ch, dtype=float)

        def nice_ticks(wl: np.ndarray, prefix: str, k: int = 10):
            if wl.size == 0: return [], []
            idx = np.linspace(0, wl.size - 1, min(k, wl.size)).round().astype(int).tolist()
            lab = [f"{prefix} {i} ({wl[i]:.1f} nm)" for i in idx]
            return idx, lab

        @interact(
            fichier_name=self.file_names.Title,
            Data_type=Dropdown(options=data_types, value=default_type),
            Channel=IntSlider(min=1, max=785, step=1, value=100)
        )
        def plot_for_file(fichier_name, Data_type, Channel):
            My_file = self.all_data[fichier_name]

            # image index
            spectrum_num = int(
                self.dataframe()[self.dataframe()['File'] == fichier_name]['Spectrum'].values[0].split('_')[1]
            ) - 1

            window_length = self.savgol_window
            polyorder     = self.savgol_polyorder
            display_name  = 'vis' if Data_type.lower() == 'vis' else Data_type

            try:
                # --- Build cube + X axis ---
                if Data_type == 'VIS+SWIR':
                    if 'vis' not in My_file or 'swir' not in My_file:
                        raise KeyError("Missing 'vis' and/or 'swir' groups in the HDF5 for VIS+SWIR.")

                    data_vis  = My_file['vis']['data'][()]   # (H,W,785)
                    data_swir = My_file['swir']['data'][()]  # (H,W,225)
                    data = np.concatenate([data_vis, data_swir], axis=2)

                    wl_vis  = wl_for('vis',  data_vis.shape[2])
                    wl_swir = wl_for('swir', data_swir.shape[2])
                    wavelengths = np.concatenate([wl_vis, wl_swir])

                    idx_vis,  lab_vis  = nice_ticks(wl_vis,  'VIS',  k=8)
                    idx_swir, lab_swir = nice_ticks(wl_swir, 'SWIR', k=8)
                    xticks_pos    = idx_vis + [len(wl_vis) + i for i in idx_swir]
                    xticks_labels = lab_vis + lab_swir
                else:
                    if Data_type not in My_file or 'data' not in My_file[Data_type]:
                        raise KeyError(f"The group/dataset '{Data_type}/data' does not exist in HDF5.")
                    data = My_file[Data_type]['data'][()]
                    wavelengths = wl_for(Data_type, data.shape[2])
                    xticks_pos, xticks_labels = nice_ticks(wavelengths, Data_type.upper(), k=10)

                # safety: equal lengths
                if wavelengths.shape[0] != data.shape[2]:
                    wavelengths = np.arange(data.shape[2], dtype=float)
                    xticks_pos, xticks_labels = nice_ticks(wavelengths, 'CH', k=10)

                # extra labels
                labels_mixture = My_file['labels']['labels_mixture'][:, :94]
                labels_binder  = My_file['labels']['labels_binder'][:, :94]

                # clamp image index
                spectrum_num = max(0, min(spectrum_num, data.shape[2] - 1))

                # --- FIGURE 1×5 ---
                fig, axes = plt.subplots(
                    1, 5, figsize=(26, 6),  # wider for 5 panels
                    constrained_layout=True
                )
                ax_img, ax_vis, ax_swir, ax_bind, ax_varn = axes

                labels_pts = [f'P{i+1}' for i in range(len(self.points))]

                # Spectral image
                ax_img.imshow(data[:, :, spectrum_num])
                ax_img.set_title(f'{display_name.upper()} Data')

                # --- VIS spectra ---
                if 'vis' in My_file:
                    for i, (x, y) in enumerate(self.points):
                        spectrum = data_vis[x, y].astype(float)
                        wl = max(3, min(spectrum.size - 1, window_length))
                        if wl % 2 == 0: wl = max(3, wl - 1)
                        po = min(polyorder, wl - 1)
                        spectrum = savgol_filter(spectrum, wl, po)
                        ax_vis.plot(wl_vis, spectrum, label=f'{labels_pts[i]} ({x},{y})')

                    ax_vis.set_title("VIS (395–1019 nm)")
                    ax_vis.set(xlabel="Wavelength (nm)", ylabel="Reflectance (%)")

                # --- SWIR spectra ---
                if 'swir' in My_file:
                    for i, (x, y) in enumerate(self.points):
                        spectrum = data_swir[x, y].astype(float)
                        wl = max(3, min(spectrum.size - 1, window_length))
                        if wl % 2 == 0: wl = max(3, wl - 1)
                        po = min(polyorder, wl - 1)
                        spectrum = savgol_filter(spectrum, wl, po)
                        ax_swir.plot(wl_swir, spectrum, label=f'{labels_pts[i]} ({x},{y})')

                    ax_swir.set_title("SWIR (1098–2577 nm)")
                    ax_swir.set(xlabel="Wavelength (nm)")

                # Maps
                ax_bind.imshow(labels_binder);  ax_bind.set_title('Binders')
                ax_varn.imshow(labels_mixture); ax_varn.set_title('Mixtures')

                # Points over the maps and image
                for i, (x, y) in enumerate(self.points):
                    for ax in (ax_img, ax_bind, ax_varn):
                        ax.scatter(x, y, color='red')
                        ax.text(x, y - 3, labels_pts[i], color='black', fontsize=12,
                                ha='center', va='center')

                # Legend (below the SWIR subplot)
                handles, lbls = ax_swir.get_legend_handles_labels()
                leg = ax_swir.legend(
                    handles, lbls,
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.35),
                    ncol=3,
                    title="Spectra",
                    frameon=False,
                    bbox_transform=ax_swir.transAxes
                )

                plt.show()


            except KeyError as e:
                print(f"Data error: {e}")


    def _get_wavelengths_and_labels(self, columns):
            wavelengths = []
            xticks_positions = []
            xticks_labels = []

            # ---- vis parameters ----
            vis_start = 395.3
            vis_end_original = 1019.61
            vis_total_channels_original = 785
            vis_step = (vis_end_original - vis_start) / (vis_total_channels_original - 1)

            # ---- SWIR parameters ----
            swir_start = 1098.03
            swir_end_original = 2577.88
            swir_total_channels_original = 225
            swir_step = (swir_end_original - swir_start) / (swir_total_channels_original - 1)

            # ---- Extract columns ----
            vis_columns = [col for col in columns if col.startswith('val_vis_')]
            swir_columns = [col for col in columns if col.startswith('val_swir_')]

            vis_channels = len(vis_columns)
            swir_channels = len(swir_columns)

            # ---- Ticks ----
            ideal_num_ticks = 10
            vis_ticks = np.linspace(0, vis_channels - 1, ideal_num_ticks).round().astype(int).tolist()
            swir_ticks = np.linspace(0, swir_channels - 1, ideal_num_ticks).round().astype(int).tolist()

            # ---- Iterate over columns ----
            for i, col in enumerate(columns):
                if col.startswith('val_vis_'):
                    vis_idx = vis_columns.index(col)
                    channel = int(col.split('_')[-1])  # original channel
                    wl = vis_start + channel * vis_step
                    wavelengths.append(wl)

                    if vis_idx in vis_ticks:
                        xticks_positions.append(i)
                        xticks_labels.append(f"vis {channel} ({wl:.1f} nm)")

                elif col.startswith('val_swir_'):
                    swir_idx = swir_columns.index(col)
                    channel = int(col.split('_')[-1])  # original channel
                    wl = swir_start + channel * swir_step
                    wavelengths.append(wl)

                    if swir_idx in swir_ticks:
                        xticks_positions.append(i)
                        xticks_labels.append(f"SWIR {channel} ({wl:.1f} nm)")

            return wavelengths, xticks_positions, xticks_labels

    def compare_spectra(self, per_file=True, num_blocks=6):
            df_merged = self.dataframe()
            spectra_columns = [col for col in df_merged.columns if col.startswith('val_')]

            wavelengths, xticks_pos, xticks_labels = self._get_wavelengths_and_labels(spectra_columns)

            if per_file:
                for file in df_merged['File'].unique():
                    df_file = df_merged[df_merged['File'] == file]
                    spectra_array = df_file[spectra_columns].values

                    if len(spectra_array) < num_blocks:
                        print(f"{file}: there are not enough files to divide into {num_blocks} blocks.")
                        continue

                    block_size = len(spectra_array) // num_blocks

                    fig, ax = plt.subplots(figsize=(14, 6))

                    for i in range(num_blocks):
                        start = i * block_size
                        end = (i + 1) * block_size if i < num_blocks - 1 else len(spectra_array)
                        block = spectra_array[start:end]
                        moy = np.mean(block, axis=0)
                        min_ = np.min(block, axis=0)
                        max_ = np.max(block, axis=0)
                        ax.plot(wavelengths, moy, label=f"Spectrum {i+1}")
                        ax.fill_between(wavelengths, min_, max_, alpha=0.2)

                    ax.set_title(f"Spectra per block - {file}")
                    ax.set_xlabel("Data type, channel, and wavelength (nm)")
                    ax.set_ylabel("Reflectance (%)")
                    ax.set_xticks([wavelengths[i] for i in xticks_pos])
                    xticklabels = ax.set_xticklabels(xticks_labels, rotation=90, fontsize=8)

                    # Force a draw to measure xticklabels
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    max_height = max([label.get_window_extent(renderer).height for label in xticklabels])
                    fig_height = fig.get_size_inches()[1] * fig.dpi
                    bottom_space = max_height / fig_height + 0.05  # extra margin

                    # Adjust bottom margin
                    fig.subplots_adjust(bottom=bottom_space)

                    # Place legend just below
                    # Move legend outside the axis
                    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)

                    # Adjust layout
                    fig.tight_layout(rect=[0, 0.1, 1, 1])  # leave space below (0.1)

                    plt.show()

            else:
                fig, ax = plt.subplots(figsize=(14, 6))

                for file in df_merged['File'].unique():
                    df_file = df_merged[df_merged['File'] == file]
                    spectra_array = df_file[spectra_columns].values
                    if len(spectra_array) == 0:
                        continue
                    moy = np.mean(spectra_array, axis=0)
                    min_ = np.min(spectra_array, axis=0)
                    max_ = np.max(spectra_array, axis=0)
                    ax.plot(wavelengths, moy, label=f"{file} (n={len(spectra_array)})")
                    ax.fill_between(wavelengths, min_, max_, alpha=0.2)

                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Reflectance (%)")
                ax.set_title("Average spectra per pigment")
                ax.set_xticks([wavelengths[i] for i in xticks_pos])
                xticklabels = ax.set_xticklabels(xticks_labels, rotation=90, fontsize=8)

                # Force a draw to measure xticklabels
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                max_height = max([label.get_window_extent(renderer).height for label in xticklabels])
                fig_height = fig.get_size_inches()[1] * fig.dpi
                bottom_space = max_height / fig_height + 0.05  # extra margin

                # Adjust bottom margin
                fig.subplots_adjust(left=0.12, bottom=bottom_space)

                # Place legend right below
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_space / 2), ncol=3)

                plt.tight_layout()
                plt.show()

    def PCA_analysis(self, df, n_components=4, titre="Principal Component Analysis"):
            val_columns = [col for col in df.columns if col.startswith('val_')]
            
            if not val_columns:
                print("There are no columns with 'val_'")
                return None

            X = df[val_columns]
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)

            # Create a DataFrame with principal components
            df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
            df_pca['File'] = df['File'].values if 'File' in df.columns else range(len(df))

            # Plot
            plt.figure(figsize=(10, 10))
            cmap = plt.cm.get_cmap('tab20', len(df_pca['File'].unique()))
            for i, fichier in enumerate(df_pca['File'].unique()):
                subset = df_pca[df_pca['File'] == fichier]
                plt.scatter(subset['PC1'], subset['PC2'], label=fichier, color=cmap(i % 20))
            
            plt.title(titre, fontweight='bold', fontsize=12)
            plt.xlabel('PC1', fontweight='bold', fontsize=12)
            plt.ylabel('PC2', fontweight='bold', fontsize=12)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), title="File", ncol=3)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            return df_pca

