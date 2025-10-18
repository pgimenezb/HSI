from hsi_lab.data.config import variables
import os, random, numpy as np, pandas as pd, h5py, tensorflow as tf

class HSIDataProcessor:

    SEED = 42

    def __init__(self, variables):
        self.set_global_seed()
        self.data_folder = variables['data_folder']
        self.excel_file  = variables['excel_file']
        self.file_names = pd.read_csv(self.excel_file) if self.excel_file else None
        print(f"active data_folder: {self.data_folder}")
        print(f"active excel_file: {self.excel_file}")
        self.all_data = {}
        self.loaded_files_range = (0, 0)
        self.binary_representations = {}
        self.Pigment_nom = None
        self.start_index = variables['start_index']
        self.num_files = variables['num_files']                  # nº de pigmentos (bits)
        self.selected_regions = variables['selected_regions']    # lista de ints o []
        self.selected_subregions = variables['selected_subregions']
        self.data_type = variables['data_type']
        self.points = [(21, 9), (81, 9), (81, 32), (47, 47), (20, 60), (60, 68), (85, 75), (20, 90), (90, 93)]
        self.savgol_window = variables['savgol_window']
        self.savgol_polyorder = variables['savgol_polyorder']
        self.num_mixture = variables['num_mixture']
        self.mixture_columns = variables['mixture_columns']
        self.all_pigment_columns = pd.read_csv(self.excel_file)['Title'].tolist()
        self.meta_label_map = variables['meta_label_map']
        self.mixture_mapping = variables['mixture_mapping']      # dict: bits -> nombre
        self.pigments_mapping = {}
        self._multi_to_label = None
        # copia limpia de algunos dicts
        self.variables = variables.copy()
        self.variables["mixture_columns"] = {
            int(k): v for k, v in self.variables.get("mixture_columns", {}).items()
        }
        self.mixture_columns = self.variables["mixture_columns"]

    @staticmethod
    def detect_region(v, num_pigments: int, binder_len: int = 2, mixture_len: int = 4) -> int:
        """Devuelve 1..8 si el vector es completo (binder+mixture) o 1..4 si es reducido (solo mixture)."""
        L = len(v)
        if L == num_pigments + binder_len + mixture_len:
            b0 = int(v[num_pigments + 0] == 1)
            b1 = int(v[num_pigments + 1] == 1)
            m_idx = next((i for i in range(mixture_len)
                          if v[num_pigments + binder_len + i] == 1), None)
            if m_idx is None:
                return 0
            if   (b0, b1) == (1, 0): b_idx = 0
            elif (b0, b1) == (0, 1): b_idx = 1
            else:                    return 0
            return m_idx * 2 + b_idx + 1  # 1..8
        if L == num_pigments + mixture_len:
            m_idx = next((i for i in range(mixture_len) if v[num_pigments + i] == 1), None)
            return (m_idx + 1) if m_idx is not None else 0  # 1..4
        return 0

    def set_global_seed(self):
        seed = self.SEED
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # =============================
    # Carga .h5  (binder se ignora)
    # =============================
    def load_h5_files(self):
        self.loaded_files_range = (self.start_index, self.start_index + self.num_files)
        self.all_data = {}

        for fichier in self.file_names.Title[self.start_index:self.start_index + self.num_files]:
            file_path = os.path.join(self.data_folder, fichier)
            try:
                h5_file = h5py.File(file_path, 'r')

                # Cargamos arrays originales
                binder     = h5_file['labels/labels_binder'][()]          # no usado
                mixture    = h5_file['labels/labels_mixture'][()]
                multilabel = h5_file['labels/vector_multilabel'][()]

                h5_filtered = {
                    'labels/labels_binder': binder,       # por compatibilidad
                    'labels/labels_mixture': mixture,
                    'labels/vector_multilabel': multilabel
                }
                h5_file.filtered_labels = h5_filtered
                self.all_data[fichier] = h5_file

            except Exception as e:
                print(f"Error loading {fichier}: {e}")

        self.file_names = self.file_names[self.file_names['Title'].isin(self.all_data.keys())]

        print(f"{len(self.all_data)} loaded files.")
        if self.all_data:
            last_key = list(self.all_data.keys())[-1]
            print(f"Last file loaded: {last_key}")
        else:
            print("No valid files loaded.")
        return self.all_data

    # =============================
    # DataFrame (1 fila por píxel)
    # =============================
    def dataframe(self):
        import numpy as np
        import pandas as pd

        def _first_one_idx(bitstr: str) -> int:
            idxs = [i for i, c in enumerate(bitstr) if c == '1']
            if len(idxs) != 1:
                raise ValueError(f"Bitstring inválido (un solo '1'): {bitstr}")
            return idxs[0]

        multilabel_rows = []

        NUM_PIGMENTS = int(getattr(self, "num_pigments", self.num_files))

        # --- SOLO mixture es obligatorio ---
        if not getattr(self, "mixture_mapping", None):
            raise ValueError("self.mixture_mapping vacío o no definido")

        # Mapeos de mixture
        mixture_bits = list(self.mixture_mapping.keys())             # ['1000','0100','0010','0001']
        mixture_len = len(mixture_bits[0])
        if not all(len(b) == mixture_len for b in mixture_bits):
            raise ValueError("Todos los bitstrings de mixture deben tener la misma longitud")

        # Orden 1000,0100,0010,0001
        mixture_bits_ordered   = sorted(mixture_bits, key=_first_one_idx)
        mixture_names_ordered  = [self.mixture_mapping[b] for b in mixture_bits_ordered]  # nombres en ese orden

        # nombre -> offset (0..3)
        mixture_pos_by_name = { self.mixture_mapping[b]: _first_one_idx(b) for b in mixture_bits }

        # Bloques (sin binder)
        MIX_BASE  = NUM_PIGMENTS
        TOTAL_LEN = NUM_PIGMENTS + mixture_len

        num_mixture = getattr(self, "num_mixture", {})

        # Filtros opcionales desde config
        sel_regions  = set(getattr(self, "selected_regions", []) or [])
        sel_subregs  = set(getattr(self, "selected_subregions", []) or [])

        # =======================
        #      Recorrido H5
        # =======================
        for file_name, h5_file in self.all_data.items():
            data = h5_file.filtered_labels["labels/vector_multilabel"]  # (H, W, D_in)
            H, W, D_in = data.shape

            # Necesario para "coger la mezcla que toca"
            try:
                mixture_map = h5_file.filtered_labels["labels/labels_mixture"]
                assert mixture_map.shape == (H, W)
            except Exception:
                # Sin mixture_map no podemos saber la mezcla real del píxel
                continue

            # Rangos verticales por mezcla (para subregión)
            mixture_ranges = {}
            for mnum in np.unique(mixture_map):
                if mnum == 0:
                    continue
                rows = np.any(mixture_map == mnum, axis=1)
                ys = np.where(rows)[0]
                if ys.size:
                    mixture_ranges[int(mnum)] = (int(ys[0]), int(ys[-1]) + 1)

            # píxeles con alguna etiqueta activa
            mask = (mixture_map > 0)
            ys_idx, xs_idx = np.nonzero(mask)
            if len(ys_idx) == 0:
                continue

            for y, x in zip(ys_idx, xs_idx):
                v = data[y, x, :]

                # 1) Pigmento
                pigment_idx = int(np.argmax(v[:NUM_PIGMENTS]))

                # 2) Mezcla REAL del píxel (1..4). Si 0 o fuera de rango, saltar
                mnum_px = int(mixture_map[y, x])
                if mnum_px <= 0 or mnum_px > 4:
                    continue

                m_name = mixture_names_ordered[mnum_px - 1]  # nombre de mezcla
                m_off  = mixture_pos_by_name[m_name]         # offset 0..3

                # 3) Región = índice de mezcla (1..4)
                region = mnum_px

                # 4) Subregión por cuadrantes (sin binder)
                if mnum_px in mixture_ranges:
                    y0, y1 = mixture_ranges[mnum_px]
                    ymid = (y0 + y1) / 2.0
                else:
                    ymid = H / 2.0
                xmid = W / 2.0
                if x < xmid and y >= ymid:   subregion = 1
                elif x >= xmid and y >= ymid: subregion = 2
                elif x < xmid and y < ymid:   subregion = 3
                else:                          subregion = 4

                # 5) Filtros opcionales
                if sel_regions and region not in sel_regions:
                    continue
                if sel_subregs and subregion not in sel_subregs:
                    continue

                # 6) Construir SOLO UNA fila por píxel (mezcla real)
                full_vector = [0] * TOTAL_LEN
                full_vector[pigment_idx] = 1                 # pigmento
                full_vector[MIX_BASE + m_off] = 1            # mixture real

                multilabel_rows.append({
                    "File": str(file_name).strip(),
                    "X": int(x),
                    "Y": int(y),
                    "Pigment Index": int(pigment_idx),
                    "Mixture": m_name,
                    "Num. Mixture": int(num_mixture.get(m_name, 0)),
                    "Region": int(region),        # 1..4
                    "Subregion": int(subregion),  # 1..4
                    "Multi": full_vector,         # [pigments] + [4 mixture]
                })

        df_multilabel = pd.DataFrame(multilabel_rows)

        # ====== cubos espectrales (sin cambios) ======
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
                ["File", "Region", "Mixture", "Y", "X"], kind="mergesort"
            ).reset_index(drop=True)
            df_multilabel["Spectrum"] = (df_multilabel.groupby("File").cumcount() + 1)\
                .astype(str).radd("Spectrum_")

        if not df_multilabel.empty and df_cubes is not None and not df_cubes.empty:
            df_merged = df_multilabel.merge(df_cubes, on=["File", "X", "Y"], how="left")
        else:
            df_merged = df_cubes if (df_cubes is not None and not df_cubes.empty) else df_multilabel

        return df_merged
