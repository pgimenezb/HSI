from hsi_lab.data.config import variables
import os, random, re, numpy as np, pandas as pd, h5py, tensorflow as tf

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

        # Config y metadatos
        self.start_index = variables['start_index']
        self.num_files = variables['num_files']                  # = nº de pigmentos activos
        self.selected_regions = variables['selected_regions']
        self.selected_subregions = variables['selected_subregions']
        self.data_type = variables['data_type']
        self.savgol_window = variables['savgol_window']
        self.savgol_polyorder = variables['savgol_polyorder']
        self.num_mixture = variables['num_mixture']
        self.mixture_columns = variables['mixture_columns']
        self.meta_label_map = variables['meta_label_map']
        self.mixture_mapping = variables['mixture_mapping']
        self.variables = variables.copy()

        # Normaliza llaves a int para mixture_columns
        self.variables["mixture_columns"] = {
            int(k): v for k, v in self.variables.get("mixture_columns", {}).items()
        }
        self.mixture_columns = self.variables["mixture_columns"]

        # Control de cuotas/balanceo
        self.region_row_quota    = self.variables.get("region_row_quota", {}) or {}
        self.subregion_row_quota = self.variables.get("subregion_row_quota", {}) or {}
        self.balance_seed        = int(self.variables.get("balance_seed", 42))

    # =============================
    # Utilidades internas
    # =============================
    @staticmethod
    def _first_one_idx(bitstr: str) -> int:
        idxs = [i for i, c in enumerate(bitstr) if c == '1']
        if len(idxs) != 1:
            raise ValueError(f"Bitstring inválido (un solo '1'): {bitstr}")
        return idxs[0]

    @staticmethod
    def _extract_first_int(s: str) -> int:
        """Extrae el primer número entero que aparezca en s (o -1 si no hay)."""
        m = re.search(r"\d+", str(s))
        return int(m.group(0)) if m else -1

    def _balanced_quota_sample(self, df_grp: pd.DataFrame, total: int) -> pd.DataFrame:
        """
        Devuelve 'total' filas de df_grp (grupo por Región o Subregión),
        balanceando por (Pigment Index × Mixture). Determinista con balance_seed.
        Si total <= 0 o el grupo es más pequeño, devuelve el grupo sin tocar.
        """
        if total <= 0 or len(df_grp) <= total:
            return df_grp

        rng = np.random.default_rng(self.balance_seed)
        strata = df_grp.groupby(["Pigment Index", "Mixture"], sort=False)
        groups = [(k, g.index.to_numpy()) for k, g in strata]
        S = len(groups)
        if S == 0:
            return df_grp.iloc[0:0]

        base = total // S
        rem  = total % S
        order = np.arange(S); rng.shuffle(order)
        take = np.full(S, base, dtype=int); take[order[:rem]] += 1

        chosen = []
        for (_, idxs), k_take in zip(groups, take):
            if k_take <= 0:
                continue
            if len(idxs) <= k_take:
                chosen.append(idxs)
            else:
                chosen.append(rng.choice(idxs, size=k_take, replace=False))

        if not chosen:
            return df_grp.iloc[0:0]
        sel = np.concatenate(chosen); rng.shuffle(sel)
        return df_grp.loc[sel]

    def _balanced_quota_sample_by(self, df_grp: pd.DataFrame, total: int, by_cols) -> pd.DataFrame:
        """
        Versión general: balancea por columnas dadas (p.ej., ['Mixture'] o ['Mixture','Region']).
        """
        if total <= 0 or len(df_grp) <= total:
            return df_grp

        rng = np.random.default_rng(self.balance_seed)
        strata = df_grp.groupby(by_cols, sort=False)
        groups = [(k, g.index.to_numpy()) for k, g in strata]
        S = len(groups)
        if S == 0:
            return df_grp.iloc[0:0]

        base = total // S
        rem  = total % S
        order = np.arange(S); rng.shuffle(order)
        take = np.full(S, base, dtype=int); take[order[:rem]] += 1

        chosen = []
        for (_, idxs), k_take in zip(groups, take):
            if k_take <= 0:
                continue
            if len(idxs) <= k_take:
                chosen.append(idxs)
            else:
                chosen.append(rng.choice(idxs, size=k_take, replace=False))

        if not chosen:
            return df_grp.iloc[0:0]
        sel = np.concatenate(chosen); rng.shuffle(sel)
        return df_grp.loc[sel]

    def _equalize_r234_to_r1_by_pigment(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Para cada pigmento p, fuerza que (# filas en Regiones 2+3+4) == (# filas en Región 1).
        Se hace por *downsampling* balanceado:
          - Reg1: muestreo simple (solo hay una mezcla real en Reg1).
          - Reg(2,3,4): balanceando por ['Mixture','Region'].
        """
        if df_in.empty:
            return df_in

        out_parts = []
        for p in sorted(df_in["Pigment Index"].unique()):
            dfp = df_in[df_in["Pigment Index"] == p]
            r1   = dfp[dfp["Region"] == 1]
            r234 = dfp[dfp["Region"].isin([2, 3, 4])]

            n1, n234 = len(r1), len(r234)
            if n1 == n234:
                out_parts.append(dfp)
                continue

            target = min(n1, n234)

            # Downsample ambos lados al mismo objetivo
            r1_sel   = self._balanced_quota_sample_by(r1,   target, by_cols=["Mixture"])           if n1   > target else r1
            r234_sel = self._balanced_quota_sample_by(r234, target, by_cols=["Mixture","Region"])  if n234 > target else r234

            # Si hay filas en otras regiones (no 1-4), las conservamos tal cual
            others = dfp[~dfp["Region"].isin([1, 2, 3, 4])]
            out_parts.append(pd.concat([r1_sel, r234_sel, others], ignore_index=True))

        out = pd.concat(out_parts, ignore_index=True)
        return out

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

                h5_file.filtered_labels = {
                    'labels/labels_binder': binder,       # compat
                    'labels/labels_mixture': mixture,
                    'labels/vector_multilabel': multilabel
                }
                self.all_data[fichier] = h5_file

            except Exception as e:
                print(f"Error loading {fichier}: {e}")

        self.file_names = self.file_names[self.file_names['Title'].isin(self.all_data.keys())]

        print(f"{len(self.all_data)} loaded files.")
        if self.all_data:
            print(f"Last file loaded: {list(self.all_data.keys())[-1]}")
        else:
            print("No valid files loaded.")
        return self.all_data


    # =============================
    # DataFrame (1 fila por píxel)
    # =============================
    def dataframe(self, mode: str = "filtered", debug_csv_dir: str | None = None):
        """
        mode:
        - "raw": sin muestreo/filtros/cuotas/igualado.
        - "filtered": comportamiento clásico.
        Si 'debug_csv_dir' se pasa, guarda CSVs de diagnóstico por etapas.
        """
        import numpy as np, pandas as pd, os

        RAW = (mode == "raw")
        multilabel_rows = []

        # 🔎 debug: acumuladores por etapa
        dbg_stage1_mask = []        # por File: pixels con mixture_map>0 (antes de muestrear)
        dbg_stage2_after_sample = []# por File: nº de coords tras sample_every
        dbg_stage3_after_filters = [] # por File: nº filas que pasarían filtros region/subregion
        dbg_files_seen = set()

        NUM_PIGMENTS = int(getattr(self, "num_pigments", self.num_files))
        if not getattr(self, "mixture_mapping", None):
            raise ValueError("self.mixture_mapping vacío o no definido")

        # --- Mapeo mixtures como antes ---
        mixture_bits = list(self.mixture_mapping.keys())
        mixture_len = len(mixture_bits[0])
        if not all(len(b) == mixture_len for b in mixture_bits):
            raise ValueError("Todos los bitstrings de mixture deben tener la misma longitud")
        mixture_bits_ordered   = sorted(mixture_bits, key=self._first_one_idx)
        mixture_names_ordered  = [self.mixture_mapping[b] for b in mixture_bits_ordered]
        mixture_pos_by_name    = { self.mixture_mapping[b]: self._first_one_idx(b) for b in mixture_bits }

        MIX_BASE  = NUM_PIGMENTS
        TOTAL_LEN = NUM_PIGMENTS + mixture_len
        num_mixture = getattr(self, "num_mixture", {})

        # Filtros opcionales desde config (se desactivan en modo RAW)
        sel_regions = set([] if RAW else (getattr(self, "selected_regions", []) or []))
        sel_subregs = set([] if RAW else (getattr(self, "selected_subregions", []) or []))

        # -----------------------
        # Recorrer ficheros H5
        # -----------------------
        for file_name, h5_file in self.all_data.items():
            dbg_files_seen.add(str(file_name).strip())

            data = h5_file.filtered_labels["labels/vector_multilabel"]  # (H, W, D_in)
            H, W, D_in = data.shape

            try:
                mixture_map = h5_file.filtered_labels["labels/labels_mixture"]
                assert mixture_map.shape == (H, W)
            except Exception:
                continue

            # === MASK de píxeles válidos: solo >0 ===
            mask = (mixture_map > 0)
            ys_idx_all, xs_idx_all = np.nonzero(mask)

            # 🔎 debug stage1: tamaño del mask
            dbg_stage1_mask.append({"File": str(file_name).strip(), "mask_pixels_gt0": int(mask.sum())})

            if len(ys_idx_all) == 0:
                continue

            # === Muestreo ambiental ===
            sample_every = 0 if RAW else int(os.getenv("HSI_SAMPLE_EVERY", "0"))
            if sample_every > 1:
                ys_idx = ys_idx_all[::sample_every]
                xs_idx = xs_idx_all[::sample_every]
            else:
                ys_idx, xs_idx = ys_idx_all, xs_idx_all

            # 🔎 debug stage2: coords tras muestreo (potencial recorte global)
            dbg_stage2_after_sample.append({"File": str(file_name).strip(), "coords_after_sampling": int(len(ys_idx))})

            # === Subregiones (como ya tienes) ===
            mixture_ranges = {}
            for mnum in np.unique(mixture_map):
                if mnum == 0: continue
                rows = np.any(mixture_map == mnum, axis=1)
                ys = np.where(rows)[0]
                if ys.size:
                    mixture_ranges[int(mnum)] = (int(ys[0]), int(ys[-1]) + 1)

            xmid, ymid_default = W / 2.0, H / 2.0

            # 🔎 debug: contador provisional antes de filtros región/subregión
            prefilter_count = 0
            postfilter_count = 0

            # --- loop de píxeles válidos ---
            for y, x in zip(ys_idx, xs_idx):
                v = data[y, x, :]

                pigment_idx = int(np.argmax(v[:NUM_PIGMENTS]))

                mnum_px = int(mixture_map[y, x])
                if mnum_px <= 0 or mnum_px > mixture_len:
                    continue

                prefilter_count += 1

                m_name = mixture_names_ordered[mnum_px - 1]
                m_off  = mixture_pos_by_name[m_name]
                region = mnum_px

                # Subregión por cuadrantes relativos a su mezcla
                if mnum_px in mixture_ranges:
                    y0, y1 = mixture_ranges[mnum_px]; ymid = (y0 + y1) / 2.0
                else:
                    ymid = ymid_default
                if x < xmid and y >= ymid:   subregion = 1
                elif x >= xmid and y >= ymid: subregion = 2
                elif x < xmid and y < ymid:   subregion = 3
                else:                          subregion = 4

                # === Filtros de región/subregión (se omiten en RAW) ===
                if sel_regions and region not in sel_regions:
                    continue
                if sel_subregs and subregion not in sel_subregs:
                    continue

                postfilter_count += 1

                full_vector = [0] * TOTAL_LEN
                full_vector[pigment_idx] = 1
                full_vector[MIX_BASE + m_off] = 1

                multilabel_rows.append({
                    "File": str(file_name).strip(),
                    "X": int(x), "Y": int(y),
                    "Pigment Index": int(pigment_idx),
                    "Mixture": m_name,
                    "Num. Mixture": int(num_mixture.get(m_name, 0)),
                    "Region": int(region), "Subregion": int(subregion),
                    "Multi": full_vector,
                })

            # 🔎 debug stage3: tras filtros (solo aplica si no RAW)
            dbg_stage3_after_filters.append({
                "File": str(file_name).strip(),
                "prefilter_candidates": int(prefilter_count),
                "after_region_filters": int(postfilter_count)
            })

        df_multilabel = pd.DataFrame(multilabel_rows)

        # 🔎 debug: volcados intermedios si se pide
        if debug_csv_dir:
            os.makedirs(debug_csv_dir, exist_ok=True)
            pd.DataFrame(dbg_stage1_mask).sort_values("File").to_csv(
                os.path.join(debug_csv_dir, "stage1_mask_pixels_gt0.csv"), index=False)
            pd.DataFrame(dbg_stage2_after_sample).sort_values("File").to_csv(
                os.path.join(debug_csv_dir, "stage2_coords_after_sampling.csv"), index=False)
            pd.DataFrame(dbg_stage3_after_filters).sort_values("File").to_csv(
                os.path.join(debug_csv_dir, "stage3_after_region_filters.csv"), index=False)


        # ====== cubos espectrales ======
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
                dfs.append(pd.concat([base, vals], axis=1))
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        cube_types = self.data_type
        if isinstance(cube_types, str):
            cube_types = [cube_types]
        elif not isinstance(cube_types, (list, tuple)):
            cube_types = [str(cube_types)]

        df_cubes = None
        for t in cube_types:
            df_t = build_cube_df(t)
            if df_t.empty:
                continue
            cols = ["File", "X", "Y"] + [c for c in df_t.columns if c.startswith(f"{t}_")]
            df_t = df_t[cols]
            df_cubes = df_t if df_cubes is None else df_cubes.merge(df_t, on=["File", "X", "Y"], how="inner")

        # Merge multilabel + cubos (LEFT para no perder filas etiquetadas)
        if not df_multilabel.empty and df_cubes is not None and not df_cubes.empty:
            df_merged = df_multilabel.merge(df_cubes, on=["File", "X", "Y"], how="left")
        else:
            df_merged = df_cubes if (df_cubes is not None and not df_cubes.empty) else df_multilabel

        # 🔎 debug: resúmenes por File/Region/Subregion/Mixture del multilabel y del final
        if debug_csv_dir:
            # por File (multilabel puro)
            (df_multilabel.groupby("File").size()
                .rename("rows").reset_index()
                .to_csv(os.path.join(debug_csv_dir, "stage4_multilabel_rows_by_file.csv"), index=False))
            # por File/Region/Mixture/Subregion
            for cols, name in [
                (["File","Region"], "by_File_Region"),
                (["File","Subregion"], "by_File_Subregion"),
                (["File","Mixture"], "by_File_Mixture"),
                (["File","Pigment Index","Mixture"], "by_File_Pigment_Mixture"),
            ]:
                (df_multilabel.groupby(cols).size()
                    .rename("rows").reset_index()
                    .to_csv(os.path.join(debug_csv_dir, f"stage4_multilabel_{name}.csv"), index=False))
            # final tras merge
            if df_merged is not None and not df_merged.empty:
                (df_merged.groupby("File").size()
                    .rename("rows").reset_index()
                    .to_csv(os.path.join(debug_csv_dir, "stage5_final_df_merged_by_file.csv"), index=False))


        # === EN RAW NO aplicamos cuotas ni igualación ===
        if RAW:
            return df_merged

        # === (modo 'filtered') Cuotas ===
        df_out = df_merged
        if self.subregion_row_quota:
            parts = []
            for s, sub_df in df_out.groupby("Subregion", sort=False):
                target = int(self.subregion_row_quota.get(int(s), 0))
                parts.append(self._balanced_quota_sample(sub_df, target) if target > 0 else sub_df)
            df_out = pd.concat(parts, ignore_index=True)
        elif self.region_row_quota:
            parts = []
            for r, reg_df in df_out.groupby("Region", sort=False):
                target = int(self.region_row_quota.get(int(r), 0))
                parts.append(self._balanced_quota_sample(reg_df, target) if target > 0 else reg_df)
            df_out = pd.concat(parts, ignore_index=True)

        # === Igualar por pigmento: sum(Reg 2+3+4) == Reg 1 ===
        if df_out is not None and not df_out.empty:
            df_out = self._equalize_r234_to_r1_by_pigment(df_out)

        # Orden final
        if df_out is not None and not df_out.empty:
            df_out = df_out.assign(
                _FileNum = df_out["File"].map(self._extract_first_int)
            ).sort_values(
                ["_FileNum", "Region", "Subregion", "Mixture", "Y", "X"],
                kind="mergesort"
            ).drop(columns=["_FileNum"]).reset_index(drop=True)

        return df_out



    def save_rows_for_file(self, target_file: str, out_csv_path: str):
        df = self.dataframe()
        df_t = df[df["File"] == target_file]
        if df_t.empty:
            print(f"[WARN] No hay filas para File == {target_file}")
            return None
        # Asegura carpeta
        os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
        df_t.to_csv(out_csv_path, index=False)
        print(f"[OK] Guardado {len(df_t)} filas en: {out_csv_path}")
        return out_csv_path







