

# train.py
import os
import argparse
import importlib
import numpy as np
import pandas as pd

from hsi_lab.data.config import variables
from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.eval.report import save_confusion_pngs, evaluation_metrics, export_global_table, plot_confusion_global
from sklearn.model_selection import StratifiedShuffleSplit

# =========================
# Utilidades
# =========================
def ensure_outputs_dir(arg_out: str | None, cfg_out: str | None) -> str:
    out_dir = arg_out or cfg_out or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def parse_limit_map(s: str | None):
    if not s:
        return {}
    out = {}
    for item in s.split(","):
        k, v = item.split("=")
        out[str(k).strip()] = int(v.strip())
    return out

def log_df_info(tag: str, dfx: pd.DataFrame):
    print(f"[DATA] {tag}: shape={dfx.shape}")
    if "Region" in dfx.columns:
        try:
            vc = dfx["Region"].value_counts(dropna=False).sort_index()
            print(f"[DATA] {tag}: filas por Region ->")
            for k, v in vc.items():
                print(f"  - Region {k}: {v}")
        except Exception as e:
            print(f"[DATA] {tag}: no se pudo listar Region ({e})")
    if "Multi" in dfx.columns:
        try:
            lens = dfx["Multi"].map(len)
            print(f"[DATA] {tag}: len(Multi): min={lens.min()}, max={lens.max()}, únicos={sorted(set(lens))}")
        except Exception:
            pass

def load_model_module(name: str):
    mod_name = name if name.startswith("hsi_lab.models.") else f"hsi_lab.models.{name}"
    try:
        return importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        raise ValueError(f"Non supported model: {name}") from e

# --- sampler balanceado por pigmento dentro de cada región ---
def sample_by_region_balanced(df, region_col, limit_map, balance_col="Pigment Index",
                              num_files=None, seed=42):
    """
    Balancea por pigmento dentro de cada región.
    Si num_files se pasa (p.ej. 4), balancea entre pigmentos 0..num_files-1.
    """
    parts = []

    target_pigments = set(range(int(num_files))) if num_files is not None else None

    for region, subdf in df.groupby(region_col, dropna=False):
        lim = limit_map.get(region, None)
        if lim is None or lim >= len(subdf):
            parts.append(subdf)
            continue

        # filtra a pigmentos objetivo si se especifica
        if target_pigments is not None:
            subdf = subdf[subdf[balance_col].isin(target_pigments)]
            if subdf.empty:
                continue

        # ids de pigmento a repartir
        pigment_ids = (list(subdf[balance_col].dropna().unique())
                       if target_pigments is None else list(sorted(target_pigments)))

        k = max(1, len(pigment_ids))
        base = lim // k

        chosen = []
        taken_idx = set()

        # 1) cupo base por pigmento
        for pid in pigment_ids:
            g = subdf[subdf[balance_col] == pid]
            n = min(base, len(g))
            if n > 0:
                pick = g.sample(n=n, random_state=seed)
                chosen.append(pick)
                taken_idx.update(pick.index.tolist())

        # 2) completa hasta 'lim' con el remanente (si faltan)
        need = lim - sum(len(c) for c in chosen)
        if need > 0:
            remaining = subdf.drop(index=list(taken_idx)) if taken_idx else subdf
            if not remaining.empty:
                extra_n = min(need, len(remaining))
                chosen.append(remaining.sample(n=extra_n, random_state=seed))

        if chosen:
            parts.append(pd.concat(chosen, axis=0))

    return pd.concat(parts, axis=0, ignore_index=True) if parts else df.head(0).copy()

def build_strat_labels(y: np.ndarray, n_pigments: int) -> np.ndarray:
    """
    y: (N, n_pigments+4) one-hot por bloques.
    Devuelve etiqueta entera para estratificar: pig*2 + is_mix (is_mix=0 si Pure, 1 si {Mixture1,2,3})
    """
    pig_idx = y[:, :n_pigments].argmax(axis=1)
    mix_idx = y[:, n_pigments:n_pigments+4].argmax(axis=1)
    is_mix = (mix_idx != 0).astype(int)
    return pig_idx * 2 + is_mix  # [0..(2*n_pigments-1)]

# =========================
# Orquestador
# =========================
def main():
    parser = argparse.ArgumentParser(description="Entrena modelos HSI especificados.")
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--group-by", type=str, default=None,
                        help="Columna para agrupar/limitar (ej: Region o Subregion).")
    parser.add_argument("--per-group-limit-map", type=str, default=None,
                        help="Límites por grupo, p.ej. '1=300,2=100,3=100,4=100'.")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--outputs-dir", type=str, default=None)
    parser.add_argument("--regions", type=str, default="", help="Filtro, p.ej. '1,2'.")
    parser.add_argument("--subregions", type=str, default="", help="Filtro, p.ej. '1,2,3,4'.")
    args = parser.parse_args()

    out_dir = ensure_outputs_dir(args.outputs_dir, variables.get("outputs_dir"))
    csv_full  = os.path.join(out_dir, "dataframe.csv")

    # ---- carga + dataframe principal ----
    processor = HSIDataProcessor(variables)
    processor.load_h5_files()
    df = processor.dataframe()
    log_df_info("DF (antes de filtros/limites)", df)

    # ---- filtros CLI opcionales ----
    if args.regions and "Region" in df.columns:
        keep = {int(x) for x in args.regions.split(",") if x.strip()}
        df = df[df["Region"].isin(keep)].copy()
    if args.subregions and "Subregion" in df.columns:
        keep = {int(x) for x in args.subregions.split(",") if x.strip()}
        df = df[df["Subregion"].isin(keep)].copy()

    # seguridad: solo 1..4 si existe Region
    if "Region" in df.columns:
        df = df[df["Region"].isin([1, 2, 3, 4])].copy()

    # ---- límites por grupo en DF principal (balanceado por pigmento) ----
    limit_map_raw = parse_limit_map(args.per_group_limit_map)
    target_group_col = args.group_by or ("Region" if limit_map_raw else None)

    if target_group_col and limit_map_raw:
        if target_group_col not in df.columns:
            raise ValueError(f"Columna '{target_group_col}' no existe en el DataFrame.")
        if np.issubdtype(df[target_group_col].dtype, np.integer):
            limit_map = {int(k): int(v) for k, v in limit_map_raw.items()}
        else:
            limit_map = {str(k): int(v) for k, v in limit_map_raw.items()}

        before = len(df)
        print(f"[LIMIT] group-by='{target_group_col}', limit_map={limit_map} "
              f"(dtype col={df[target_group_col].dtype})")

        df = sample_by_region_balanced(
            df,
            region_col=target_group_col,
            limit_map=limit_map,
            balance_col="Pigment Index",
            num_files=int(variables.get("num_files", 4)),   # ← balancea 0..num_files-1
            seed=int(variables.get("seed", 42)),
        )

        after = len(df)
        print(f"[INFO] Aplicado group-by='{target_group_col}' (balanceado por pigmento). Filas: {before} -> {after}")
    else:
        print("[INFO] Sin límites por grupo (no se pasó --per-group-limit-map).")

    # ---- log y guardado del DF FINAL ----
    log_df_info("DF (final, tras filtros/limites)", df)
    df.to_csv(csv_full, index=False)
    print(f"[OK] Guardado dataframe (final): {csv_full}")

    # ---- matrices X/y (sin normalizar) ----
    multi = np.array([np.array(label) for label in df["Multi"]], dtype=np.float32)
    num_classes = multi.shape[1]
    spec_cols = [c for c in df.columns if c.startswith(("vis_", "swir_"))]
    if not spec_cols:
        raise ValueError("No spectral columns found starting with 'vis_' or 'swir_'.")
    spec_cols = sorted(spec_cols, key=lambda c: (0 if c.startswith("vis_") else 1, c))
    X_mat = df[spec_cols].astype(np.float32).fillna(0.0).values  # (N, L)
    y_mat = multi                                                 # (N, C)
    input_len = X_mat.shape[1]
    num_files = int(variables["num_files"])

    # ---- etiquetas para estratificar por (pigmento, pure/mixture) ----
    y_strat = build_strat_labels(y_mat, n_pigments=num_files)

    # ---- split estratificado: 70% trainval / 30% test ----
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    trainval_idx, test_idx = next(sss1.split(X_mat, y_strat))
    X_trainval, X_test = X_mat[trainval_idx], X_mat[test_idx]
    y_trainval, y_test = y_mat[trainval_idx], y_mat[test_idx]
    y_strat_trainval   = y_strat[trainval_idx]

    # ---- split estratificado: 50/50 dentro de trainval => 35% train / 35% val ----
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    train_idx, val_idx = next(sss2.split(X_trainval, y_strat_trainval))
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    print(f"[DATA] Split estratificado -> "
          f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # en train.py, tras el split
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # añade canal
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    print(f"[DATA] Espectrales: #{input_len}  | X_train: {X_train.shape}  y_train: {y_train.shape}  num_classes: {num_classes}")

    # ---- modelos a entrenar ----
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    trained_models = {}

    for name in model_names:
        print(f"[TRAIN] {name}")
        mod = load_model_module(name)
        model, study, best = mod.tune_and_train(
            X_train, y_train, X_val, y_val,
            input_len=input_len,
            num_classes=num_classes,
            trials=args.trials,   # None -> usa variables['trials']
            epochs=args.epochs,   # None -> usa variables['epochs']
            n_jobs=variables.get("optuna_n_jobs", 1),
            seed=variables.get("seed", 42),
        )
        trained_models[name] = model

    # ---- evaluación: matrices, tablas y métricas ----
    pigment_names = [f"Pigment {i+1}" for i in range(num_files)]
    cm_dir = os.path.join(out_dir, "conf_mats")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    png_global = os.path.join(cm_dir, f"{name}_cm_global.png")
    plot_confusion_global(y_true=y_test, y_pred=y_pred_concat,
                      pigment_names=pigment_names, out_path=png_global,
                      annotate_min=5.0)  # anota sólo ≥5%

    for name, model in trained_models.items():
        # 1) Predicción
        y_pred_out = model.predict(X_test, verbose=0)

        # 2) Unifica a (N, num_files+4)
        if isinstance(y_pred_out, (list, tuple)):
            pig_prob, mix_prob = y_pred_out
            y_pred_concat = np.concatenate([pig_prob, mix_prob], axis=1)
        else:
            y_pred_concat = y_pred_out

        # 3) PNGs de matrices de confusión
        save_confusion_pngs(
            y_true=y_test,
            y_pred=y_pred_concat,
            pigment_names=pigment_names,
            out_dir=cm_dir,
            prefix=name
        )

        # 4) Tabla global (CSV)
        csv_table = os.path.join(tables_dir, f"{name}_global_table.csv")
        export_global_table(y_true=y_test,
                            y_pred=y_pred_concat,
                            pigment_names=pigment_names,
                            out_csv=csv_table,   # ← guarda CSV
                            threshold=0.5)

        # 5) Métricas (dict) + resumen CSV
        metrics = evaluation_metrics(
            y_true=y_test,
            y_pred=y_pred_concat,
            pigment_names=pigment_names,
            threshold=0.5,
            save_table_path=os.path.join(tables_dir, f"{name}_per_sample_scores.csv"),
            print_report=False
        )
        metrics_csv = os.path.join(tables_dir, f"{name}_metrics_summary.csv")
        metrics_light = {k: v for k, v in metrics.items()
                         if k not in ("table", "pigment_names", "mix_names")}
        pd.DataFrame([metrics_light]).to_csv(metrics_csv, index=False)

        print(f"[OK] Matrices de confusión -> {cm_dir}")
        print(f"[OK] Tabla global -> {csv_table}")
        print(f"[OK] Métricas resumen -> {metrics_csv}")

if __name__ == "__main__":
    main()


