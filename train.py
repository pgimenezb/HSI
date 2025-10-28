import os
import argparse
import importlib
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hsi_lab.data.config import variables
from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.eval import report as rep
from sklearn.metrics import confusion_matrix, accuracy_score
import csv



# ─────────────────────────────────────────────────────────────────────────────
# Carga dinámica del trainer
# ─────────────────────────────────────────────────────────────────────────────
def import_model_trainer(name: str):
    candidates = [name] if "." in name else []
    candidates.append(f"hsi_lab.models.{name}")
    last_err = None
    for mod_path in candidates:
        try:
            mod = importlib.import_module(mod_path)
            return getattr(mod, "tune_and_train")
        except (ModuleNotFoundError, AttributeError) as e:
            last_err = e
    raise ImportError(
        f"No se pudo cargar `tune_and_train` para '{name}'. Probé: {candidates}. "
        f"Último error: {last_err}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Features/targets
# ─────────────────────────────────────────────────────────────────────────────
def build_Xy(df: pd.DataFrame):
    spec_cols = [c for c in df.columns if c.startswith(("vis_", "swir_"))]
    if not spec_cols:
        raise ValueError("No spectral columns starting with 'vis_' or 'swir_'.")
    spec_cols = sorted(spec_cols, key=lambda c: (0 if c.startswith("vis_") else 1, c))
    X = df[spec_cols].astype(np.float32).fillna(0.0).values[..., np.newaxis]
    y = np.array([np.array(v) for v in df["Multi"]], dtype=np.float32)
    return X, y, X.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# Splits
# ─────────────────────────────────────────────────────────────────────────────
def pigment_ids(df: pd.DataFrame, vars_: dict) -> np.ndarray:
    n_p = int(vars_["num_files"])
    pig = []
    for v in df["Multi"]:
        a = np.asarray(v, dtype=np.float32)
        pig.append(int(np.argmax(a[:n_p])))
    return np.array(pig, dtype=int)

def stratified_balanced_split_by_file_pigment_mixture(df, vars_, per_mix=2, seed=42):
    rng = np.random.default_rng(seed)

    # Genera etiqueta combinada
    labels = df["File"].astype(str) + "_" + df["Pigment Index"].astype(str) + "_" + df["Mixture"].astype(str)
    grp = df.groupby(labels)

    train_idx, val_idx, test_idx = [], [], []

    for name, g in grp:
        idxs = g.index.to_numpy()
        rng.shuffle(idxs)

        if len(idxs) >= 4:
            n_test = per_mix
            n_val = max(1, int(0.15 * len(idxs)))
            n_train = len(idxs) - n_test - n_val
        else:
            n_test = min(per_mix, len(idxs)//2)
            n_val = 1 if len(idxs) > 2 else 0
            n_train = len(idxs) - n_test - n_val

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train+n_val])
        test_idx.extend(idxs[-n_test:])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def stratified_split_70_15_15(
    df: pd.DataFrame,
    vars_: dict,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_pig = pigment_ids(df, vars_)
    idx_all = np.arange(len(df))
    idx_train, idx_tmp = train_test_split(idx_all, test_size=0.30, random_state=seed, stratify=y_pig)
    y_tmp = y_pig[idx_tmp]
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    return np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test)


# ─────────────────────────────────────────────────────────────────────────────
# Decoders (argmax)
# ─────────────────────────────────────────────────────────────────────────────
def decode_pigment_and_group(y_like: np.ndarray, n_p: int):
    pig = np.argmax(y_like[:, :n_p], axis=1)
    mix_idx = np.argmax(y_like[:, n_p:n_p+4], axis=1)
    group = np.where(mix_idx == 0, "Pure", "Mixture")
    return np.array([f"P{p+1:02d}_{g}" for p, g in zip(pig, group)])

def decode_pigment_and_mix4(y_like: np.ndarray, n_p: int):
    pig = np.argmax(y_like[:, :n_p], axis=1)
    mix_idx = np.argmax(y_like[:, n_p:n_p+4], axis=1)
    names = np.array(["Pure", "M1", "M2", "M3"])
    return np.array([f"P{p+1:02d}_{names[m]}" for p, m in zip(pig, mix_idx)])

def decode_mix_group(y_like: np.ndarray, n_p: int):
    mix_idx = np.argmax(y_like[:, n_p:n_p+4], axis=1)
    return np.where(mix_idx == 0, "Pure", "Mixture")


# ─────────────────────────────────────────────────────────────────────────────
# Exportación de splits (SOLO train.csv, val.csv, test.csv)
# ─────────────────────────────────────────────────────────────────────────────
def export_splits_csv(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,   # puede venir con NaNs en train/val o incluso ser None
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    out_dir: str,
    vars_: dict,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    n_p = int(vars_["num_files"])
    N, D = y_true.shape

    # y_pred_prob puede ser None → creamos un array lleno de NaN del mismo tamaño
    if y_pred_prob is None:
        y_pred_prob = np.full((N, D), np.nan, dtype=np.float32)

    # --- Decodificación de y_true (completa) ---
    y_true_2N = decode_pigment_and_group(y_true, n_p)
    y_true_4N = decode_pigment_and_mix4(y_true, n_p)

    # --- Decodificación de y_pred (solo filas válidas; el resto queda vacío) ---
    def safe_decode(decoder_fn, y_like):
        out = np.array([""] * len(y_like), dtype=object)  # vacío por defecto
        valid = np.isfinite(y_like).all(axis=1)           # filas sin NaN/inf
        if valid.any():
            out[valid] = decoder_fn(y_like[valid], n_p)
        return out

    y_pred_2N = safe_decode(decode_pigment_and_group, y_pred_prob)
    y_pred_4N = safe_decode(decode_pigment_and_mix4,  y_pred_prob)

    def split_pm(lbl_4n: str):
        if not lbl_4n:  # vacío → devuelve vacíos
            return "", ""
        px, m = lbl_4n.split("_", 1)
        return px, m

    pig_true, mix_true = zip(*[split_pm(s) for s in y_true_4N])
    pig_pred, mix_pred = zip(*[split_pm(s) for s in y_pred_4N])

    # Construye un dataframe base con todas las filas y columnas calculadas
    base = df.copy()
    base = base.assign(
        y_true_2N = y_true_2N,
        y_pred_2N = y_pred_2N,
        y_true_4N = y_true_4N,
        y_pred_4N = y_pred_4N,
        pig_true  = np.array(pig_true, dtype=object),
        mix_true  = np.array(mix_true, dtype=object),
        pig_pred  = np.array(pig_pred, dtype=object),
        mix_pred  = np.array(mix_pred, dtype=object),
    )

    # Guardado de cada split (ordenado por File si existe)
    paths = {}
    def save_subset(indices: np.ndarray, name: str):
        d = base.iloc[np.sort(indices)].copy()
        if "File" in d.columns:
            d = d.sort_values("File")
        path = os.path.join(out_dir, f"{name}.csv")
        d.to_csv(path, index=False)
        paths[name] = path

    save_subset(idx_train, "train")
    save_subset(idx_val,   "val")
    save_subset(idx_test,  "test")

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Cuotas por región (por archivo) + Igualado GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Cuotas por región/subregión (por archivo) + Igualado GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

def _balanced_take(df_grp: pd.DataFrame, total: int, by_cols, seed: int = 42) -> pd.DataFrame:
    """Selecciona hasta 'total' filas de df_grp balanceando por las columnas dadas."""
    if total <= 0 or len(df_grp) <= total:
        return df_grp

    rng = np.random.default_rng(seed)
    strata = [(k, g.index.to_numpy()) for k, g in df_grp.groupby(by_cols, sort=False)]
    S = len(strata)
    base, rem = divmod(total, S)
    order = np.arange(S)
    rng.shuffle(order)
    take = np.full(S, base, int)
    take[order[:rem]] += 1

    chosen = []
    for (_, idxs), k in zip(strata, take):
        if len(idxs) <= k:
            chosen.append(idxs)
        else:
            chosen.append(rng.choice(idxs, size=k, replace=False))
    sel = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    rng.shuffle(sel)
    return df_grp.loc[sel]


def apply_per_file_region_quotas(df: pd.DataFrame, vars_: dict, seed: int = 42) -> pd.DataFrame:
    """
    Aplica cuotas por archivo, según region_row_quota o subregion_row_quota si está definido.
    Ejemplo:
      region_row_quota = {1:300, 2:100, 3:100, 4:100}
      → Cada archivo ("File") tomará como máximo esas filas por región.
    """
    # Prioridad: subregion_row_quota > region_row_quota
    quotas = vars_.get("subregion_row_quota", {}) or vars_.get("region_row_quota", {})
    if not quotas:
        print("[INFO] No se aplican cuotas por región/subregión (diccionario vacío).")
        return df

    parts = []
    for f, df_file in df.groupby("File", sort=False):
        sub_parts = []
        if "Subregion" in df_file.columns and vars_.get("subregion_row_quota"):
            # Aplica por subregión
            for (r, s), df_sub in df_file.groupby(["Region", "Subregion"], sort=False):
                q = int(quotas.get(int(s), 0)) or int(quotas.get(int(r), 0))
                df_sel = _balanced_take(df_sub, q, by_cols=["Pigment Index", "Mixture"], seed=seed)
                sub_parts.append(df_sel)
        else:
            # Aplica solo por región
            for r, df_reg in df_file.groupby("Region", sort=False):
                q = int(quotas.get(int(r), 0))
                df_sel = _balanced_take(df_reg, q, by_cols=["Pigment Index", "Mixture"], seed=seed)
                sub_parts.append(df_sel)

        df_used_file = pd.concat(sub_parts, ignore_index=True)
        parts.append(df_used_file)

    df_out = pd.concat(parts, ignore_index=True)
    print(f"[INFO] Cuotas aplicadas por archivo con {quotas}")
    print(f"[INFO] Filas originales: {len(df)}, tras cuotas: {len(df_out)}")
    return df_out


def save_region_subregion_usage(df_raw: pd.DataFrame, df_used: pd.DataFrame, out_path: str):
    """Genera CSV con resumen de filas totales y usadas por región, subregión y archivo."""
    has_sub = "Subregion" in df_raw.columns or "Subregion" in df_used.columns

    raw = df_raw.copy()
    used = df_used.copy()

    if has_sub:
        if "Subregion" not in raw.columns:
            raw["Subregion"] = np.nan
        if "Subregion" not in used.columns:
            used["Subregion"] = np.nan
        keys = ["File", "Region", "Subregion"]
    else:
        keys = ["File", "Region"]

    total = raw.groupby(keys, dropna=False).size().rename("total_rows").reset_index()
    usedc = used.groupby(keys, dropna=False).size().rename("used_rows").reset_index()

    summary = pd.merge(total, usedc, on=keys, how="outer")
    summary["total_rows"] = summary["total_rows"].fillna(0).astype(int)
    summary["used_rows"] = summary["used_rows"].fillna(0).astype(int)

    # Totales por región y subregión (sin "File")
    region_summary = summary.groupby(keys[1:], as_index=False)[["total_rows", "used_rows"]].sum()
    region_summary["File"] = "ALL_FILES"
    summary = pd.concat([summary, region_summary], ignore_index=True)

    summary = summary.sort_values(keys).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"[SAVE] Region/Subregion usage -> {out_path}")


def equalize_across_files_by_pigment(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    out = []
    for p, dfp in df.groupby("Pigment Index", sort=False):
        counts = dfp.assign(is_r1=dfp["Region"]==1, is_r234=dfp["Region"].isin([2,3,4])) \
                    .groupby("File").agg(n1=("is_r1","sum"), n234=("is_r234","sum"))
        T = int(counts[["n1","n234"]].min(axis=1).min())
        if T <= 0:
            out.append(dfp); continue
        for f, dff in dfp.groupby("File", sort=False):
            r1   = dff[dff["Region"] == 1]
            r234 = dff[dff["Region"].isin([2,3,4])]
            r1_sel   = _balanced_take(r1,   min(len(r1),   T), by_cols=["Mixture"], seed=seed)
            r234_sel = _balanced_take(r234, min(len(r234), T), by_cols=["Mixture","Region"], seed=seed)
            others   = dff[~dff["Region"].isin([1,2,3,4])]
            out.append(pd.concat([r1_sel, r234_sel, others], ignore_index=True))
    return pd.concat(out, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train HSI models + export CMs y CSVs de splits.")
    p.add_argument("--outputs-dir", type=str, default=None)
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = args.outputs_dir or variables.get("outputs_dir") or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Carga datos
    pr = HSIDataProcessor(variables)
    pr.load_h5_files()

    # 2) Bruto (sin filtros/cuotas/igualado)
    df_raw = pr.dataframe(mode="raw")

    # 3) Cuotas por región por archivo
    region_quotas = variables.get("region_row_quota", {}) or {}
    APPLY_QUOTAS = variables.get("apply_region_quotas", True)
    df_after_quotas = apply_per_file_region_quotas(
    df_raw, variables, seed=variables.get("balance_seed", 42)
    ) if (APPLY_QUOTAS and region_quotas) else df_raw

    # 4) Igualado GLOBAL por pigmento
    DO_EQUALIZE_GLOBAL = variables.get("equalize_across_files", True)
    df_used = equalize_across_files_by_pigment(
        df_after_quotas, seed=variables.get("balance_seed", 42)
    ) if DO_EQUALIZE_GLOBAL else df_after_quotas

    usage_csv = os.path.join(out_dir, "region_subregion_usage.csv")
    save_region_subregion_usage(df_raw, df_used, usage_csv)

    # 5) Split
    if variables.get("balance_test_by_mixture", True):
        idx_train, idx_val, idx_test = stratified_balanced_split_by_file_pigment_mixture(
            df_used,
            variables,
            per_mix=int(variables.get("test_per_mixture", 2)),
            seed=variables.get("seed", 42),
        )
    else:
        idx_train, idx_val, idx_test = stratified_split_70_15_15(
            df_used, variables, seed=variables.get("seed", 42)
        )

    # 6) X/y
    X, y, input_len = build_Xy(df_used)
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    print(f"[DATA] input_len={input_len} | X_train={X_train.shape} | X_val={X_val.shape} | X_test={X_test.shape}")

    # 7) Entrenar y evaluar
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in model_names:
        print(f"[TRAIN] {name}")
        tune = import_model_trainer(name)
        res = tune(
            X_train, y_train, X_val, y_val,
            input_len=input_len, num_classes=y.shape[1],
            trials=args.trials if args.trials is not None else variables.get("trials"),
            epochs=args.epochs if args.epochs is not None else variables.get("epochs"),
            n_jobs=variables.get("optuna_n_jobs", 1),
            seed=variables.get("seed", 42),
        )
        model = res[0] if isinstance(res, tuple) else res

        # === PREDICCIONES ===
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred_bin = (y_pred_prob > 0.5).astype(int)
        n_p = int(variables["num_files"])

        # === MÉTRICAS BIT A BIT ===
        TP = np.sum((y_test == 1) & (y_pred_bin == 1))
        TN = np.sum((y_test == 0) & (y_pred_bin == 0))
        FP = np.sum((y_test == 0) & (y_pred_bin == 1))
        FN = np.sum((y_test == 1) & (y_pred_bin == 0))

        bit_acc = (TP + TN) / (TP + TN + FP + FN)
        print(f"\n[INFO] Bitwise Accuracy (multi-label real): {bit_acc:.4f}")

        # === CLASES BASE ===
        classes_pig = [f"P{i+1:02d}" for i in range(n_p)]
        classes_mix = ["Pure", "M1", "M2", "M3"]

        # === 1️⃣ MATRIZ: SOLO PIGMENTS ===
        pig_true = np.argmax(y_test[:, :n_p], axis=1)
        pig_pred = np.argmax(y_pred_bin[:, :n_p], axis=1)
        cm_pig, used_pig = rep.confusion_from_labels(pig_true, pig_pred, classes=classes_pig)
        rep.plot_confusion_matrix(cm_pig, used_pig,
            "Confusión entre Pigmentos (bitwise multi-label)",
            os.path.join(sub_out, f"{name}_cm_PIGMENTS.png"))

        # === 2️⃣ MATRIZ: SOLO MIXTURES (sin Pure) ===
        mix_true_full = np.argmax(y_test[:, n_p:], axis=1)
        mix_pred_full = np.argmax(y_pred_bin[:, n_p:], axis=1)

        # Filtramos solo las filas donde no sea Pure
        mask_mix = mix_true_full != 0
        mix_true = mix_true_full[mask_mix]
        mix_pred = mix_pred_full[mask_mix]

        classes_mix_wo_pure = ["M1", "M2", "M3"]
        cm_mix, used_mix = rep.confusion_from_labels(
            mix_true - 1, mix_pred - 1, classes=classes_mix_wo_pure
        )
        rep.plot_confusion_matrix(cm_mix, used_mix,
            "Confusión entre Mezclas (M1–M3, bitwise multi-label)",
            os.path.join(sub_out, f"{name}_cm_MIXTURES_only.png"))

        # === 3️⃣ MATRIZ: PURE vs MIXTURE ===
        pure_true = np.where(mix_true_full == 0, "Pure", "Mixture")
        pure_pred = np.where(mix_pred_full == 0, "Pure", "Mixture")
        classes_puremix = ["Pure", "Mixture"]
        cm_puremix, used_pm = rep.confusion_from_labels(pure_true, pure_pred, classes=classes_puremix)
        rep.plot_confusion_matrix(cm_puremix, used_pm,
            "Confusión global Pure vs Mixture (multi-label)",
            os.path.join(sub_out, f"{name}_cm_PURE_vs_MIXTURE.png"))

        # === 4️⃣ MATRIZ: FULL Pigment + Mix4 (como antes, pero bitwise coherente) ===
        pig_idx_true = np.argmax(y_test[:, :n_p], axis=1)
        mix_idx_true = np.argmax(y_test[:, n_p:], axis=1)
        pig_idx_pred = np.argmax(y_pred_bin[:, :n_p], axis=1)
        mix_idx_pred = np.argmax(y_pred_bin[:, n_p:], axis=1)

        y_true_full = [f"P{p+1:02d}_{classes_mix[m]}" for p, m in zip(pig_idx_true, mix_idx_true)]
        y_pred_full = [f"P{p+1:02d}_{classes_mix[m]}" for p, m in zip(pig_idx_pred, mix_idx_pred)]

        classes_full = [f"P{i+1:02d}_{s}" for i in range(n_p) for s in classes_mix]
        cm_full, used_full = rep.confusion_from_labels(y_true_full, y_pred_full, classes=classes_full)
        rep.plot_confusion_matrix(cm_full, used_full,
            "Confusión completa Pigment + Mix4 (bitwise multi-label)",
            os.path.join(sub_out, f"{name}_cm_FULL_4CASES.png"))

        # === ACCURACY GLOBAL ===
        eval_res = model.evaluate(X_test, y_test, verbose=0)
        keras_acc = eval_res[1] if isinstance(eval_res, (list, tuple)) else float(eval_res)
        print(f"Accuracy (keras evaluate) = {keras_acc:.4f}")
        print(f"Accuracy (bitwise, manual) = {bit_acc:.4f}")
        print(f"Diferencia (|eval - bitwise|) = {abs(keras_acc - bit_acc):.4f}\n")

        print("[INFO] ==============================================\n")



        # Mostramos la comparación explícita
        print(f"Accuracy (Pigment+Mix4, decodificada) = {acc_4N:.4f}")
        print(f"Accuracy desde CM 4N = {acc_from_cm_4N:.4f}")
        print(f"Diferencia (|eval - CM|) = {abs(keras_acc - acc_from_cm_4N):.4f}")
        print(f"Diferencia (|eval - 4N|) = {abs(keras_acc - acc_4N):.4f}")
        print(f"Accuracy (Pigment+Group2) = {acc_2N:.4f}")
        print(f"Accuracy (Pure vs Mixture) = {acc_mix:.4f}")
        print("[INFO] ==============================================\n")

        # ─────────────────────────────────────────────────────────────
        # Guardar resultados y comparación en CSV
        # ─────────────────────────────────────────────────────────────
        metrics_out_dir = os.path.join(out_dir, name)
        os.makedirs(metrics_out_dir, exist_ok=True)
        csv_path = os.path.join(metrics_out_dir, "results_summary.csv")

        header = [
            "model_name",
            "keras_acc",
            "acc_4N",
            "acc_from_cm_4N",
            "diff_eval_vs_cm",
            "diff_eval_vs_4N",
            "acc_2N",
            "acc_mix"
        ]
        row = [
            name,
            round(float(keras_acc), 6),
            round(float(acc_4N), 6),
            round(float(acc_from_cm_4N), 6),
            round(abs(keras_acc - acc_from_cm_4N), 6),
            round(abs(keras_acc - acc_4N), 6),
            round(float(acc_2N), 6),
            round(float(acc_mix), 6)
        ]

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

        print(f"[SAVE] Resultados comparativos -> {csv_path}")


        # CSVs usados (solo el dataframe final por modelo)
        csv_out_dir = os.path.join(out_dir, name, "datasets"); os.makedirs(csv_out_dir, exist_ok=True)
        df_used_path = os.path.join(csv_out_dir, "dataframe_used.csv")
        df_used.to_csv(df_used_path, index=False)
        print(f"[SAVE] DataFrame usado -> {df_used_path}")

        y_pred_prob_full = np.full_like(y, fill_value=np.nan, dtype=np.float32)
        y_pred_prob_full[idx_test] = y_pred_prob
        _ = export_splits_csv(
            df=df_used, y_true=y, y_pred_prob=y_pred_prob_full,
            idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
            out_dir=csv_out_dir, vars_=variables,
        )

if __name__ == "__main__":
    main()



