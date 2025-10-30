# ============================================================================ #
# IMPORTS
# ============================================================================ #
import os
import csv
import argparse
import importlib
import warnings
from typing import Tuple, Dict
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, hamming_loss, log_loss, average_precision_score
)

from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.data.config import variables


# ============================================================================ #
# MODEL TRAINER IMPORT
# ============================================================================ #
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
        f"Failed to load `tune_and_train` for '{name}'. Tried: {candidates}. "
        f"Last error: {last_err}"
    )


# ============================================================
# Kubelka–Munk 
# ============================================================

def reflectance_to_ks(R):
    R = np.clip(R, 1e-3, 1.0)
    return ((1 - R) ** 2) / (2 * R)


def ks_to_reflectance(KS):
    return 1 + KS - np.sqrt(KS ** 2 + 2 * KS)


def km_unmix(KS_mix, k_set, s_set):
    n = k_set.shape[0]

    def loss_fn(alphas):
        K_mix_est = np.dot(alphas, k_set)
        S_mix_est = np.dot(alphas, s_set)
        KS_est = K_mix_est / S_mix_est
        return np.mean((KS_est - KS_mix) ** 2)

    constraints = [{'type': 'eq', 'fun': lambda a: np.sum(a) - 1}]
    bounds = [(0, 1)] * n
    x0 = np.full(n, 1 / n)
    result = minimize(loss_fn, x0, bounds=bounds, constraints=constraints)
    return result.x if result.success else np.zeros(n)


def apply_km_unmixing(R_all, k_set, s_set):
    KS_all = reflectance_to_ks(R_all)
    alphas = []
    for i, KS in enumerate(KS_all):
        alpha = km_unmix(KS, k_set, s_set)
        alphas.append(alpha)
        if (i + 1) % 500 == 0:
            print(f"[KM] {i + 1} processed samples")
    return np.array(alphas)


def fast_km_unmixing(mixtures, k_set, s_set):
    n_pigments, n_wl = k_set.shape
    KS_mix = mixtures.T  # (n_wl, n_samples)

    # Solve for all mixtures at once using least-squares
    alphas_pred, _, _, _ = np.linalg.lstsq(k_set.T, KS_mix, rcond=None)
    alphas_pred = alphas_pred.T  # (n_samples, n_pigments)

    # Ensure positivity and normalization
    alphas_pred = np.clip(alphas_pred, 0, 1)
    alphas_pred /= np.maximum(np.sum(alphas_pred, axis=1, keepdims=True), 1e-8)

    return alphas_pred



def match_spectral_resolution(spectra, target_bands=267, data_type=None):
    from hsi_lab.data.config import variables
    if data_type is None:
        data_type = variables.get("data_type", ["vis"])
    if isinstance(data_type, str):
        data_type = [data_type.lower()]

    n_bands = spectra.shape[-1]

    # Caso 1: igual tamaño
    if n_bands == target_bands:
        return spectra

    # Caso 2: degenerado (1 sola banda)
    if n_bands == 1:
        return np.repeat(spectra, target_bands, axis=-1)

    # === Interpolación por región ===
    if data_type == ["vis"]:
        start_nm, end_nm = 400, 1000
        old_axis = np.linspace(start_nm, end_nm, n_bands)
        new_axis = np.linspace(start_nm, end_nm, target_bands)
        f = interp1d(old_axis, spectra, axis=-1, kind="linear", fill_value="extrapolate")
        return f(new_axis)

    elif data_type == ["swir"]:
        start_nm, end_nm = 1000, 2500
        old_axis = np.linspace(start_nm, end_nm, n_bands)
        new_axis = np.linspace(start_nm, end_nm, target_bands)
        f = interp1d(old_axis, spectra, axis=-1, kind="linear", fill_value="extrapolate")
        return f(new_axis)

    elif set(data_type) == {"vis", "swir"}:
        # ⚠️ Combinar respetando longitud real de cada bloque
        vis_n = n_bands // 2
        swir_n = n_bands - vis_n

        vis_old = np.linspace(400, 1000, vis_n)
        swir_old = np.linspace(1000, 2500, swir_n)

        vis_new = np.linspace(400, 1000, target_bands)
        swir_new = np.linspace(1000, 2500, target_bands)

        f_vis = interp1d(vis_old, spectra[..., :vis_n], axis=-1, kind="linear", fill_value="extrapolate")
        f_swir = interp1d(swir_old, spectra[..., vis_n:], axis=-1, kind="linear", fill_value="extrapolate")

        vis_interp = f_vis(vis_new)
        swir_interp = f_swir(swir_new)

        # Concatenar con una pequeña separación NaN si quieres un gap visible
        return np.concatenate([vis_interp, swir_interp], axis=-1)

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")




def generate_synthetic_mixtures(R_pigments, n_samples=12000, n_mix=(2, 3)):
    n_pigments, n_wl = R_pigments.shape
    KS_pigments = reflectance_to_ks(R_pigments)

    mixtures, proportions, components = [], [], []

    for _ in range(n_samples):
        n = np.random.choice(n_mix)
        idx = np.random.choice(n_pigments, n, replace=False)

        # Mezclas fijas o aleatorias
        if n == 2 and np.random.rand() < 0.5:
            alpha = np.array([0.2, 0.8])
            np.random.shuffle(alpha)
        elif n == 3 and np.random.rand() < 0.5:
            base = np.array([0.78, 0.12, 0.10])
            alpha = base / base.sum()
            np.random.shuffle(alpha)
        else:
            a = np.random.rand(n)
            alpha = a / a.sum()

        # Mezcla lineal en el dominio K/S
        KS_mix = np.dot(alpha, KS_pigments[idx])
        R_mix = ks_to_reflectance(KS_mix)

        mixtures.append(R_mix)
        proportions.append(alpha)
        components.append(idx)

    return np.array(mixtures), proportions, components


# ============================================================================ #
# DATASET BUILDING AND SPLITTING
# ============================================================================ #
def build_Xy(df: pd.DataFrame):
    spec_cols = [c for c in df.columns if c.startswith(("vis_", "swir_"))]
    if not spec_cols:
        raise ValueError("No spectral columns starting with 'vis_' or 'swir_'.")
    vis_cols = sorted(
        [c for c in spec_cols if c.startswith("vis_")],
        key=lambda c: int(c.split("_")[1])
    )
    swir_cols = sorted(
        [c for c in spec_cols if c.startswith("swir_")],
        key=lambda c: int(c.split("_")[1])
    )
    spec_cols_sorted = vis_cols + swir_cols
    X = df[spec_cols_sorted].astype(np.float32).fillna(0.0).values[..., np.newaxis]
    y = np.array([np.array(v) for v in df["Multi"]], dtype=np.float32)
    return X, y, X.shape[1]


def pigment_ids(df: pd.DataFrame, vars_: dict) -> np.ndarray:
    """Returns the main pigment ID (argmax) from 'Multi' per sample."""
    n_p = int(vars_["num_files"])
    return np.array([int(np.argmax(np.asarray(v, dtype=np.float32)[:n_p])) for v in df["Multi"]], dtype=int)


def stratified_balanced_split_by_file_pigment_mixture(df, vars_, per_mix=2, seed=42):
    """Performs a balanced stratified split by File × Pigment × Mixture."""
    rng = np.random.default_rng(seed)
    labels = df["File"].astype(str) + "_" + df["Pigment Index"].astype(str) + "_" + df["Mixture"].astype(str)
    grp = df.groupby(labels)
    train_idx, val_idx, test_idx = [], [], []

    for _, g in grp:
        idxs = g.index.to_numpy()
        rng.shuffle(idxs)
        if len(idxs) >= 4:
            n_test = per_mix
            n_val = max(1, int(0.15 * len(idxs)))
            n_train = len(idxs) - n_test - n_val
        else:
            n_test = min(per_mix, len(idxs) // 2)
            n_val = 1 if len(idxs) > 2 else 0
            n_train = len(idxs) - n_test - n_val
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[-n_test:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def stratified_split_70_15_15(df: pd.DataFrame, vars_: dict, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits dataset into 70/15/15 stratified by pigment ID."""
    y_pig = pigment_ids(df, vars_)
    idx_all = np.arange(len(df))
    idx_train, idx_tmp = train_test_split(idx_all, test_size=0.30, random_state=seed, stratify=y_pig)
    y_tmp = y_pig[idx_tmp]
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    return np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test)


# ============================================================================ #
# DATA SPLIT EXPORT (SOFT)
# ============================================================================ #
def export_splits_csv(df: pd.DataFrame, y_true: np.ndarray, y_pred_prob: np.ndarray,
                      idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray,
                      out_dir: str, vars_: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    n_p = int(vars_["num_files"])
    N, D = y_true.shape

    if y_pred_prob is None:
        y_pred_prob = np.full((N, D), np.nan, dtype=np.float32)

    y_true_2N = decode_pigment_and_group(y_true, n_p)
    y_true_4N = decode_pigment_and_mix4(y_true, n_p)

    def safe_decode(decoder_fn, y_like):
        out = np.array([""] * len(y_like), dtype=object)
        valid = np.isfinite(y_like).all(axis=1)
        if valid.any():
            out[valid] = decoder_fn(y_like[valid], n_p)
        return out

    y_pred_2N = safe_decode(decode_pigment_and_group, y_pred_prob)
    y_pred_4N = safe_decode(decode_pigment_and_mix4, y_pred_prob)

    def split_pm(lbl_4n: str):
        if not lbl_4n:
            return "", ""
        px, m = lbl_4n.split("_", 1)
        return px, m

    pig_true, mix_true = zip(*[split_pm(s) for s in y_true_4N])
    pig_pred, mix_pred = zip(*[split_pm(s) for s in y_pred_4N])

    base = df.copy()
    base = base.assign(
        y_true_2N=y_true_2N,
        y_pred_2N=y_pred_2N,
        y_true_4N=y_true_4N,
        y_pred_4N=y_pred_4N,
        pig_true=np.array(pig_true, dtype=object),
        mix_true=np.array(mix_true, dtype=object),
        pig_pred=np.array(pig_pred, dtype=object),
        mix_pred=np.array(mix_pred, dtype=object),
    )

    paths = {}

    def save_subset(indices: np.ndarray, name: str):
        d = base.iloc[np.sort(indices)].copy()
        if "File" in d.columns:
            d = d.sort_values("File")
        path = os.path.join(out_dir, f"{name}.csv")
        d.to_csv(path, index=False)
        paths[name] = path

    save_subset(idx_train, "train")
    save_subset(idx_val, "val")
    save_subset(idx_test, "test")

    return paths


# ============================================================================ #
# REGION / SUBREGION QUOTAS
# ============================================================================ #
def _balanced_take(df_grp: pd.DataFrame, total: int, by_cols, seed: int = 42) -> pd.DataFrame:
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
    quotas = vars_.get("subregion_row_quota", {}) or vars_.get("region_row_quota", {})
    if not quotas:
        print("[INFO] No quotas applied (empty dictionary).")
        return df
    parts = []
    for f, df_file in df.groupby("File", sort=False):
        sub_parts = []
        if "Subregion" in df_file.columns and vars_.get("subregion_row_quota"):
            for (r, s), df_sub in df_file.groupby(["Region", "Subregion"], sort=False):
                q = int(quotas.get(int(s), 0)) or int(quotas.get(int(r), 0))
                df_sel = _balanced_take(df_sub, q, by_cols=["Pigment Index", "Mixture"], seed=seed)
                sub_parts.append(df_sel)
        else:
            for r, df_reg in df_file.groupby("Region", sort=False):
                q = int(quotas.get(int(r), 0))
                df_sel = _balanced_take(df_reg, q, by_cols=["Pigment Index", "Mixture"], seed=seed)
                sub_parts.append(df_sel)
        df_used_file = pd.concat(sub_parts, ignore_index=True)
        parts.append(df_used_file)
    df_out = pd.concat(parts, ignore_index=True)
    print(f"[INFO] Region quotas applied with {quotas}. Original {len(df)} → {len(df_out)}.")
    return df_out


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


def save_region_subregion_usage(df_raw: pd.DataFrame, df_used: pd.DataFrame, out_path: str):
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
    region_summary = summary.groupby(keys[1:], as_index=False)[["total_rows", "used_rows"]].sum()
    region_summary["File"] = "ALL_FILES"
    summary = pd.concat([summary, region_summary], ignore_index=True)
    summary = summary.sort_values(keys).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"[SAVE] Region/Subregion usage -> {out_path}")


# ============================================================================ #
# CLI PARSER
# ============================================================================ #
def parse_args():
    p = argparse.ArgumentParser(description="Train HSI models + export CMs and CSVs.")
    p.add_argument("--outputs-dir", type=str, default=None)
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


# ============================================================================ #
# MODEL SUMMARY AND METRICS
# ============================================================================ #
def summarize_model(name, variables, X_train, y_train, model, base_out):
    n_pigments = int(variables["num_files"])
    n_mixtures = 4
    total_classes = y_train.shape[1]
    summary = {
        "Model name": name,
        "Input bands": X_train.shape[1],
        "Output classes": total_classes,
        "Pigments": n_pigments,
        "Mixtures": n_mixtures,
        "Loss": "Binary cross-entropy",
        "Optimizer": "Adam",
        "Epochs": variables.get("epochs", "N/A"),
        "Batch size": variables.get("batch_size", "N/A"),
        "Samples (train)": len(X_train),
    }
    df_summary = pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"])
    out_path = os.path.join(base_out, f"{name}_model_summary.csv")
    df_summary.to_csv(out_path, index=False)
    print(f"[SAVE] Model summary -> {out_path}")
    return df_summary


def per_pigment_metrics(y_true, y_pred_prob, variables, base_out, name):
    n_p = int(variables["num_files"])
    mix_names = ["Pure", "M1", "M2", "M3"]
    rows = []
    for i in range(n_p):
        for j, mix in enumerate(mix_names):
            y_t = y_true[:, [i, n_p + j]]
            y_p = y_pred_prob[:, [i, n_p + j]]
            m, _ = compute_metrics(y_t, (y_p > 0.5).astype(int), y_p)
            rows.append({
                "Pigment": f"P{i+1:02d}",
                "Mixture": mix,
                "F1": m["f1"],
                "Precision": m["precision"],
                "Recall": m["recall"]
            })
    df = pd.DataFrame(rows)
    out_path = os.path.join(base_out, f"{name}_per_pigment_mix_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Per-pigment × mixture metrics -> {out_path}")
    return df


def top_confusions(cm, classes, base_out, name, top_k=15):
    flat = []
    for i, c_true in enumerate(classes):
        for j, c_pred in enumerate(classes):
            if i != j and cm[i, j] > 0:
                flat.append((c_true, c_pred, cm[i, j]))
    df = pd.DataFrame(flat, columns=["True", "Predicted", "Value"])
    df = df.sort_values("Value", ascending=False).head(top_k)
    path = os.path.join(base_out, f"{name}_top_confusions.csv")
    df.to_csv(path, index=False)
    print(f"[SAVE] Top-{top_k} confusions -> {path}")
    return df


# ============================================================
# === SPECTRAL CONFIGURATION (FIXED & ROBUST) ===
# ============================================================

import numpy as np

def get_wavelengths_and_labels(total_bands, data_type=["vis"], vis_start=400, vis_end_nominal=1000, swir_start=1000, swir_end_nominal=2500):
    """
    Genera ejes de longitudes de onda (wavelengths) VIS y/o SWIR ajustados al número real de bandas.
    - Si solo hay VIS, escala automáticamente hasta el rango real.
    - Si hay VIS+SWIR, divide proporcionalmente según las bandas reales.
    - Incluye etiquetas (ticks) adaptadas al rango.
    """

    wavelengths = []
    tick_positions = []
    tick_labels = []

    data_type = [d.lower() for d in data_type]

    # --- Parámetros nominales de referencia ---
    vis_nominal_bands = 267
    swir_nominal_bands = 267

    # --- Caso VIS+SWIR ---
    if "vis" in data_type and "swir" in data_type:
        vis_total = total_bands // 2
        swir_total = total_bands - vis_total
    elif "vis" in data_type:
        vis_total = total_bands
        swir_total = 0
    elif "swir" in data_type:
        vis_total = 0
        swir_total = total_bands
    else:
        raise ValueError("data_type must include 'vis', 'swir', or both.")

    # --- VIS ---
    if vis_total > 0:
        vis_step_nominal = (vis_end_nominal - vis_start) / (vis_nominal_bands - 1)
        vis_end_real = vis_start + vis_step_nominal * (vis_total - 1)

        vis = np.linspace(vis_start, vis_end_real, vis_total, endpoint=True)
        wavelengths.extend(vis)

        vis_ticks = np.linspace(0, vis_total - 1, 10, dtype=int)
        for i in vis_ticks:
            tick_positions.append(vis[i])
            tick_labels.append(f"VIS ({int(vis[i])} nm)")

    # --- SWIR ---
    if swir_total > 0:
        swir_step_nominal = (swir_end_nominal - swir_start) / (swir_nominal_bands - 1)
        swir_end_real = swir_start + swir_step_nominal * (swir_total - 1)

        swir = np.linspace(swir_start, swir_end_real, swir_total, endpoint=True)
        wavelengths.extend(swir)

        swir_ticks = np.linspace(0, swir_total - 1, 10, dtype=int)
        for i in swir_ticks:
            tick_positions.append(swir[i])
            tick_labels.append(f"SWIR ({int(swir[i])} nm)")

    return np.array(wavelengths), tick_positions, tick_labels




# ============================================================================ #
# QUALITATIVE VISUALIZATION: Spectra comparison
# ============================================================================ #
def plot_comparative_spectra(X_test, y_test, y_pred_prob, base_out, name,
                             pigment_indices=None, avg_blocks=5, threshold=0.5):

    spectra_dir = os.path.join(base_out, "Spectra")
    os.makedirs(spectra_dir, exist_ok=True)

    # Ensure shape consistency
    X_test = np.array(X_test)
    if X_test.ndim == 2:
        X_test = X_test[..., np.newaxis]
    n_samples, n_bands, _ = X_test.shape

    # Load wavelength configuration
    wavelengths, wavelength_axis, spectral_ticks = get_wavelength_config(variables["data_type"])

    if len(wavelength_axis) != n_bands:
        print(f"[WARN] Mismatch between wavelength axis ({len(wavelength_axis)}) and data bands ({n_bands}). Adjusting...")
        if len(wavelength_axis) > n_bands:
            wls = wavelength_axis[:n_bands]
        else:
            wls = np.linspace(wavelength_axis[0], wavelength_axis[-1], n_bands)
    else:
        wls = wavelength_axis

    n_pigments = y_test.shape[1] - 4  # assumes last 4 cols are mixtures
    if pigment_indices is None:
        pigment_indices = range(min(3, n_pigments))

    # Loop through pigments to visualize spectra
    for pigment_idx in pigment_indices:
        true_mask = y_test[:, pigment_idx] > threshold
        pred_mask = y_pred_prob[:, pigment_idx] > threshold

        if not np.any(true_mask):
            print(f"[WARN] Pigment P{pigment_idx+1:02d} has no true samples.")
            continue

        spectra_true = X_test[true_mask, :, 0]
        spectra_pred = X_test[pred_mask, :, 0]

        # Compute min/max bands for fill
        min_t, max_t = spectra_true.min(axis=0), spectra_true.max(axis=0)
        min_p, max_p = (spectra_pred.min(axis=0), spectra_pred.max(axis=0)) if len(spectra_pred) > 0 else (None, None)

        plt.figure(figsize=(10, 6))
        plt.fill_between(wls, min_t, max_t, alpha=0.15, color="blue", label="True range")
        if min_p is not None:
            plt.fill_between(wls, min_p, max_p, alpha=0.15, color="orange", label="Pred range")
        plt.plot(wls, np.mean(spectra_true, axis=0), color="blue", linewidth=2, label="True avg")
        if len(spectra_pred) > 0:
            plt.plot(wls, np.mean(spectra_pred, axis=0), color="orange", linewidth=2, label="Pred avg")

        # === Axis formatting: consistent VIS/SWIR + channel + wavelength ===
        plt.title(f"Pigment P{pigment_idx+1:02d} — Average spectra")
        plt.xlabel("Data type, channel and wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.legend(loc="upper right", fontsize=8)

        # Use unified spectral tick style
        tick_positions, tick_labels = spectral_ticks_labels(wls, n_ticks_vis=10, n_ticks_swir=10)
        plt.xticks(tick_positions, tick_labels, rotation=65, ha="right", fontsize=8)

        plt.tight_layout()


        print(f"[SAVE] Spectra plot -> {out_path}")

        print("[DEBUG] X_test sample shape:", X_test[0, :, 0].shape)
        print("[DEBUG] First 10 reflectance values:", X_test[0, :10, 0])
        print("[DEBUG] Wavelength axis (first 10):", wls[:10])

# ============================================================================ #
# SUMMARY + CONCLUSIONS
# ============================================================================ #
def summarize_confusion_matrices(cm_dict, detailed_results, name, base_out):
    def multilabel_metrics_from_coactivation(cm):
        cm = np.array(cm, dtype=np.float64)
        diag = np.diag(cm)
        row_sums = np.sum(cm, axis=1)
        col_sums = np.sum(cm, axis=0)
        precisions = np.divide(diag, col_sums, out=np.zeros_like(diag), where=col_sums != 0)
        recalls = np.divide(diag, row_sums, out=np.zeros_like(diag), where=row_sums != 0)
        f1s = np.divide(2 * precisions * recalls, precisions + recalls + 1e-8)
        return {
            "soft_accuracy": np.nanmean(diag),
            "soft_precision_macro": np.nanmean(precisions),
            "soft_recall_macro": np.nanmean(recalls),
            "soft_f1_macro": np.nanmean(f1s)
        }

    global_metrics, _ = detailed_results.get("global", ({}, {}))
    model_hamming = global_metrics.get("hamming_acc", np.nan)
    model_f1 = global_metrics.get("f1", np.nan)

    csv_out = os.path.join(base_out, "datasets", f"{name}_confusion_summary.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    header = [
        "Matrix Type", "Classes", "Soft Accuracy", "Soft Precision (macro)",
        "Soft Recall (macro)", "Soft F1 (macro)", "Model HammingAcc",
        "Model F1", "Δ(SoftAcc - HammingAcc)", "Δ(SoftF1 - ModelF1)", "Interpretation"
    ]

    rows = []
    for key, info in cm_dict.items():
        cm = info["matrix"]
        desc = info["desc"]
        classes = len(info["classes"])
        soft_metrics = multilabel_metrics_from_coactivation(cm)
        diff_acc = soft_metrics["soft_accuracy"] - model_hamming
        diff_f1 = soft_metrics["soft_f1_macro"] - model_f1
        interp = "Soft and model accuracies align closely." if abs(diff_acc) < 0.02 else (
            "Matrix shows lower consistency; possible inter-label confusion." if diff_acc < 0
            else "Matrix consistency slightly higher; model may underpenalize overlaps."
        )
        rows.append([
            desc, classes,
            round(soft_metrics["soft_accuracy"], 4),
            round(soft_metrics["soft_precision_macro"], 4),
            round(soft_metrics["soft_recall_macro"], 4),
            round(soft_metrics["soft_f1_macro"], 4),
            round(model_hamming, 4),
            round(model_f1, 4),
            round(diff_acc, 4),
            round(diff_f1, 4),
            interp
        ])

    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[SAVE] Confusion/Coactivation summary -> {csv_out}")
    return csv_out


def write_conclusions(detailed_results, base_out, name):
    m_global, _ = detailed_results["global"]
    f1 = m_global.get("f1", 0)
    acc = m_global.get("strict_acc", 0)
    roc = m_global.get("roc_auc", 0)
    path = os.path.join(base_out, f"{name}_conclusions.txt")
    with open(path, "w") as f:
        f.write("=== Conclusions ===\n")
        f.write(f"Global F1: {f1:.3f}\n")
        f.write(f"Strict accuracy: {acc:.3f}\n")
        f.write(f"ROC-AUC: {roc:.3f}\n\n")
        f.write("Model shows strong discrimination between pigments, but mixtures introduce variability.\n")
        f.write("Improving data balance across mixtures or applying spectral attention could help.\n")
    print(f"[SAVE] Conclusions -> {path}")


# ============================================================================ #
# LABEL DECODERS
# ============================================================================ #
def decode_pigment_and_group(y_like: np.ndarray, n_p: int, threshold=0.5):
    y_pig = y_like[:, :n_p]
    y_mix = y_like[:, n_p:n_p + 4]
    results = []
    for i in range(len(y_like)):
        active_pigs = np.where(y_pig[i] > threshold)[0]
        active_mix = np.where(y_mix[i] > threshold)[0]
        labels = []
        for p in active_pigs:
            if 0 in active_mix:
                labels.append(f"P{p + 1:02d}_Pure")
            if any(m > 0 for m in active_mix):
                labels.append(f"P{p + 1:02d}_Mixture")
        results.append(";".join(labels) if labels else "")
    return np.array(results, dtype=object)


def decode_pigment_and_mix4(y_like: np.ndarray, n_p: int, threshold=0.5):
    y_pig = y_like[:, :n_p]
    y_mix = y_like[:, n_p:n_p + 4]
    mix_names = ["Pure", "M1", "M2", "M3"]
    results = []
    for i in range(len(y_like)):
        active_pigs = np.where(y_pig[i] > threshold)[0]
        active_mix = np.where(y_mix[i] > threshold)[0]
        labels = [f"P{p + 1:02d}_{mix_names[m]}" for p in active_pigs for m in active_mix]
        results.append(";".join(labels) if labels else "")
    return np.array(results, dtype=object)


# ============================================================================ #
# VISUALIZATION UTILITIES
# ============================================================================ #
def plot_confusion_matrix(cm, classes, title, save_path, annotate_percent=True):
    """Plots confusion or coactivation matrices with consistent styling."""
    n = len(classes)
    fig_size = max(6, min(0.45 * n, 20))
    plt.figure(figsize=(fig_size, fig_size))
    ax = plt.gca()

    base_cmap = mpl_cm.get_cmap("Blues", 256)
    cmap_array = base_cmap(np.linspace(0, 1, 256))
    cmap_array[0, :] = [1, 1, 1, 1]
    cmap = ListedColormap(cmap_array)

    vmax = np.max(cm) if np.max(cm) > 0 else 1.0
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j] * 100 if annotate_percent else cm[i, j]
            if value < 0.005:
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.2)
    plt.colorbar(im, cax=cax, format="%.2f")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVE] Matrix -> {save_path}")


# ============================================================================ #
# CONFUSION & COACTIVATION BUILDERS
# ============================================================================ #
def confusion_from_labels(y_true_labels, y_pred_labels, classes=None):
    """Builds a normalized confusion matrix from label strings."""
    y_true_labels = np.asarray(y_true_labels, dtype=object)
    y_pred_labels = np.asarray(y_pred_labels, dtype=object)
    if classes is None:
        seen = []
        for x in list(y_true_labels) + list(y_pred_labels):
            if x not in seen:
                seen.append(x)
        classes = list(seen)

    idx = {c: i for i, c in enumerate(classes)}
    valid_true = np.array([lbl in idx for lbl in y_true_labels])
    valid_pred = np.array([lbl in idx for lbl in y_pred_labels])
    mask = valid_true & valid_pred
    ti = np.array([idx[lbl] for lbl in y_true_labels[mask]], dtype=int)
    pi = np.array([idx[lbl] for lbl in y_pred_labels[mask]], dtype=int)
    cm = sk_confusion_matrix(ti, pi, labels=range(len(classes)))
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0.0] = 1.0
    return cm.astype(float) / row_sums, classes


def soft_confusion_matrix(y_true, y_pred_prob, class_names, normalize="row"):
    """Computes soft (coactivation) confusion matrix."""
    cm = np.dot(y_true.T, y_pred_prob)
    if normalize == "row":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
    return cm


# ============================================================================ #
# METRICS COMPUTATION
# ============================================================================ #
def compute_metrics(y_true, y_pred_bin, y_pred_prob):
    """Computes all main multi-label metrics."""
    metrics = {}
    y_true_b = (y_true > 0.5).astype(bool)
    y_pred_b = (y_pred_bin > 0.5).astype(bool)

    metrics["non_zero_acc_sample"] = np.mean(np.any(y_true_b & y_pred_b, axis=1))
    metrics["non_zero_acc_label"]  = np.mean(np.any(y_true_b == y_pred_b, axis=0))
    metrics["strict_acc"]          = np.mean(np.all(y_true_b == y_pred_b, axis=1))
    metrics["general_acc"]         = accuracy_score(y_true_b.flatten(), y_pred_b.flatten())
    metrics["keras_like_acc"]      = (y_true_b == y_pred_b).mean(axis=1).mean()
    metrics["hamming_acc"]         = 1 - hamming_loss(y_true_b, y_pred_b)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_b, np.clip(y_pred_prob, 0, 1), average="micro")
        except ValueError:
            metrics["roc_auc"] = np.nan
        try:
            metrics["avg_precision"] = average_precision_score(y_true_b, np.clip(y_pred_prob, 0, 1), average="micro")
        except ValueError:
            metrics["avg_precision"] = np.nan
        try:
            metrics["log_loss"] = log_loss(y_true_b.astype(int), np.clip(y_pred_prob, 1e-7, 1 - 1e-7))
        except ValueError:
            metrics["log_loss"] = np.nan

    metrics["f1"]        = f1_score(y_true_b, y_pred_b, average="micro", zero_division=0)
    metrics["precision"] = precision_score(y_true_b, y_pred_b, average="micro", zero_division=0)
    metrics["recall"]    = recall_score(y_true_b, y_pred_b, average="micro", zero_division=0)

    descriptions = {
        "non_zero_acc_sample": "Fraction of samples where at least one true label was correctly predicted.",
        "non_zero_acc_label":  "Fraction of labels with at least one correct prediction across all samples.",
        "strict_acc":          "Strict accuracy: all predicted labels must match the true labels.",
        "general_acc":         "Overall accuracy across all labels.",
        "keras_like_acc":      "Sample-wise mean accuracy (Keras-style).",
        "hamming_acc":         "Average per-label accuracy (1 − Hamming loss).",
        "roc_auc":             "Area under ROC (micro).",
        "avg_precision":       "Average precision (micro).",
        "log_loss":            "Penalty for confident incorrect predictions.",
        "f1":                  "F1-score (micro average).",
        "precision":           "Fraction of predicted positives that were correct.",
        "recall":              "Fraction of true positives that were correctly predicted.",
    }

    return metrics, descriptions


# ============================================================================ #
# COMBINED REPORT CREATOR
# ============================================================================ #
def generate_combined_report(y_true, y_pred_prob, n_pigments, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    cm_conf_dir = os.path.join(output_dir, "confusion_matrix")
    cm_coact_dir = os.path.join(output_dir, "coactivation_matrix")
    os.makedirs(cm_conf_dir, exist_ok=True)
    os.makedirs(cm_coact_dir, exist_ok=True)

    y_pred_bin = (y_pred_prob > 0.5).astype(int)
    y_true_pig, y_true_mix = y_true[:, :n_pigments], y_true[:, n_pigments:n_pigments+4]
    y_pred_pig, y_pred_mix = y_pred_bin[:, :n_pigments], y_pred_bin[:, n_pigments:n_pigments+4]

    classes_pig = [f"P{i+1:02d}" for i in range(n_pigments)]
    classes_mix = ["Pure", "M1", "M2", "M3"]

    # === CONFUSION ===
    y_true_pig_idx = np.argmax(y_true_pig, axis=1)
    y_pred_pig_idx = np.argmax(y_pred_prob[:, :n_pigments], axis=1)
    cm_conf_pig = sk_confusion_matrix(y_true_pig_idx, y_pred_pig_idx, normalize="true")
    plot_confusion_matrix(cm_conf_pig, classes_pig, "Pigments (Confusion)", os.path.join(cm_conf_dir, f"{name}_cm_PIGMENTS.png"))

    y_true_mix_idx = np.argmax(y_true_mix[:, 1:], axis=1)
    y_pred_mix_idx = np.argmax(y_pred_prob[:, n_pigments+1:], axis=1)
    cm_conf_mix = sk_confusion_matrix(y_true_mix_idx, y_pred_mix_idx, normalize="true")
    plot_confusion_matrix(cm_conf_mix, ["M1", "M2", "M3"], "Mixtures (Confusion)", os.path.join(cm_conf_dir, f"{name}_cm_MIXTURES.png"))

    # === COACTIVATION ===
    cm_coact_pig = soft_confusion_matrix(y_true_pig, y_pred_prob[:, :n_pigments], class_names=classes_pig)
    plot_confusion_matrix(cm_coact_pig, classes_pig, "Pigments (Coactivation)", os.path.join(cm_coact_dir, f"{name}_cm_PIGMENTS_SOFT.png"))

    cm_coact_mix = soft_confusion_matrix(y_true_mix, y_pred_prob[:, n_pigments:n_pigments+4], class_names=classes_mix)
    plot_confusion_matrix(cm_coact_mix, classes_mix, "Mixtures (Coactivation)", os.path.join(cm_coact_dir, f"{name}_cm_MIXTURES_SOFT.png"))

    # === METRICS SUMMARY ===
    metrics, descriptions = compute_metrics(y_true, y_pred_bin, y_pred_prob)
    summary_path = os.path.join(output_dir, f"{name}_combined_report.csv")
    with open(summary_path, "w", newline="") as f:
        f.write("metric,value,description\n")
        for k, v in metrics.items():
            f.write(f"{k},{v:.6f},{descriptions.get(k, '')}\n")

    print(f"[SAVE] Combined report -> {summary_path}")
    return summary_path




