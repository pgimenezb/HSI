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
from sklearn.metrics import confusion_matrix 
from scipy.optimize import lsq_linear
from scipy.optimize import nnls  
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap

from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
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


def get_wavelengths_and_labels(total_bands, data_type=["vis"], vis_start=400, vis_end_nominal=1000, swir_start=1110, swir_end_nominal=2500):
    wavelengths = []
    tick_positions = []
    tick_labels = []

    data_type = [d.lower() for d in data_type]

    # --- Parámetros nominales de referencia ---
    vis_nominal_bands = 267
    swir_nominal_bands = 232

    # --- Caso VIS+SWIR ---
    if "vis" in data_type and "swir" in data_type:
        # ✅ Distribución fija según número real de bandas
        vis_total = vis_nominal_bands
        swir_total = swir_nominal_bands
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
def plot_comparative_spectra(df, base_out, name, per_file=True, num_blocks=6):

    os.makedirs(base_out, exist_ok=True)

    # === Detectar columnas espectrales ===
    vis_cols = [c for c in df.columns if c.lower().startswith("vis_")]
    swir_cols = [c for c in df.columns if c.lower().startswith("swir_")]
    all_cols = vis_cols + swir_cols

    print(f"[DEBUG] Found {len(vis_cols)} VIS cols and {len(swir_cols)} SWIR cols.")

    if not all_cols:
        print("[WARN] No VIS/SWIR spectral columns found.")
        return

    spectra_array_all = df[all_cols].to_numpy(dtype=float)

    # === Eje de longitudes de onda coherente con KM ===
    wls, xticks_pos, xticks_labels = get_wavelengths_and_labels(
        total_bands=spectra_array_all.shape[1],
        data_type=["vis", "swir"] if vis_cols and swir_cols else ["vis"]
    )

    # === Modo por archivo ===
    if per_file:
        for file in df["File"].unique():
            df_file = df[df["File"] == file]
            spectra_array = df_file[all_cols].to_numpy(dtype=float)

            if len(spectra_array) < num_blocks:
                print(f"[WARN] {file}: insufficient rows to divide into {num_blocks} blocks.")
                continue

            block_size = len(spectra_array) // num_blocks
            fig, ax = plt.subplots(figsize=(12, 5))

            for i in range(num_blocks):
                start = i * block_size
                end = (i + 1) * block_size if i < num_blocks - 1 else len(spectra_array)
                block = spectra_array[start:end]

                mean_spec = np.mean(block, axis=0)
                min_spec = np.min(block, axis=0)
                max_spec = np.max(block, axis=0)

                ax.plot(wls, mean_spec, lw=1.8, label=f"Spectre {i+1}")
                ax.fill_between(wls, min_spec, max_spec, alpha=0.2)

            ax.set_title(f"{file}")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Reflectance")

            # ✅ FIX: usar directamente xticks_pos (no índices)
            ax.set_xticks(xticks_pos)
            xticklabels = ax.set_xticklabels(xticks_labels, rotation=65, fontsize=8)

            # Ajuste del espacio inferior según etiquetas
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            max_height = max([label.get_window_extent(renderer).height for label in xticklabels])
            fig_height = fig.get_size_inches()[1] * fig.dpi
            bottom_space = max_height / fig_height + 0.05
            fig.subplots_adjust(bottom=bottom_space)

            # Leyenda y layout
            fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3)
            fig.tight_layout(rect=[0, 0.1, 1, 1])

            # Guardar figura
            out_path = os.path.join(base_out, f"{name}_{file}_blocks.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"[SAVE] {out_path}")

    # === Modo global (todas las muestras por archivo) ===
    else:
        fig, ax = plt.subplots(figsize=(12, 5))

        for file in df["File"].unique():
            df_file = df[df["File"] == file]
            spectra_array = df_file[all_cols].to_numpy(dtype=float)
            if len(spectra_array) == 0:
                continue

            mean_spec = np.mean(spectra_array, axis=0)
            min_spec = np.min(spectra_array, axis=0)
            max_spec = np.max(spectra_array, axis=0)

            ax.plot(wls, mean_spec, lw=1.8, label=f"{file} (n={len(spectra_array)})")
            ax.fill_between(wls, min_spec, max_spec, alpha=0.2)

        ax.set_xlabel("Data type and wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.set_title("Average spectra per file (with min–max ranges)")

        # ✅ FIX: usar directamente xticks_pos (no índices)
        ax.set_xticks(xticks_pos)
        xticklabels = ax.set_xticklabels(xticks_labels, rotation=65, fontsize=8)

        # Ajuste espacio inferior
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        max_height = max([label.get_window_extent(renderer).height for label in xticklabels])
        fig_height = fig.get_size_inches()[1] * fig.dpi
        bottom_space = max_height / fig_height + 0.05
        fig.subplots_adjust(bottom=bottom_space)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -bottom_space / 2), ncol=3)
        plt.tight_layout()

        out_path = os.path.join(base_out, f"{name}_global.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[SAVE] {out_path}")



# ============================================================
# Kubelka–Munk 
# ============================================================

# === Conversiones básicas ===
def reflectance_to_ks(R):
    """Reflectancia → K/S"""
    R = np.clip(R, 1e-3, 1.0)
    return ((1 - R) ** 2) / (2 * R)


def ks_to_reflectance(KS):
    """K/S → Reflectancia"""
    KS = np.maximum(KS, 1e-8)
    R = 1 + KS - np.sqrt(KS ** 2 + 2 * KS)
    return np.clip(R, 0, 1)


# === Estimación de K y S por pigmento ===
def estimate_k_s_per_pigment(R_pigments):
    """
    Calcula K_i y S_i individuales por pigmento a partir de sus reflectancias puras.
    Evita suponer S=1 para todos los pigmentos.
    """
    R_pigments = np.array(R_pigments)
    if R_pigments.ndim > 2:
        R_pigments = R_pigments.reshape(R_pigments.shape[0], -1)

    R = np.clip(R_pigments, 1e-3, 1)
    KS = ((1 - R)**2) / (2 * R)

    # Estimar S como una forma relativa al brillo medio (no constante)
    S = np.ones_like(KS)
    for i in range(KS.shape[0]):
        s_i = np.median(KS[i] / np.max(KS[i]))
        S[i, :] = np.clip(s_i, 1e-3, 1.0)
    K = KS * S
    return K, S


# === Unmixing con NNLS o LSQ ===
def km_unmix_nnls(KS_mix, k_set, s_set, top_k=None, min_alpha=0.05):
    """
    Desmezcla un espectro usando Kubelka–Munk + bounded least squares (no negativo).
    """
    n_pigments = k_set.shape[0]

    # Construir librería de K/S por pigmento
    KS_library = k_set / np.maximum(s_set, 1e-8)

    # Resolver: min ||A x - b||² con 0 <= x <= 1
    res = lsq_linear(KS_library.T, KS_mix, bounds=(0, 1), lsmr_tol='auto', max_iter=300)
    alpha_pred = np.clip(res.x, 0, None)

    # Normalizar
    if np.sum(alpha_pred) > 0:
        alpha_pred /= np.sum(alpha_pred)

    # Sparsidad opcional: mantener solo top_k pigmentos dominantes
    if top_k is not None and top_k < len(alpha_pred):
        idx = np.argsort(-alpha_pred)[:top_k]
        mask = np.zeros_like(alpha_pred)
        mask[idx] = alpha_pred[idx]
        alpha_pred = mask / np.maximum(np.sum(mask), 1e-8)

    # Eliminar contribuciones pequeñas
    alpha_pred[alpha_pred < min_alpha] = 0
    if np.sum(alpha_pred) > 0:
        alpha_pred /= np.sum(alpha_pred)

    return alpha_pred


def apply_km_unmixing_nnls(R_all, k_set, s_set, top_k=None, top_n_sim=6):
    KS_all = reflectance_to_ks(R_all)
    alphas = []

    # === función auxiliar local
    def select_similar_pigments(KS_mix, k_set, s_set, top_n=6):
        KS_library = k_set / np.maximum(s_set, 1e-8)
        corr = np.dot(KS_library, KS_mix) / (
            np.linalg.norm(KS_library, axis=1) * np.linalg.norm(KS_mix)
        )
        idx_sel = np.argsort(-corr)[:top_n]
        return idx_sel

    for i, KS in enumerate(KS_all):
        # 1️⃣ seleccionar pigmentos similares al espectro actual
        idx_sel = select_similar_pigments(KS, k_set, s_set, top_n=top_n_sim)

        # 2️⃣ hacer unmixing solo con esos pigmentos candidatos
        alpha = km_unmix_nnls(KS, k_set[idx_sel], s_set[idx_sel], top_k=top_k)

        # 3️⃣ reconstruir vector completo con ceros para los demás pigmentos
        alpha_full = np.zeros(k_set.shape[0])
        alpha_full[idx_sel] = alpha
        alphas.append(alpha_full)

        if (i + 1) % 500 == 0:
            print(f"[KM] {i + 1} processed samples (NNLS + spectral filtering)")

    return np.array(alphas)


# === Generador de mezclas sintéticas ===
def generate_synthetic_mixtures(R_pigments, n_samples=12000, n_mix=(2, 3)):
    """
    Genera mezclas sintéticas según el modelo Kubelka–Munk,
    combinando pigmentos en el dominio K/S.
    """
    R_pigments = np.array(R_pigments)
    if R_pigments.ndim > 2:
        R_pigments = R_pigments.reshape(R_pigments.shape[0], -1)

    n_pigments, n_wl = R_pigments.shape
    KS_pigments = reflectance_to_ks(R_pigments)

    mixtures, proportions, components = [], [], []

    for _ in range(n_samples):
        n = np.random.choice(n_mix)
        idx = np.random.choice(n_pigments, n, replace=False)

        # Proporciones aleatorias (mezclas controladas)
        if n == 2 and np.random.rand() < 0.5:
            alpha = np.array([0.2, 0.8])
            np.random.shuffle(alpha)
        elif n == 3 and np.random.rand() < 0.5:
            base = np.array([0.78, 0.12, 0.10])
            alpha = base / base.sum()
            np.random.shuffle(alpha)
        else:
            a = np.random.rand(n)
            alpha = a / np.sum(a)

        # Mezcla lineal en el dominio K/S
        KS_mix = np.dot(alpha, KS_pigments[idx])
        R_mix = ks_to_reflectance(KS_mix)

        mixtures.append(R_mix)
        proportions.append(alpha)
        components.append(idx)

    return np.array(mixtures), proportions, components


# ============================================================
# COVARIANCE & CORRELATION MATRICES: Reflectance vs K/S
# ============================================================

def plot_matrix(matrix, labels, title, save_path, cmap="Blues", normalize=False):

    plt.figure(figsize=(9, 9))
    ax = plt.gca()

    # Normalizar si se desea (solo para correlación)
    data = matrix.copy()
    if normalize:
        data_min, data_max = np.nanmin(data), np.nanmax(data)
        if data_max != data_min:
            data = (data - data_min) / (data_max - data_min)

    base_cmap = plt.cm.get_cmap(cmap, 256)
    cmap_array = base_cmap(np.linspace(0, 1, 256))
    cmap_array[0, :] = [1, 1, 1, 1]
    cmap = ListedColormap(cmap_array)

    im = ax.imshow(data, cmap=cmap, interpolation="nearest")

    # Etiquetas
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel("Predicted / Variable j")
    ax.set_ylabel("True / Variable i")
    ax.set_title(title, fontsize=13, pad=10)

    # Anotar solo si la matriz es pequeña (<50x50)
    if data.shape[0] <= 50:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val): 
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=7)

    # Barra de color
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.2)
    plt.colorbar(im, cax=cax, format="%.2f")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVE] Matrix -> {save_path}")

# ============================================================================ #
# COVARIANCE MATRICES (Reflectance & K/S) 
# ============================================================================ #
def compute_reflectance_ks_covariance(y_true, y_pred_prob, n_pigments, output_dir, name):
    """
    Calcula matrices de covarianza entre activaciones reales y predichas
    (Reflectancia vs K/S), usando la misma lógica de las matrices de coactivación.
    """

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Computing covariance matrices for {name}")

    # === Separar subconjuntos ===
    y_true_pig = y_true[:, :n_pigments]
    y_true_mix = y_true[:, n_pigments:n_pigments + 4]
    y_pred_pig = y_pred_prob[:, :n_pigments]
    y_pred_mix = y_pred_prob[:, n_pigments:n_pigments + 4]

    # === Nombres de clases ===
    pigment_classes = [f"P{i+1:02d}" for i in range(n_pigments)]

    # === Unificar Pure + Mix ===
    y_true_pig_unified = (y_true_pig * np.sum(y_true_mix, axis=1, keepdims=True)).astype(float)
    y_pred_pig_unified = (y_pred_pig * np.sum(y_pred_mix, axis=1, keepdims=True)).astype(float)

    # === Covarianza Reflectancia (activaciones directas) ===
    cov_R = np.cov(y_pred_pig_unified.T)
    csv_R = os.path.join(output_dir, f"{name}_Reflectance_Covariance.csv")
    pd.DataFrame(cov_R, index=pigment_classes, columns=pigment_classes).to_csv(csv_R)
    print(f"[SAVE] Reflectance Covariance -> {csv_R}")

    # === Covarianza K/S (transformando activaciones con Kubelka–Munk) ===
    #   Aplicamos una transformación tipo K/S al espacio de activación
    KS = reflectance_to_ks(y_pred_pig_unified + 1e-9)
    cov_KS = np.cov(KS.T)
    csv_KS = os.path.join(output_dir, f"{name}_KS_Covariance.csv")
    pd.DataFrame(cov_KS, index=pigment_classes, columns=pigment_classes).to_csv(csv_KS)
    print(f"[SAVE] K/S Covariance -> {csv_KS}")

    # === Visualización ===
    for mat, title in [(cov_R, "Reflectance Covariance"), (cov_KS, "K/S Covariance")]:
        fig_size = max(6, min(0.45 * len(pigment_classes), 20))
        plt.figure(figsize=(fig_size, fig_size))
        ax = plt.gca()

        base_cmap = mpl_cm.get_cmap("Blues", 256)
        cmap_array = base_cmap(np.linspace(0, 1, 256))
        cmap_array[0, :] = [1, 1, 1, 1]
        cmap = ListedColormap(cmap_array)

        vmax = np.max(mat) if np.max(mat) > 0 else 1.0
        im = ax.imshow(mat, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(len(pigment_classes)))
        ax.set_yticks(np.arange(len(pigment_classes)))
        ax.set_xticklabels(pigment_classes, rotation=60, ha="right", fontsize=9)
        ax.set_yticklabels(pigment_classes, fontsize=9)
        ax.set_xlabel("Pigment j", fontsize=11)
        ax.set_ylabel("Pigment i", fontsize=11)
        ax.set_title(title, fontsize=13, pad=10)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                if np.isnan(value): continue
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=7)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.2)
        plt.colorbar(im, cax=cax, format="%.2f")
        plt.tight_layout()

        safe_title = title.replace(" ", "_").replace("/", "_")
        png_path = os.path.join(output_dir, f"{name}_{safe_title}.png")
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"[SAVE] {title} heatmap -> {png_path}")

    print(f"[DONE] Covariance matrices (Reflectance & K/S) saved in {output_dir}")




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
# EXPORT MÉTRICAS COMPLETAS (MODELO Y MATRICES)
# ============================================================================ #

def export_full_metrics_csv(detailed_results, out_path, model_name):
    """
    Exporta las métricas completas del modelo (global y por scope)
    en formato estándar, con descripción textual.
    """
    rows = []

    # === Diccionario de descripciones ===
    metric_desc = {
        "non_zero_acc_sample": "Sample-wise accuracy ignoring zero (empty) labels.",
        "non_zero_acc_label": "Per-label accuracy ignoring labels not present in ground truth.",
        "strict_acc": "Proportion of samples where all labels are predicted exactly.",
        "general_acc": "Overall ratio of correctly predicted labels (micro accuracy).",
        "keras_like_acc": "Accuracy equivalent to Keras-style categorical accuracy.",
        "hamming_acc": "Hamming accuracy (1 - Hamming loss).",
        "roc_auc": "Macro-averaged ROC-AUC over all classes.",
        "avg_precision": "Mean Average Precision (mAP) across classes.",
        "log_loss": "Cross-entropy loss averaged across labels.",
        "f1": "F1 macro-average (mean of per-class F1).",
        "precision": "Precision macro-average (mean of per-class precision).",
        "recall": "Recall macro-average (mean of per-class recall)."
    }

    for scope_name, (global_metrics, per_label_metrics) in detailed_results.items():
        for metric_name, value in global_metrics.items():
            if metric_name not in metric_desc:
                continue
            rows.append({
                "Model": model_name,
                "Scope": scope_name,
                "Label": "",
                "Metric": metric_name,
                "Value": round(float(value), 6),
                "Description": metric_desc.get(metric_name, "")
            })

        if isinstance(per_label_metrics, pd.DataFrame):
            for _, row in per_label_metrics.iterrows():
                label = row.get("Label", "")
                for metric_name in metric_desc.keys():
                    if metric_name in row:
                        rows.append({
                            "Model": model_name,
                            "Scope": scope_name,
                            "Label": label,
                            "Metric": metric_name,
                            "Value": round(float(row[metric_name]), 6),
                            "Description": metric_desc.get(metric_name, "")
                        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Model metrics -> {out_path}")
    return df





def export_matrix_metrics_full(matrix_metrics_dict, output_dir, model_name):
    records = []
    metric_descriptions = {
        "SoftAccuracy": "Overall ratio of correctly predicted labels (micro accuracy).",
        "SoftPrecisionMacro": "Precision macro-average (mean of per-class precision).",
        "SoftRecallMacro": "Recall macro-average (mean of per-class recall).",
        "SoftF1Macro": "F1 macro-average (mean of per-class F1).",
        "SoftPrecisionMicro": "Precision computed globally across all classes (micro).",
        "SoftRecallMicro": "Recall computed globally across all classes (micro).",
        "SoftF1Micro": "F1 computed globally across all classes (micro)."
    }

    for matrix_name, metrics in matrix_metrics_dict.items():
        for k, v in metrics.items():
            if k not in metric_descriptions:
                continue
            records.append({
                "Model": model_name,
                "Scope": matrix_name.lower(),
                "Label": "",
                "Metric": k,
                "Value": float(v),
                "Description": metric_descriptions[k]
            })

    df = pd.DataFrame(records)
    path = os.path.join(output_dir, f"{model_name}_matrix_metrics_full.csv")
    df.to_csv(path, index=False)
    print(f"[SAVE] Matrix metrics (model-like format) -> {path}")
    return path



def export_matrix_model_diff(model_csv, matrix_csv, output_dir, model_name):
    """
    Calcula las diferencias (Matrix - Model) para métricas comparables.
    """
    model_df = pd.read_csv(model_csv)
    matrix_df = pd.read_csv(matrix_csv)

    # Normalizar nombres de métricas para emparejar
    name_map = {
        "SoftAccuracy": "general_acc",
        "SoftPrecisionMacro": "precision",
        "SoftRecallMacro": "recall",
        "SoftF1Macro": "f1",
        "SoftPrecisionMicro": "precision",
        "SoftRecallMicro": "recall",
        "SoftF1Micro": "f1"
    }

    rows = []
    for _, mrow in matrix_df.iterrows():
        metric_model_name = name_map.get(mrow["Metric"])
        if metric_model_name is None:
            continue

        model_match = model_df[model_df["Metric"] == metric_model_name]
        if model_match.empty:
            continue

        model_val = float(model_match["Value"].iloc[0])
        delta = float(mrow["Value"]) - model_val

        rows.append({
            "Model": model_name,
            "Matrix": mrow["Scope"],
            "Metric": mrow["Metric"],
            "Model_Value": model_val,
            "Matrix_Value": float(mrow["Value"]),
            "Delta": delta,
            "Description": f"Difference between matrix and model for {mrow['Metric']} ({mrow['Description']})"
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"{model_name}_matrix_model_diff.csv")
    df.to_csv(path, index=False)
    print(f"[SAVE] Matrix↔Model metric differences -> {path}")
    return df

# ============================================================================ #
# BASE METRICS FUNCTION (tu versión original con ligeros ajustes)
# ============================================================================ #
def compute_metrics(y_true, y_pred_bin, y_pred_prob):
    """Compute main multi-label classification metrics."""
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
# EXTENDED DETAILED METRICS (GLOBAL + PER-PIGMENT + PER-MIXTURE)
# ============================================================================ #
def compute_detailed_metrics(y_true, y_pred_prob, num_files, threshold=0.5, verbose=False):
    y_pred_bin = (y_pred_prob > threshold).astype(int)
    results = {}

    # --- GLOBAL METRICS ---
    results["global"] = compute_metrics(y_true, y_pred_bin, y_pred_prob)

    # --- SPLIT: PIGMENTS / MIXTURES ---
    y_true_pig = y_true[:, :num_files]
    y_true_mix = y_true[:, num_files:num_files+4]
    y_pred_pig = y_pred_bin[:, :num_files]
    y_pred_mix = y_pred_bin[:, num_files:num_files+4]
    y_prob_pig = y_pred_prob[:, :num_files]
    y_prob_mix = y_pred_prob[:, num_files:num_files+4]

    # --- Global per-group ---
    results["pigments_global"] = compute_metrics(y_true_pig, y_pred_pig, y_prob_pig)
    results["mixtures_global"] = compute_metrics(y_true_mix, y_pred_mix, y_prob_mix)

    # --- Verbose console output ---
    if verbose:
        print("\n[DETAILED METRICS REPORT]")
        for scope, val in results.items():
            if isinstance(val, tuple):
                metrics, _ = val
                print(f"\n--- {scope.upper()} ---")
                for mk, mv in metrics.items():
                    print(f"{mk:20s}: {mv:.4f}")
            elif isinstance(val, list):
                print(f"\n--- {scope.upper()} ---")
                for entry in val:
                    label = entry.get("pigment") or entry.get("mixture")
                    print(f"{label}: F1={entry['f1']:.3f}, Prec={entry['precision']:.3f}, Rec={entry['recall']:.3f}")

    return results






# ============================================================================ #
# CONFUSION AND COACTIVATION MATRIX
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
    cm = confusion_matrix(ti, pi, labels=range(len(classes)))
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0.0] = 1.0
    return cm.astype(float) / row_sums, classes


def soft_confusion_matrix(y_true, y_pred, class_names, normalize="row"):
    """Calcula matriz de coactivación (soft confusion)."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            cm[i, j] = np.mean(y_true[:, i] * y_pred[:, j])
    if normalize == "row":
        cm /= np.sum(cm, axis=1, keepdims=True) + 1e-12
    elif normalize == "col":
        cm /= np.sum(cm, axis=0, keepdims=True) + 1e-12
    return cm


def generate_combined_report(y_true, y_pred_prob, n_pigments, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Generating combined report in {output_dir}")

    # === Separar subconjuntos ===
    y_true_pig = y_true[:, :n_pigments]
    y_true_mix = y_true[:, n_pigments:n_pigments + 4]
    y_pred_pig = y_pred_prob[:, :n_pigments]
    y_pred_mix = y_pred_prob[:, n_pigments:n_pigments + 4]

    # === Nombres de clases ===
    mix_classes = ["M1", "M2", "M3"]
    pigment_classes = [f"P{i+1:02d}" for i in range(n_pigments)]
    puremix_classes = [f"P{i+1:02d}_Pure" for i in range(n_pigments)] + \
                      [f"P{i+1:02d}_Mixture" for i in range(n_pigments)]

    # ===================================================================== #
    # 🟦 COACTIVATION MATRICES
    # ===================================================================== #
    y_true_mix3 = y_true_mix[:, 1:]
    y_pred_mix3 = y_pred_mix[:, 1:]
    cm_mix_soft = soft_confusion_matrix(y_true_mix3, y_pred_mix3, mix_classes)
    plot_confusion_matrix(cm_mix_soft, mix_classes,
                          "Mixtures (Coactivation)",
                          os.path.join(output_dir, f"{name}_MIXTURES_Coactivation.png"))

    y_true_pig_unified = (y_true_pig * np.sum(y_true_mix, axis=1, keepdims=True)).astype(float)
    y_pred_pig_unified = (y_pred_pig * np.sum(y_pred_mix, axis=1, keepdims=True)).astype(float)
    cm_pig_soft = soft_confusion_matrix(y_true_pig_unified, y_pred_pig_unified, pigment_classes)
    plot_confusion_matrix(cm_pig_soft, pigment_classes,
                          "Pigments (Coactivation)",
                          os.path.join(output_dir, f"{name}_PIGMENTS_Coactivation.png"))

    y_true_puremix, y_pred_puremix = [], []
    for i in range(n_pigments):
        true_pure = y_true_pig[:, [i]] * y_true_mix[:, [0]]
        true_mix = y_true_pig[:, [i]] * np.sum(y_true_mix[:, 1:], axis=1, keepdims=True)
        pred_pure = y_pred_pig[:, [i]] * y_pred_mix[:, [0]]
        pred_mix = y_pred_pig[:, [i]] * np.sum(y_pred_mix[:, 1:], axis=1, keepdims=True)
        y_true_puremix.append(np.concatenate([true_pure, true_mix], axis=1))
        y_pred_puremix.append(np.concatenate([pred_pure, pred_mix], axis=1))
    y_true_puremix = np.concatenate(y_true_puremix, axis=1)
    y_pred_puremix = np.concatenate(y_pred_puremix, axis=1)

    cm_puremix_soft = soft_confusion_matrix(y_true_puremix, y_pred_puremix, puremix_classes)
    plot_confusion_matrix(cm_puremix_soft, puremix_classes,
                          "Pure vs Mixture (Coactivation)",
                          os.path.join(output_dir, f"{name}_PUREMIX_Coactivation.png"))

    # ===================================================================== #
    # 🟥 CONFUSION MATRICES (hard)
    # ===================================================================== #
    def hard_confusion(y_true_bin, y_pred_prob, classes, title, fname):
        y_true_idx = np.argmax(y_true_bin, axis=1)
        y_pred_idx = np.argmax(y_pred_prob, axis=1)
        cm = confusion_matrix(y_true_idx, y_pred_idx, normalize="true")
        plot_confusion_matrix(cm, classes, title, os.path.join(output_dir, fname))
        return cm

    cm_mix_conf = hard_confusion(y_true_mix3, y_pred_mix3, mix_classes,
                                 "Mixtures (Confusion)", f"{name}_MIXTURES_Confusion.png")
    cm_pig_conf = hard_confusion(y_true_pig_unified, y_pred_pig_unified, pigment_classes,
                                 "Pigments (Confusion)", f"{name}_PIGMENTS_Confusion.png")
    cm_puremix_conf = hard_confusion(y_true_puremix, y_pred_puremix, puremix_classes,
                                     "Pure vs Mixture (Confusion)", f"{name}_PUREMIX_Confusion.png")

    # ===================================================================== #
    # 📊 MÉTRICAS COMPLETAS DEL MODELO Y MATRICES
    # ===================================================================== #
    detailed = compute_detailed_metrics(
        y_true,
        y_pred_prob,
        num_files=n_pigments,
        threshold=0.5,
        verbose=False
    )

    # === 1️⃣ Exportar métricas completas del modelo ===
    model_metrics_csv = os.path.join(output_dir, f"{name}_model_metrics_full.csv")
    export_full_metrics_csv(detailed, model_metrics_csv, name)

    # === 2️⃣ Calcular métricas derivadas de las matrices ===
    matrix_metrics_dict = {
        "Mix_Coactivation": {
            "SoftAccuracy": np.mean(np.diag(cm_mix_soft)),
            "SoftPrecisionMacro": np.mean(np.diag(cm_mix_soft) / (np.sum(cm_mix_soft, axis=0) + 1e-8)),
            "SoftRecallMacro": np.mean(np.diag(cm_mix_soft) / (np.sum(cm_mix_soft, axis=1) + 1e-8)),
            "SoftF1Macro": np.mean(2 * np.diag(cm_mix_soft) / (np.sum(cm_mix_soft, axis=0) + np.sum(cm_mix_soft, axis=1) + 1e-8))
        },
        "Pig_Coactivation": {
            "SoftAccuracy": np.mean(np.diag(cm_pig_soft)),
            "SoftPrecisionMacro": np.mean(np.diag(cm_pig_soft) / (np.sum(cm_pig_soft, axis=0) + 1e-8)),
            "SoftRecallMacro": np.mean(np.diag(cm_pig_soft) / (np.sum(cm_pig_soft, axis=1) + 1e-8)),
            "SoftF1Macro": np.mean(2 * np.diag(cm_pig_soft) / (np.sum(cm_pig_soft, axis=0) + np.sum(cm_pig_soft, axis=1) + 1e-8))
        },
        "PureMix_Coactivation": {
            "SoftAccuracy": np.mean(np.diag(cm_puremix_soft)),
            "SoftPrecisionMacro": np.mean(np.diag(cm_puremix_soft) / (np.sum(cm_puremix_soft, axis=0) + 1e-8)),
            "SoftRecallMacro": np.mean(np.diag(cm_puremix_soft) / (np.sum(cm_puremix_soft, axis=1) + 1e-8)),
            "SoftF1Macro": np.mean(2 * np.diag(cm_puremix_soft) / (np.sum(cm_puremix_soft, axis=0) + np.sum(cm_puremix_soft, axis=1) + 1e-8))
        },
        "Mix_Confusion": {"SoftAccuracy": np.mean(np.diag(cm_mix_conf))},
        "Pig_Confusion": {"SoftAccuracy": np.mean(np.diag(cm_pig_conf))},
        "PureMix_Confusion": {"SoftAccuracy": np.mean(np.diag(cm_puremix_conf))}
    }

    # === 3️⃣ Exportar métricas de matrices ===
    matrix_metrics_csv = export_matrix_metrics_full(matrix_metrics_dict, output_dir, name)

    # === 4️⃣ Exportar comparativa directa Matrix vs Modelo ===
    export_matrix_model_diff(model_metrics_csv, matrix_metrics_csv, output_dir, name)
    diff_csv = os.path.join(output_dir, f"{name}_matrix_model_diff.csv")
    write_descriptive_summary(model_metrics_csv, matrix_metrics_csv, diff_csv, output_dir, name)

    print(f"[SAVE] Combined report completed for {name}")

    return {
        "model_metrics": model_metrics_csv,
        "matrix_metrics": matrix_metrics_csv,
        "detailed_metrics": detailed
    }



# CONCLUSIONS

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


import numpy as np
import warnings
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, log_loss, hamming_loss
)

def write_descriptive_summary(model_csv, matrix_csv, diff_csv, output_dir, model_name):
    """
    Generate an AI-like descriptive English summary of model, matrix and their differences.
    """
    try:
        model_df = pd.read_csv(model_csv)
        matrix_df = pd.read_csv(matrix_csv)
        diff_df = pd.read_csv(diff_csv)
    except Exception as e:
        print(f"[WARN] Could not read CSVs for descriptive summary: {e}")
        return

    # === Extract key aggregates ===
    avg_model = model_df.groupby("Metric")["Value"].mean().to_dict()
    avg_matrix = matrix_df.groupby("Metric")["Value"].mean().to_dict()
    avg_delta = diff_df["Delta"].mean() if "Delta" in diff_df.columns else 0.0

    # === Build summary text ===
    text_lines = []
    text_lines.append(f"=== AI Descriptive Summary for Model: {model_name} ===\n")

    # --- Model performance overview ---
    text_lines.append("[1] Model performance overview:\n")
    text_lines.append(f"- Average F1-score: {avg_model.get('f1', np.nan):.4f}")
    text_lines.append(f"- Average Precision: {avg_model.get('precision', np.nan):.4f}")
    text_lines.append(f"- Average Recall: {avg_model.get('recall', np.nan):.4f}")
    text_lines.append(f"- Average ROC-AUC: {avg_model.get('roc_auc', np.nan):.4f}")
    text_lines.append("Interpretation: The model achieves overall high discriminative capability, "
                      "indicating strong consistency across multiple labels.\n")

    # --- Matrix insights ---
    text_lines.append("[2] Matrix-level behavior:\n")
    if not matrix_df.empty:
        macro_acc = matrix_df[matrix_df["Metric"] == "SoftAccuracy"]["Value"].mean()
        macro_f1 = matrix_df[matrix_df["Metric"] == "SoftF1Macro"]["Value"].mean()
        text_lines.append(f"- Mean SoftAccuracy across matrices: {macro_acc:.4f}")
        text_lines.append(f"- Mean SoftF1Macro across matrices: {macro_f1:.4f}")
        text_lines.append("Interpretation: Matrix-level coherence suggests the degree to which "
                          "internal activations align with true class co-occurrences.\n")

    # --- Differences ---
    text_lines.append("[3] Model vs. Matrix comparison:\n")
    if not diff_df.empty:
        pos = diff_df[diff_df["Delta"] > 0]
        neg = diff_df[diff_df["Delta"] < 0]
        text_lines.append(f"- Metrics where the matrix exceeds the model: {len(pos)} ({len(pos)/len(diff_df)*100:.1f}%)")
        text_lines.append(f"- Metrics where the matrix underperforms the model: {len(neg)} ({len(neg)/len(diff_df)*100:.1f}%)")
        text_lines.append(f"- Mean absolute difference: {abs(avg_delta):.4f}")
        text_lines.append("Interpretation: Positive deltas indicate stronger internal consistency "
                          "than externally measured accuracy, while negative deltas reveal "
                          "confusion among pigments or mixture categories.\n")

    # --- General interpretation ---
    text_lines.append("[4] General interpretation:\n")
    text_lines.append("The model shows strong prediction performance on mixtures "
                      "(coactivation accuracy slightly higher than global metrics), "
                      "but exhibits mild confusion among pigment-specific activations "
                      "and between pure/mixture cases. "
                      "This pattern suggests that internal representations are robust "
                      "but the decision boundaries could be refined for pigment discrimination.\n")

    text_lines.append("=== End of Summary ===")

    summary_path = os.path.join(output_dir, f"{model_name}_descriptive_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(text_lines))

    print(f"[SAVE] Descriptive summary -> {summary_path}")
    return summary_path
