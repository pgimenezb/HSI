import os
import argparse
import importlib
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import csv
from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.eval.report import generate_combined_report

import matplotlib.pyplot as plt
from hsi_lab.data.config import get_wavelength_config, spectral_ticks, variables


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
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
            n_test = min(per_mix, len(idxs)//2)
            n_val = 1 if len(idxs) > 2 else 0
            n_train = len(idxs) - n_test - n_val
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train+n_val])
        test_idx.extend(idxs[-n_test:])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

def stratified_split_70_15_15(df: pd.DataFrame, vars_: dict, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_pig = pigment_ids(df, vars_)
    idx_all = np.arange(len(df))
    idx_train, idx_tmp = train_test_split(idx_all, test_size=0.30, random_state=seed, stratify=y_pig)
    y_tmp = y_pig[idx_tmp]
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    return np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test)

# ─────────────────────────────────────────────────────────────────────────────
# Decoders (soft)
# ─────────────────────────────────────────────────────────────────────────────
def decode_pigment_and_group(y_like: np.ndarray, n_p: int, threshold=0.5):
    y_pig = y_like[:, :n_p]
    y_mix = y_like[:, n_p:n_p+4]

    results = []
    for i in range(len(y_like)):
        active_pigs = np.where(y_pig[i] > threshold)[0]
        active_mix = np.where(y_mix[i] > threshold)[0]
        labels = []
        for p in active_pigs:
            if 0 in active_mix:  # Pure activo
                labels.append(f"P{p+1:02d}_Pure")
            if any(m > 0 for m in active_mix):  # Cualquier mezcla
                labels.append(f"P{p+1:02d}_Mixture")
        results.append(";".join(labels) if labels else "")
    return np.array(results, dtype=object)


def decode_pigment_and_mix4(y_like: np.ndarray, n_p: int, threshold=0.5):
    y_pig = y_like[:, :n_p]
    y_mix = y_like[:, n_p:n_p+4]
    mix_names = ["Pure", "M1", "M2", "M3"]

    results = []
    for i in range(len(y_like)):
        active_pigs = np.where(y_pig[i] > threshold)[0]
        active_mix = np.where(y_mix[i] > threshold)[0]
        labels = [f"P{p+1:02d}_{mix_names[m]}" for p in active_pigs for m in active_mix]
        results.append(";".join(labels) if labels else "")
    return np.array(results, dtype=object)

# ─────────────────────────────────────────────────────────────────────────────
# Splits exportation (soft)
# ─────────────────────────────────────────────────────────────────────────────
def export_splits_csv(df: pd.DataFrame, y_true: np.ndarray, y_pred_prob: np.ndarray, idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray, out_dir: str, vars_: dict,) -> dict:
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


# ─────────────────────────────────────────────────────────────────────────────
# Cuotas por región/subregión (por archivo) + Igualado GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
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
        print("[INFO] No se aplican cuotas por región/subregión (diccionario vacío).")
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
    print(f"[INFO] Cuotas aplicadas por archivo con {quotas}")
    print(f"[INFO] Filas originales: {len(df)}, tras cuotas: {len(df_out)}")
    return df_out

# ─────────────────────────────────────────────────────────────────────────────
# Igualado global de pigmentos entre archivos
# ─────────────────────────────────────────────────────────────────────────────
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
# Region/Subregion usage
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train HSI models + export CMs y CSVs de splits.")
    p.add_argument("--outputs-dir", type=str, default=None)
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Soft confusion matrix (coactivación difusa)
# ─────────────────────────────────────────────────────────────────────────────
def soft_confusion_matrix(y_true, y_pred, class_names, normalize="col"):
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)

    for i in range(n_classes):
        for j in range(n_classes):
            cm[i, j] = np.mean(y_true[:, i] * y_pred[:, j])

    if normalize == "col":
        col_sums = cm.sum(axis=0, keepdims=True)
        cm = np.divide(cm, col_sums, where=col_sums != 0)
    elif normalize == "row":
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)
    elif normalize == "both":
        total = cm.sum()
        if total != 0:
            cm /= total

    return cm

# ─────────────────────────────────────────────────────────────────────────────
# Model summary
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Per-pigment and per-mixture performance
# ─────────────────────────────────────────────────────────────────────────────

def per_pigment_metrics(y_true, y_pred_prob, variables, base_out, name):
    import hsi_lab.eval.report_coactivation as rep
    n_p = int(variables["num_files"])
    mix_names = ["Pure", "M1", "M2", "M3"]
    rows = []
    for i in range(n_p):
        for j, mix in enumerate(mix_names):
            y_t = y_true[:, [i, n_p + j]]
            y_p = y_pred_prob[:, [i, n_p + j]]
            m, _ = rep.compute_metrics(y_t, (y_p > 0.5).astype(int), y_p)
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

# ─────────────────────────────────────────────────────────────────────────────
# Confusion and Coactivation Analysis (Top Confusions)
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# QUALITATIVE VISUALIZATION: Spectra comparison (multi-label context)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparative_spectra(
    X_test, y_test, y_pred_prob, base_out, name,
    pigment_indices=None, avg_blocks=5, threshold=0.5
):
    # 🧱 Ensure spectra folder exists inside model directory
    spectra_dir = os.path.join(base_out, "Spectra")
    os.makedirs(spectra_dir, exist_ok=True)

    # Infer dimensions
    X_test = np.array(X_test)
    if X_test.ndim == 2:
        X_test = X_test[..., np.newaxis]  # (samples, bands, 1)
    n_samples, n_bands, _ = X_test.shape
    bands = X_test.shape[1]
    wavelengths, _ = get_wavelength_config(variables["data_type"])
    wls = wavelengths[:bands]

    plt.xticks(
        [wls[i] for i in spectral_ticks["positions"]],
        spectral_ticks["labels"],
        rotation=90,
        fontsize=8
    )
    plt.xlabel("Data type, channel and wavelength (nm)")


    n_p = y_test.shape[1] - 4  # assuming last 4 columns are mixtures

    # Default pigments: first 3
    if pigment_indices is None:
        pigment_indices = range(min(3, n_p))

    for pigment_idx in pigment_indices:
        true_mask = y_test[:, pigment_idx] > threshold
        pred_mask = y_pred_prob[:, pigment_idx] > threshold

        if not np.any(true_mask):
            print(f"[WARN] Pigment P{pigment_idx+1:02d} has no true samples.")
            continue

        # Extract spectra sets
        spectra_true = X_test[true_mask, :, 0]
        spectra_pred = X_test[pred_mask, :, 0]

        # Helper: split spectra into averaged blocks
        def avg_blocks_spectra(spectra, blocks=avg_blocks):
            if len(spectra) == 0:
                return []
            block_size = max(1, len(spectra) // blocks)
            out = []
            for i in range(blocks):
                start, end = i * block_size, (i + 1) * block_size
                if start >= len(spectra):
                    break
                end = min(end, len(spectra))
                out.append(np.mean(spectra[start:end], axis=0))
            return out

        # Compute variability bands
        min_t, max_t = spectra_true.min(axis=0), spectra_true.max(axis=0)
        min_p, max_p = (
            (spectra_pred.min(axis=0), spectra_pred.max(axis=0))
            if len(spectra_pred) > 0 else (None, None)
        )

        plt.figure(figsize=(10, 6))
        plt.fill_between(wls, min_t, max_t, alpha=0.15, color="blue", label="True range")
        if min_p is not None:
            plt.fill_between(wls, min_p, max_p, alpha=0.15, color="orange", label="Predicted range")

        plt.plot(wls, np.mean(spectra_true, axis=0), color="blue", linewidth=2, label="True avg")
        if len(spectra_pred) > 0:
            plt.plot(wls, np.mean(spectra_pred, axis=0), color="orange", linewidth=2, label="Pred avg")

        plt.title(f"Pigment P{pigment_idx+1:02d} — average & range of spectra")
        plt.xlabel("Spectral band")
        plt.ylabel("Reflectance (a.u.)")
        plt.legend()
        plt.tight_layout()

        # Save plot
        out_path = os.path.join(spectra_dir, f"{name}_spectra_P{pigment_idx+1:02d}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[SAVE] Spectra plot -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY: Confusion/Coactivation Matrix Metrics Comparison (multi-label)
# ─────────────────────────────────────────────────────────────────────────────

def summarize_confusion_matrices(cm_dict, detailed_results, name, base_out):
    """
    Generate a CSV summarizing how each soft coactivation matrix behaves
    compared to model-level metrics (HammingAcc, F1, etc.).
    Designed for multi-label problems (e.g., pigments × mixtures).
    """

    def multilabel_metrics_from_coactivation(cm):
        """Compute soft metrics (accuracy, precision, recall, f1) from coactivation matrix."""
        cm = np.array(cm, dtype=np.float64)
        diag = np.diag(cm)
        row_sums = np.sum(cm, axis=1)
        col_sums = np.sum(cm, axis=0)
        precisions = np.divide(diag, col_sums, out=np.zeros_like(diag), where=col_sums != 0)
        recalls = np.divide(diag, row_sums, out=np.zeros_like(diag), where=row_sums != 0)
        f1s = np.divide(2 * precisions * recalls, precisions + recalls + 1e-8)
        metrics = {
            "soft_accuracy": np.nanmean(diag),
            "soft_precision_macro": np.nanmean(precisions),
            "soft_recall_macro": np.nanmean(recalls),
            "soft_f1_macro": np.nanmean(f1s)
        }
        return metrics

    # 🔹 Model-level reference metrics
    global_metrics, _ = detailed_results.get("global", ({}, {}))
    model_hamming = global_metrics.get("hamming_acc", np.nan)
    model_f1 = global_metrics.get("f1", np.nan)

    # 🔹 Output CSV path
    csv_out = os.path.join(base_out, "datasets", f"{name}_confusion_summary.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    header = [
        "Matrix Type",
        "Classes",
        "Soft Accuracy",
        "Soft Precision (macro)",
        "Soft Recall (macro)",
        "Soft F1 (macro)",
        "Model HammingAcc",
        "Model F1",
        "Δ(SoftAcc - HammingAcc)",
        "Δ(SoftF1 - ModelF1)",
        "Interpretation"
    ]

    rows = []

    for key, info in cm_dict.items():
        cm = info["matrix"]
        desc = info["desc"]
        classes = len(info["classes"])

        soft_metrics = multilabel_metrics_from_coactivation(cm)

        diff_acc = soft_metrics["soft_accuracy"] - model_hamming
        diff_f1 = soft_metrics["soft_f1_macro"] - model_f1

        if abs(diff_acc) < 0.02:
            interp = "Soft and model accuracies align closely."
        elif diff_acc < 0:
            interp = "Matrix shows lower consistency; possible inter-label confusion."
        else:
            interp = "Matrix consistency slightly higher; model may underpenalize overlaps."

        rows.append([
            desc,
            classes,
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

    # 🔸 Save CSV
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[SAVE] Confusion/Coactivation summary -> {csv_out}")
    return csv_out

# ─────────────────────────────────────────────────────────────────────────────
# Conclusions
# ─────────────────────────────────────────────────────────────────────────────

def write_conclusions(detailed_results, base_out, name):
    m_global, _ = detailed_results["global"]
    f1 = m_global.get("f1", 0)
    acc = m_global.get("strict_acc", 0)
    roc = m_global.get("roc_auc", 0)
    with open(os.path.join(base_out, f"{name}_conclusions.txt"), "w") as f:
        f.write("=== Conclusions ===\n")
        f.write(f"Global F1: {f1:.3f}\n")
        f.write(f"Strict accuracy: {acc:.3f}\n")
        f.write(f"ROC-AUC: {roc:.3f}\n\n")
        f.write("Model shows strong discrimination between pigments, but mixtures introduce variability.\n")
        f.write("Improving data balance across mixtures or applying spectral attention could help.\n")
    print(f"[SAVE] Conclusions -> {os.path.join(base_out, f'{name}_conclusions.txt')}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = args.outputs_dir or variables.get("outputs_dir") or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1️⃣ Cargar datos
    pr = HSIDataProcessor(variables)
    pr.load_h5_files()
    df_raw = pr.dataframe(mode="raw")

    # 2️⃣ Aplicar cuotas por región/subregión
    APPLY_QUOTAS = variables.get("apply_region_quotas", True)
    region_quotas = variables.get("region_row_quota", {}) or {}
    df_after_quotas = (
        apply_per_file_region_quotas(df_raw, variables)
        if (APPLY_QUOTAS and region_quotas)
        else df_raw
    )

    # 3️⃣ Igualado global por pigmento
    DO_EQUALIZE_GLOBAL = variables.get("equalize_across_files", True)
    df_used = (
        equalize_across_files_by_pigment(df_after_quotas)
        if DO_EQUALIZE_GLOBAL
        else df_after_quotas
    )

    # 4️⃣ Split del dataset
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

    # 5️⃣ Construir X/y
    X, y, input_len = build_Xy(df_used)
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    print(f"[DATA] input_len={input_len} | X_train={X_train.shape} | X_val={X_val.shape} | X_test={X_test.shape}")

    # 6️⃣ Entrenamiento
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    for name in model_names:
        print(f"\n[TRAIN] {name}")
        tune = import_model_trainer(name)
        res = tune(
            X_train, y_train, X_val, y_val,
            input_len=input_len, num_classes=y.shape[1],
            trials=args.trials or variables.get("trials"),
            epochs=args.epochs or variables.get("epochs"),
            batch_size=args.batch_size or variables.get("batch_size"),
            n_jobs=variables.get("optuna_n_jobs", 1),
            seed=variables.get("seed", 42),
        )

        model = res[0] if isinstance(res, tuple) else res
        y_pred_prob = model.predict(X_test, verbose=0)

        base_out = os.path.join(out_dir, f"{name}")
        os.makedirs(base_out, exist_ok=True)

        # === Guardar DataFrame usado ===
        df_used_path = os.path.join(base_out, f"{name}_dataframe_used.csv")
        df_used.to_csv(df_used_path, index=False)
        print(f"[SAVE] DataFrame -> {df_used_path}")

        # === Guardar uso de región/subregión ===
        region_usage_path = os.path.join(base_out, f"{name}_region_subregion_usage.csv")
        save_region_subregion_usage(df_raw, df_used, region_usage_path)

        # === Guardar splits ===
        y_pred_prob_full = np.full_like(y, np.nan, dtype=np.float32)
        y_pred_prob_full[idx_test] = y_pred_prob
        split_paths = export_splits_csv(
            df_used, y, y_pred_prob_full,
            idx_train, idx_val, idx_test,
            base_out, variables
        )
        for k, p in split_paths.items():
            print(f"[SAVE] Split CSV ({k}) -> {p}")

        # === UNIFIED REPORT ===
        generate_combined_report(
            y_true=y_test,
            y_pred_prob=y_pred_prob,
            n_pigments=int(variables["num_files"]),
            output_dir=os.path.join(base_out, "evaluation"),
            name=name
        )

        # === Extra summaries ===
        summarize_model(name, variables, X_train, y_train, model, base_out)
        per_pigment_metrics(y_test, y_pred_prob, variables, base_out, name)
        top_confusions(np.dot(y_test.T, y_pred_prob), [f"P{i+1}" for i in range(y_test.shape[1])], base_out, name)
        plot_comparative_spectra(X_test, y_test, y_pred_prob, base_out, name, pigment_indices=[0, 5, 10])
        write_conclusions({"global": rep.compute_metrics(y_test, (y_pred_prob > 0.5).astype(int), y_pred_prob)}, base_out, name)



        # === MÉTRICAS DETALLADAS (usando report.compute_detailed_metrics) ===

        # Calcula todas las métricas detalladas
        detailed_results = rep.compute_detailed_metrics(
            y_true=y_test,
            y_pred_prob=y_pred_prob,
            num_files=int(variables["num_files"]),
            threshold=0.5,
            verbose=True
        )

        # Save metrics CSV inside the 'datasets' folder
        csv_out_dir = os.path.join(base_out, "datasets")
        os.makedirs(csv_out_dir, exist_ok=True)
        metrics_csv = os.path.join(csv_out_dir, f"{name}_metrics_summary.csv")

        header = ["model_name", "scope", "metric_name", "value", "description"]
        write_header = not os.path.exists(metrics_csv)

        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)

            for scope, result in detailed_results.items():
                if result is None:
                    continue

                # Each result should be (metrics_dict, descriptions_dict)
                metrics, descriptions = result if isinstance(result, tuple) else (result, {})

                for key, value in metrics.items():
                    writer.writerow([
                        name,
                        scope,
                        key,
                        round(value, 6),
                        descriptions.get(key, "")
                    ])

        print(f"[SAVE] Detailed metrics report -> {metrics_csv}")

        # === SUMMARY: compare coactivation matrices vs model metrics ===
        cm_dict = {
            "pigments": {
                "matrix": cm_pig_soft,
                "classes": classes_pig,
                "desc": "Pigment coactivation"
            },
            "mixtures": {
                "matrix": cm_mix_soft,
                "classes": ["M1", "M2", "M3"],  
                "desc": "Mixture coactivation"
            },
            "pure_mix": {
                "matrix": cm_puremix_soft,
                "classes": classes_puremix_soft,
                "desc": "Pure vs Mixture coactivation"
            },
            "perPigMix": {
                "matrix": cm_soft_perPigMix,
                "classes": classes_soft,
                "desc": "Per-pigment×mixture coactivation"
            },
        }

        summary_csv = summarize_confusion_matrices(
            cm_dict=cm_dict,
            detailed_results=detailed_results,
            name=name,
            base_out=base_out
        )

        # === I. Model summary ===
        summarize_model(name, variables, X_train, y_train, model, base_out)

        # === III. Per-pigment and per-mixture metrics ===
        per_pigment_metrics(y_test, y_pred_prob, variables, base_out, name)

        # === IV. Top confusions (from Pigment × Mixture CM) ===
        top_confusions(cm_soft_perPigMix, classes_soft, base_out, name, top_k=15)

        # === QUALITATIVE VISUALIZATION ===
        plot_comparative_spectra(
            X_test=X_test,
            y_test=y_test,
            y_pred_prob=y_pred_prob,
            base_out=base_out,
            name=name,
            pigment_indices=[0, 5, 10],  # pigments to visualize
            avg_blocks=5,
            threshold=0.5
        )

        # === VI. Conclusions ===
        write_conclusions(detailed_results, base_out, name)

if __name__ == "__main__":
    main()
