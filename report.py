# ============================================================================ #
# CLEAN REPORT MODULE (minimal required for orquestor)
# ============================================================================ #

import os
import importlib
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, log_loss, hamming_loss,
    confusion_matrix
)
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
# DATASET BUILDING AND SPLITTING
# ============================================================================ #
import ast
import numpy as np
import pandas as pd

def build_Xy(df: pd.DataFrame):
    """Construye X (espectros) e y (vectores multietiqueta) a partir del DataFrame."""
    spec_cols = [c for c in df.columns if c.startswith(("vis_", "swir_"))]
    if not spec_cols:
        raise ValueError("No spectral columns starting with 'vis_' or 'swir_' found.")

    vis_cols = sorted([c for c in spec_cols if c.startswith("vis_")], key=lambda c: int(c.split("_")[1]))
    swir_cols = sorted([c for c in spec_cols if c.startswith("swir_")], key=lambda c: int(c.split("_")[1]))
    spec_cols_sorted = vis_cols + swir_cols

    X = df[spec_cols_sorted].astype(np.float32).fillna(0.0).values[..., np.newaxis]

    # --- FIX: Ensure Multi is a numeric array, not a string ---
    def parse_multi(v):
        if isinstance(v, str):
            try:
                # Convierte cadena tipo "[0,1,0,...]" en lista real
                return np.array(ast.literal_eval(v), dtype=np.float32)
            except Exception:
                # Si no se puede, devuelve ceros del tamaño más probable
                return np.zeros(21, dtype=np.float32)
        elif isinstance(v, (list, np.ndarray)):
            return np.array(v, dtype=np.float32)
        else:
            return np.zeros(21, dtype=np.float32)

    y = np.array([parse_multi(v) for v in df["Multi"]], dtype=np.float32)

    return X, y, X.shape[1]



def pigment_ids(df: pd.DataFrame, vars_: dict) -> np.ndarray:
    n_p = int(vars_["num_files"])
    return np.array([int(np.argmax(np.asarray(v, dtype=np.float32)[:n_p])) for v in df["Multi"]], dtype=int)


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
            n_test = min(per_mix, len(idxs) // 2)
            n_val = 1 if len(idxs) > 2 else 0
            n_train = len(idxs) - n_test - n_val
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[-n_test:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def stratified_split_70_15_15(df: pd.DataFrame, vars_: dict, seed: int = 42):
    y_pig = pigment_ids(df, vars_)
    idx_all = np.arange(len(df))
    idx_train, idx_tmp = train_test_split(idx_all, test_size=0.30, random_state=seed, stratify=y_pig)
    y_tmp = y_pig[idx_tmp]
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    return np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test)

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



# ======================================================================
# 🧩 UNIVERSAL STRATIFIED SPLIT (for pure and multilabel datasets)
# ======================================================================
def stratified_balanced_split(df, test_size=0.15, val_size=0.15, seed=42):
    """Performs stratified split for both single-label and multilabel pigment datasets."""
    from sklearn.model_selection import train_test_split

    if "Pigment Index" in df.columns:
        # === PURE PIGMENTS ===
        strat_key = df["File"].astype(str) + "_" + df["Pigment Index"].astype(str)
        stratify_data = strat_key
        print("[INFO] Detected pure pigments dataset — stratifying by File + Pigment Index")

        idx_all = np.arange(len(df))
        idx_train, idx_test = train_test_split(
            idx_all, test_size=test_size, random_state=seed, stratify=stratify_data
        )
        idx_train, idx_val = train_test_split(
            idx_train, test_size=val_size / (1 - test_size), random_state=seed,
            stratify=stratify_data.iloc[idx_train]
        )
        return np.array(idx_train), np.array(idx_val), np.array(idx_test)

    elif "Multi" in df.columns:
        # === MULTILABEL / MIXTURE DATASETS ===
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        y = np.vstack(df["Multi"].apply(
            lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
        ))
        print("[INFO] Detected multilabel dataset — using MultilabelStratifiedKFold")

        n_splits = max(3, int(round(1 / test_size)))  # Ensure reasonable number of folds
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        idx_train, idx_test = next(mskf.split(np.zeros(len(y)), y))
        idx_train, idx_val = train_test_split(
            idx_train, test_size=val_size / (1 - test_size), random_state=seed
        )
        return np.array(idx_train), np.array(idx_val), np.array(idx_test)

    else:
        # === FALLBACK (no stratification possible) ===
        print("[WARN] No Pigment Index or Multi column found — using random split.")
        idx_all = np.arange(len(df))
        idx_train, idx_test = train_test_split(idx_all, test_size=test_size, random_state=seed)
        idx_train, idx_val = train_test_split(idx_train, test_size=val_size / (1 - test_size), random_state=seed)
        return np.array(idx_train), np.array(idx_val), np.array(idx_test)


# ======================================================================
# ⚖️ POST-SPLIT REBALANCING BY PIGMENT (according to region_row_quota)
# ======================================================================
def rebalance_by_pigment(df, y, target_per_pigment, seed=42):
    """
    Equalize number of samples per pigment (multi-label aware).
    Each pigment will have at most 'target_per_pigment' samples.
    """
    np.random.seed(seed)
    idx_selected = []

    n_pigs = y.shape[1]
    for p in range(n_pigs):
        idx_p = np.where(y[:, p] == 1)[0]
        if len(idx_p) == 0:
            continue
        if len(idx_p) > target_per_pigment:
            idx_p = np.random.choice(idx_p, target_per_pigment, replace=False)
        idx_selected.extend(idx_p)

    return np.array(sorted(set(idx_selected)))



# ============================================================================ #
# METRICS + CONFUSION MATRIX
# ============================================================================ #
def compute_metrics(y_true, y_pred_bin, y_pred_prob):
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
        "non_zero_acc_sample": "Fraction of samples with at least one correct label.",
        "non_zero_acc_label":  "Fraction of labels correctly predicted in any sample.",
        "strict_acc":          "All labels must match exactly.",
        "general_acc":         "Overall micro accuracy.",
        "keras_like_acc":      "Mean sample-wise accuracy.",
        "hamming_acc":         "1 - Hamming loss.",
        "roc_auc":             "ROC-AUC (micro).",
        "avg_precision":       "Average precision (micro).",
        "log_loss":            "Cross-entropy loss.",
        "f1":                  "F1-score (micro).",
        "precision":           "Precision (micro).",
        "recall":              "Recall (micro).",
    }
    return metrics, descriptions

# ============================================================================ #
# MULTILABEL CONFUSION MATRIX (FINAL)
# ============================================================================ #

def plot_pigment_confusion_matrix(y_true, y_pred, pigment_names, save_path,
                                  title="Confusion Matrix (per pigment)"):

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    n_classes = len(pigment_names)

    # === 1️⃣ Calcular matriz de confusión "por pigmento" ===
    cm = np.zeros((n_classes, n_classes), dtype=float)

    # Cada pigmento verdadero (fila i)
    for i in range(n_classes):
        true_mask = y_true[:, i] == 1
        if not np.any(true_mask):
            continue

        # Sumar todas las predicciones (columna j) para los casos donde pigmento i estaba activo
        cm[i, :] = y_pred[true_mask].sum(axis=0)

    # === 2️⃣ Normalizar por fila (proporción 0–1) ===
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm = np.nan_to_num(cm)

    # === 3️⃣ Guardar CSV ===
    df_cm = pd.DataFrame(cm, index=pigment_names, columns=pigment_names)
    csv_path = os.path.splitext(save_path)[0] + "_confusion.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_cm.to_csv(csv_path)
    print(f"[SAVE] Confusion matrix CSV -> {csv_path}")

    # === 4️⃣ Graficar ===
    fig_size = max(8, min(0.45 * n_classes, 18))
    plt.figure(figsize=(fig_size, fig_size))
    ax = plt.gca()

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(pigment_names, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(pigment_names, fontsize=9)
    ax.set_xlabel("Predicted Pigment", fontsize=12)
    ax.set_ylabel("True Pigment", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)

    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=7)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVE] Confusion matrix image -> {save_path}")



def plot_true_vs_predicted_proportions(csv_path, pigment_names, save_path=None, title=None):

    df = pd.read_csv(csv_path)

    # --- Parsear arrays de Multi ---
    def parse_vec(s):
        return np.array([int(x) for x in str(s).split(";")])

    y_true = np.vstack(df["True_Multi"].apply(parse_vec))
    y_pred = np.vstack(df["Pred_Multi"].apply(parse_vec))

    # --- Calcular proporciones verdaderas (si existen columnas w1 y w2) ---
    true_props = np.zeros((len(df), len(pigment_names)))
    if "w1_true" in df.columns and "w2_true" in df.columns:
        for i, row in df.iterrows():
            active = np.where(y_true[i] == 1)[0]
            if len(active) == 1:
                true_props[i, active[0]] = 1.0
            elif len(active) == 2:
                true_props[i, active[0]] = row.get("w1_true", 0.0)
                true_props[i, active[1]] = row.get("w2_true", 0.0)

    # --- Calcular proporciones predichas (basado en activaciones) ---
    pred_props = y_pred / np.maximum(y_pred.sum(axis=1, keepdims=True), 1)
    pred_props = np.nan_to_num(pred_props)

    # --- Media por pigmento ---
    mean_true = true_props.mean(axis=0)
    mean_pred = pred_props.mean(axis=0)

    # --- Plot ---
    x = np.arange(len(pigment_names))
    width = 0.4
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, mean_true, width, label="True", color="steelblue")
    plt.bar(x + width/2, mean_pred, width, label="Predicted", color="orange")
    plt.xticks(x, pigment_names, rotation=70)
    plt.ylabel("Mean proportion")
    plt.title(title or "True vs Predicted Pigment Proportions")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[SAVE] Proportion bar chart → {save_path}")
    else:
        plt.show()




# ======================================================================
# REPORT: Pigment dominance balance vs model recall
# ======================================================================
def analyze_balance_vs_recall(
    mixtures_csv: str,
    predictions_csv: str,
    out_dir: str,
    pigment_names: list = None,
):
    """Analyzes how pigment weight dominance (w1/w2) affects recall in predictions."""

    os.makedirs(out_dir, exist_ok=True)

    print(f"[LOAD] Synthetic mixtures -> {mixtures_csv}")
    df_mix = pd.read_csv(mixtures_csv)

    print(f"[LOAD] Model predictions -> {predictions_csv}")
    df_pred = pd.read_csv(predictions_csv)

    # === Extract pigment pairs and weights ===
    pairs = df_mix["File"].str.split(";", expand=True)
    w1, w2 = df_mix["w1"].to_numpy(), df_mix["w2"].to_numpy()

    # Determine pigment names
    if pigment_names is None:
        pigment_names = sorted(set(pairs[0].unique()) | set(pairs[1].unique()))
    n_pigs = len(pigment_names)

    # === Compute stats per pigment ===
    counts = {p: {"p1": 0, "p2": 0, "w1_sum": 0.0, "w2_sum": 0.0} for p in pigment_names}

    for i, (p1, p2) in enumerate(pairs.values):
        counts[p1]["p1"] += 1
        counts[p1]["w1_sum"] += w1[i]
        counts[p2]["p2"] += 1
        counts[p2]["w2_sum"] += w2[i]

    df_stats = pd.DataFrame([
        {
            "Pigment": p,
            "Count_p1": v["p1"],
            "Count_p2": v["p2"],
            "Mean_w1_if_p1": v["w1_sum"]/v["p1"] if v["p1"] else 0,
            "Mean_w2_if_p2": v["w2_sum"]/v["p2"] if v["p2"] else 0,
        }
        for p, v in counts.items()
    ])

    # === Compute recall from model predictions ===
    y_true = np.vstack(df_pred["True_Multi"].apply(lambda s: np.array(s.split(";"), dtype=int)))
    y_pred = np.vstack(df_pred["Pred_Multi"].apply(lambda s: np.array(s.split(";"), dtype=int)))

    recall = (y_true & y_pred).sum(axis=0) / np.maximum(y_true.sum(axis=0), 1)
    df_stats["Recall_model"] = recall

    # === Save CSV ===
    out_csv = os.path.join(out_dir, "Pigment_Balance_vs_Recall.csv")
    df_stats.to_csv(out_csv, index=False)
    print(f"[SAVE] Pigment balance & recall table -> {out_csv}")

    # ======================================================================
    # 1️⃣ CORRELATION PLOT (Mean dominant weight vs Recall)
    # ======================================================================
    plt.figure(figsize=(7, 6))
    plt.scatter(df_stats["Mean_w1_if_p1"], df_stats["Recall_model"],
                c=df_stats["Mean_w1_if_p1"], cmap="viridis", s=80, edgecolor="k")
    for i, row in df_stats.iterrows():
        plt.text(row["Mean_w1_if_p1"]+0.001, row["Recall_model"]+0.002, row["Pigment"].split("_")[0],
                 fontsize=7, alpha=0.6)
    plt.xlabel("Mean dominant weight (w1 when pigment is p1)")
    plt.ylabel("Model recall per pigment")
    plt.title("Recall vs Dominant Weight per Pigment")
    plt.colorbar(label="Mean w1 (dominance level)")
    plt.grid(True)
    plt.tight_layout()

    out_png_corr = os.path.join(out_dir, "Pigment_Dominance_vs_Recall.png")
    plt.savefig(out_png_corr, dpi=300)
    plt.close()
    print(f"[SAVE] Correlation plot -> {out_png_corr}")

    # ======================================================================
    # 2️⃣ HISTOGRAM (Dominant vs Secondary occurrences)
    # ======================================================================
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    indices = np.arange(len(df_stats))

    plt.bar(indices, df_stats["Count_p1"], bar_width, label="Dominant (p1)", color="steelblue")
    plt.bar(indices + bar_width, df_stats["Count_p2"], bar_width, label="Secondary (p2)", color="orange")

    plt.xticks(indices + bar_width/2, [p.split("_")[0] for p in df_stats["Pigment"]], rotation=45, ha="right")
    plt.ylabel("Number of occurrences")
    plt.title("Pigment occurrence balance (dominant vs secondary)")
    plt.legend()
    plt.tight_layout()

    out_png_hist = os.path.join(out_dir, "Pigment_Dominance_Distribution.png")
    plt.savefig(out_png_hist, dpi=300)
    plt.close()
    print(f"[SAVE] Histogram plot -> {out_png_hist}")

    # ======================================================================
    # 3️⃣ INTERPRETATION HELP
    # ======================================================================
    print("\n[INTERPRETATION]")
    print("• Pigments with more 'p1' occurrences and higher Mean_w1_if_p1 are dominant in mixtures.")
    print("• High Recall_model values correlate with dominance → model detects strong pigments better.")
    print("• The histogram shows if some pigments rarely appear as dominant (likely causing low recall).")

    return df_stats