# ==== IMPORTS NECESARIOS ====
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =========================================================
# === PLOTEO PURO (NO toca datos: solo visualización) ====
# =========================================================
def plot_confusion_matrix(cm, classes, title, out_path,
                          annotate_percent=False,      # 0.00–1.00 si False, 0.00–100.00 si True
                          cmap_name="Blues",
                          figsize=None,
                          min_font=5, max_font=11):
    n = len(classes)

    # Tamaño de figura a partir de nº de clases (celda ~0.35")
    if figsize is None:
        cell = 0.35
        side = max(8, min(36, cell * n))
        figsize = (side, side)

    fig, ax = plt.subplots(figsize=figsize, dpi=180, constrained_layout=True)

    # Asume cm normalizada a [0,1]
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap_name, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=13, pad=8)

    # ticks y etiquetas
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=90, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.tick_params(axis="x", which="major", pad=6)
    ax.tick_params(axis="y", which="major", pad=4)

    # Colorbar estrecho
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.3)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=9)

    # Celdas cuadradas y texto adaptativo
    ax.set_aspect("equal", adjustable="box")
    font_size = max(min_font, min(max_font, int(200 / max(n, 1))))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = (cm[i, j] * 100.0) if annotate_percent else cm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=font_size)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[OK] {title} -> {out_path}")


# ========================================================================
# === CONSTRUCTORES DE MATRICES (NO transforman y_true/y_pred) ===========
# ========================================================================
def _labels_to_indices(y_true_labels, y_pred_labels, classes=None):
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

    return ti, pi, classes


def confusion_from_labels(y_true_labels, y_pred_labels, classes=None):
    ti, pi, classes = _labels_to_indices(y_true_labels, y_pred_labels, classes=classes)
    labels_idx = list(range(len(classes)))

    # 1) Conteos crudos
    cm_counts = sk_confusion_matrix(ti, pi, labels=labels_idx, normalize=None)

    # 2) Normalización por filas (cada fila suma 1). Evita división por cero.
    row_sums = cm_counts.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0.0] = 1.0
    cm_norm = cm_counts.astype(float) / row_sums

    return cm_norm, classes

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS GLOBALES Y POR MATERIAL
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, hamming_loss, log_loss, average_precision_score,
    classification_report
)


def compute_detailed_metrics(y_true, y_pred_prob, num_files, threshold=0.5, verbose=True):
    """
    Calcula métricas globales y por material (pigments, binders, varnishes).
    Devuelve un dict estructurado listo para exportar a CSV.
    """
    y_pred_bin = (y_pred_prob >= threshold).astype(int)

    def compute_metrics(y_true, y_pred_bin, y_pred_prob):
        metrics = {}
        metrics["non_zero_acc_sample"] = np.mean(np.any((y_true & y_pred_bin), axis=1))
        metrics["non_zero_acc_label"] = np.mean(np.any((y_true == y_pred_bin), axis=0))
        metrics["strict_acc"] = np.mean(np.all(y_true == y_pred_bin, axis=1))
        metrics["general_acc"] = accuracy_score(y_true.flatten(), y_pred_bin.flatten())
        metrics["keras_like_acc"] = (y_true == y_pred_bin).mean(axis=1).mean()
        metrics["hamming_acc"] = 1 - hamming_loss(y_true, y_pred_bin)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_prob, average="micro")
        except ValueError:
            metrics["roc_auc"] = np.nan
        try:
            metrics["log_loss"] = log_loss(y_true, y_pred_prob)
        except ValueError:
            metrics["log_loss"] = np.nan
        metrics["f1"] = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
        metrics["precision"] = precision_score(y_true, y_pred_bin, average="micro", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred_bin, average="micro", zero_division=0)
        metrics["avg_precision"] = average_precision_score(y_true, y_pred_prob, average="micro")
        return metrics

    if verbose:
        print("\n🟢 CLASSIFICATION REPORT")
        print(classification_report(y_true, y_pred_bin, zero_division=0))

    # === GLOBAL METRICS ===
    metrics_global = compute_metrics(y_true, y_pred_bin, y_pred_prob)

    # === MATERIAL-SPECIFIC ===
    total_labels = y_true.shape[1]
    n_p = num_files
    pigment_idx = list(range(0, n_p))
    binder_idx = list(range(n_p, min(n_p + 2, total_labels)))
    varnish_idx = list(range(n_p + 2, min(n_p + 4, total_labels)))

    def eval_subset(indices):
        if not indices:
            return None
        return compute_metrics(
            y_true[:, indices], y_pred_bin[:, indices], y_pred_prob[:, indices]
        )

    metrics_pig = eval_subset(pigment_idx)
    metrics_bind = eval_subset(binder_idx)
    metrics_varn = eval_subset(varnish_idx)

    # === ORGANIZED RETURN ===
    results = {
        "GLOBAL": metrics_global,
        "PIGMENTS": metrics_pig,
        "BINDERS": metrics_bind,
        "VARNISHES": metrics_varn,
    }

    if verbose:
        for scope, vals in results.items():
            if vals is None:
                continue
            print(f"\n🔹 {scope} METRICS:")
            for k, v in vals.items():
                print(f"{k:30s}: {v:.4f}")

    return results



def plot_confusion_matrix(cm, classes, title, out_path,
                          annotate_percent=False,
                          cmap_name="Blues",
                          figsize=None,
                          min_font=5, max_font=11):
    n = len(classes)

    if figsize is None:
        cell = 0.35
        side = max(8, min(36, cell * n))
        figsize = (side, side)

    fig, ax = plt.subplots(figsize=figsize, dpi=180, constrained_layout=True)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap_name, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=13, pad=8)

    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=90, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.tick_params(axis="x", which="major", pad=6)
    ax.tick_params(axis="y", which="major", pad=4)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.3)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=9)

    ax.set_aspect("equal", adjustable="box")
    font_size = max(min_font, min(max_font, int(200 / max(n, 1))))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = (cm[i, j] * 100.0) if annotate_percent else cm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=font_size)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[OK] {title} -> {out_path}")