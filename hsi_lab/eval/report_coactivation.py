import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib.cm as mpl_cm
from matplotlib.colors import ListedColormap 
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, hamming_loss, log_loss, average_precision_score,
    classification_report
)


# =========================================================
# === PLOTEO PURO (NO toca datos: solo visualización) ====
# =========================================================

def plot_confusion_matrix(cm, classes, title, save_path, annotate_percent=True):
    """Matriz de confusión con estilo limpio, texto negro y celdas más grandes."""
    # Tamaño adaptativo: más clases → figura más grande
    n = len(classes)
    fig_size = max(6, min(0.45 * n, 20))
    plt.figure(figsize=(fig_size, fig_size))
    ax = plt.gca()

    # Colormap azul con fondo blanco
    base_cmap = mpl_cm.get_cmap("Blues", 256)
    cmap_array = base_cmap(np.linspace(0, 1, 256))
    cmap_array[0, :] = [1, 1, 1, 1]  # blanco para los valores mínimos
    cmap = ListedColormap(cmap_array)

    vmax = np.max(cm) if np.max(cm) > 0 else 1.0
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)

    # Ejes y etiquetas
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)

    # Anotar celdas (solo valores visibles)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j] * 100 if annotate_percent else cm[i, j]
            if value < 0.005:
                continue  # omitir ceros reales
            ax.text(
                j, i,
                f"{value:.2f}",
                ha="center", va="center",
                color="black",              # siempre negro
                fontsize=8, fontweight="normal"
            )

    # Barra de color lateral
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.2)
    plt.colorbar(im, cax=cax, format="%.2f")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVE] Confusion matrix -> {save_path}")






# ========================================================================
# === CONSTRUCTORES DE MATRICES (NO transforman y_true/y_pred) ===========
# ========================================================================
def _labels_to_indices(y_true_labels, y_pred_labels, classes=None):
    """
    Convierte etiquetas string en índices enteros de clase coherentes entre y_true e y_pred.
    """
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


def confusion_from_labels(y_true, y_pred, classes):
    """Genera matriz de confusión y etiquetas usadas."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    used = classes

    # Normalización por filas → cada clase real suma 1.0
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)
    return cm_norm, used




# =========================================================
# === MÉTRICAS (corregidas) ===
# =========================================================
def compute_metrics(y_true, y_pred_bin, y_pred_prob):
    from sklearn.metrics import (
        accuracy_score, hamming_loss, roc_auc_score,
        log_loss, f1_score, precision_score, recall_score,
        average_precision_score
    )
    import numpy as np

    metrics = {}

    y_true_b = (y_true > 0.5).astype(bool)
    y_pred_b = (y_pred_bin > 0.5).astype(bool)

    # 📊 Accuracy metrics
    metrics["non_zero_acc_sample"] = np.mean(np.any(y_true_b & y_pred_b, axis=1))
    metrics["non_zero_acc_label"]  = np.mean(np.any(y_true_b == y_pred_b, axis=0))
    metrics["strict_acc"]          = np.mean(np.all(y_true_b == y_pred_b, axis=1))
    metrics["general_acc"]         = accuracy_score(y_true_b.flatten(), y_pred_b.flatten())
    metrics["keras_like_acc"]      = (y_true_b == y_pred_b).mean(axis=1).mean()
    metrics["hamming_acc"]         = 1 - hamming_loss(y_true_b, y_pred_b)

    # 🔸 Probabilistic metrics
    try:
        metrics["roc_auc"] = roc_auc_score(y_true_b, np.clip(y_pred_prob, 0, 1), average="micro")
    except ValueError:
        metrics["roc_auc"] = np.nan

    try:
        metrics["log_loss"] = log_loss(y_true_b.astype(int), np.clip(y_pred_prob, 1e-7, 1 - 1e-7))
    except ValueError:
        metrics["log_loss"] = np.nan

    # ⚙️ Classification metrics
    metrics["f1"]            = f1_score(y_true_b, y_pred_b, average="micro", zero_division=0)
    metrics["precision"]     = precision_score(y_true_b, y_pred_b, average="micro", zero_division=0)
    metrics["recall"]        = recall_score(y_true_b, y_pred_b, average="micro", zero_division=0)
    metrics["avg_precision"] = average_precision_score(y_true_b, np.clip(y_pred_prob, 0, 1), average="micro")

    # 🧾 Descriptive phrases (short, report-friendly)
    descriptions = {
        "non_zero_acc_sample": "Fraction of samples where at least one true label was correctly predicted.",
        "non_zero_acc_label":  "Fraction of labels with at least one correct prediction across all samples.",
        "strict_acc":          "Strict accuracy: all predicted labels must match the true labels.",
        "general_acc":         "Overall accuracy across all labels (flattened comparison).",
        "keras_like_acc":      "Mean sample-wise accuracy, similar to Keras behavior.",
        "hamming_acc":         "Average per-label accuracy (1 − Hamming loss).",
        "roc_auc":             "Area under the ROC curve (probabilistic ranking performance).",
        "log_loss":            "Penalty for confident but incorrect predictions (logarithmic loss).",
        "f1":                  "Harmonic mean of precision and recall (F1 micro).",
        "precision":           "Fraction of predicted positives that were correct.",
        "recall":              "Fraction of true positives that were successfully detected.",
        "avg_precision":       "Average precision (area under the precision–recall curve).",
    }

    return metrics, descriptions


# =========================================================
# === MÉTRICAS DETALLADAS ===
# =========================================================
def compute_detailed_metrics(y_true, y_pred_prob, num_files, threshold=0.5, verbose=False):
    y_pred_bin = (y_pred_prob > threshold).astype(int)
    results = {}

    # --- Global metrics ---
    results["global"] = compute_metrics(y_true, y_pred_bin, y_pred_prob)

    # --- Split pigments / mixtures ---
    y_true_pig, y_true_mix = y_true[:, :num_files], y_true[:, num_files:num_files+4]
    y_pred_pig, y_pred_mix = y_pred_bin[:, :num_files], y_pred_bin[:, num_files:num_files+4]
    y_prob_pig, y_prob_mix = y_pred_prob[:, :num_files], y_pred_prob[:, num_files:num_files+4]

    results["pigments"] = compute_metrics(y_true_pig, y_pred_pig, y_prob_pig)
    results["mixtures"] = compute_metrics(y_true_mix, y_pred_mix, y_prob_mix)

    # --- Optional console print ---
    if verbose:
        print("\n[DETAILED METRICS REPORT]")
        for scope, (metrics, descriptions) in results.items():
            print(f"\n--- {scope.upper()} ---")
            for mk, mv in metrics.items():
                desc = descriptions.get(mk, "")
                print(f"{mk:20s}: {mv:.4f}  |  {desc}")

    return results
