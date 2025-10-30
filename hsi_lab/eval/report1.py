import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as mpl_cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, hamming_loss, log_loss, average_precision_score
)
import warnings


# =========================================================
# === VISUALIZATION UTILITIES ===
# =========================================================
def plot_confusion_matrix(cm, classes, title, save_path, annotate_percent=True):
    """Plot confusion or coactivation matrices with a consistent visual style."""
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


# =========================================================
# === CONFUSION & COACTIVATION BUILDERS ===
# =========================================================
def confusion_from_labels(y_true_labels, y_pred_labels, classes=None):
    """Build normalized confusion matrix from label strings."""
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
    cm_norm = cm.astype(float) / row_sums
    return cm_norm, classes


def soft_confusion_matrix(y_true, y_pred_prob, class_names, normalize="row"):
    """Soft (coactivation-style) confusion matrix."""
    cm = np.dot(y_true.T, y_pred_prob)
    if normalize == "row":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
    return cm


# =========================================================
# === METRICS ===
# =========================================================
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


# =========================================================
# === COMBINED REPORT CREATOR ===
# =========================================================
def generate_combined_report(y_true, y_pred_prob, n_pigments, output_dir, name):
    """
    Generates both confusion (discrete) and coactivation (soft) matrices,
    saves their PNGs, and compiles a single CSV summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    cm_conf_dir = os.path.join(output_dir, "confusion_matrix")
    cm_coact_dir = os.path.join(output_dir, "coactivation_matrix")
    os.makedirs(cm_conf_dir, exist_ok=True)
    os.makedirs(cm_coact_dir, exist_ok=True)

    # Split predictions
    y_pred_bin = (y_pred_prob > 0.5).astype(int)
    y_true_pig, y_true_mix = y_true[:, :n_pigments], y_true[:, n_pigments:n_pigments+4]
    y_pred_pig, y_pred_mix = y_pred_bin[:, :n_pigments], y_pred_bin[:, n_pigments:n_pigments+4]
    y_prob_pig, y_prob_mix = y_pred_prob[:, :n_pigments], y_pred_prob[:, n_pigments:n_pigments+4]

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
