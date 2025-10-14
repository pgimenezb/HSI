# hsi_lab/eval/report.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, hamming_loss, roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, log_loss
)

def plot_confusion_matrix_by_vector(
    df,
    y_pred_prob,
    y_true,
    threshold: float = 0.5,
    save_path: str | None = None,
    show: bool = False,
):
    # --- Binarización ---
    y_pred_bin = (y_pred_prob >= float(threshold)).astype(int)
    y_true_bin = y_true.astype(int)

    # --- Construcción de etiquetas legibles por vector ---
    def _row_to_label(r):
        file_ = r.get("File", "?")
        binder = r.get("Binder", r.get("binder", "?"))
        mixture = r.get("Mixture", r.get("mixture", "?"))
        return f"{file_}, {binder}, {mixture}"

    vec_to_label = {}
    if "Multi" in df.columns:
        for _, r in df.iterrows():
            try:
                key = tuple(r["Multi"])
                if key not in vec_to_label:
                    vec_to_label[key] = _row_to_label(r)
            except Exception:
                continue

    def vector_to_label(vec):
        key = tuple(vec.tolist() if hasattr(vec, "tolist") else vec)
        return vec_to_label.get(key, "Unknown")

    pred_labels = [vector_to_label(v) for v in y_pred_bin]
    true_labels = [vector_to_label(v) for v in y_true_bin]

    # orden estable siguiendo true_labels
    seen = {}
    for lab in true_labels:
        if lab not in seen:
            seen[lab] = True
    labels = np.array(list(seen.keys()), dtype=object)
    if labels.size == 0:
        labels = np.unique(true_labels + pred_labels)

    # matriz de confusión (conteos)
    cm_counts = confusion_matrix(true_labels, pred_labels, labels=labels)

    # normalización por fila
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_percent = cm_counts.astype(float) / np.maximum(row_sums, 1)
        cm_percent = np.nan_to_num(cm_percent)

    # figura
    n_labels = len(labels)
    fig_w = max(10, min(0.6 * n_labels, 40))
    fig_h = max(8,  min(0.6 * n_labels, 40))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp = ConfusionMatrixDisplay(cm_percent, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f", colorbar=True)

    ax.set_title("Confusion Matrix (row-normalized: %)", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0,  ha="right",  fontsize=9)
    plt.subplots_adjust(left=0.28, bottom=0.20, right=0.95, top=0.90)
    plt.tight_layout()

    saved_to = None
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        saved_to = save_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "labels": labels,
        "cm_counts": cm_counts,
        "cm_percent": cm_percent,
        "saved_to": saved_to,
    }

def print_global_and_per_group_metrics(y_true, y_pred_prob, y_pred_bin,
                                       pigment_idx, binder_idx, mixture_idx):
    """Imprime métricas globales y por bloques (pigments/binders/mixtures)."""
    print("\n🟢 CLASSIFICATION REPORT")
    print("----------------------------------")
    print(classification_report(y_true, y_pred_bin, zero_division=0))

    # Global
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob, average='micro')
    except ValueError:
        roc_auc = float('nan')
    try:
        avg_precision = average_precision_score(y_true, y_pred_prob, average='micro')
    except ValueError:
        avg_precision = float('nan')

    f1 = f1_score(y_true, y_pred_bin, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred_bin, average='micro', zero_division=0)
    precision = precision_score(y_true, y_pred_bin, average='micro', zero_division=0)
    try:
        ll = log_loss(y_true, y_pred_prob)
    except ValueError:
        ll = float('nan')

    print("\n🔹 Global:")
    print(f"F1={f1:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  "
          f"ROC-AUC={roc_auc:.4f}  AP={avg_precision:.4f}  LogLoss={ll:.4f}")

    def block(name, idx):
        if not idx:
            print(f"\n{name}: (sin clases)")
            return
        yt = y_true[:, idx]
        yp = y_pred_prob[:, idx]
        yb = y_pred_bin[:, idx]
        try:
            auc = roc_auc_score(yt, yp, average='micro')
        except ValueError:
            auc = float('nan')
        f1b = f1_score(yt, yb, average='micro', zero_division=0)
        rb  = recall_score(yt, yb, average='micro', zero_division=0)
        pb  = precision_score(yt, yb, average='micro', zero_division=0)
        try:
            llb = log_loss(yt, yp)
        except ValueError:
            llb = float('nan')
        print(f"\n🔸 {name}: F1={f1b:.4f}  Prec={pb:.4f}  Rec={rb:.4f}  AUC={auc:.4f}  LogLoss={llb:.4f}")

    block("Pigments", pigment_idx)
    block("Binders",  binder_idx)
    block("Mixtures", mixture_idx)
