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

def print_global_and_per_group_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    y_pred_bin: np.ndarray,
    pigment_idx: list,
    binder_idx: list,
    mixture_idx: list,
) -> None:
    """
    Imprime métricas globales y por bloques (Pigments / Binders / Mixtures).
    - y_true:    (N, L) binario
    - y_pred_prob: (N, L) probabilidades [0,1]
    - y_pred_bin:  (N, L) binario
    - *_idx: listas de índices de etiquetas para cada bloque
    """

    n_samples, n_labels = y_true.shape
    print("\n🟢 CLASSIFICATION REPORT")
    print("----------------------------------")
    print(classification_report(y_true, y_pred_bin, zero_division=0))

    # ---------- Índices usados ----------
    print("\n🔹 Index ranges:")
    print(f"Pigments: {pigment_idx if pigment_idx else '[]'}")
    print(f"Binders : {binder_idx  if binder_idx  else '[]'}")
    print(f"Mixtures: {mixture_idx if mixture_idx else '[]'}")

    # ---------- Bloque de métricas: helpers ----------
    def _safe_auc(yt, yp):
        try:
            return float(roc_auc_score(yt, yp, average='micro'))
        except Exception:
            return float('nan')

    def _safe_ap(yt, yp):
        try:
            return float(average_precision_score(yt, yp, average='micro'))
        except Exception:
            return float('nan')

    def _safe_logloss(yt, yp):
        try:
            return float(log_loss(yt, yp))
        except Exception:
            return float('nan')

    def _accuracy_pack(yt, yb):
        # métricas de "accuracy" varias
        non_zero_sample = float(np.mean(np.any((yt & yb) == 1, axis=1)))
        non_zero_label  = float(np.mean(np.any((yt == yb), axis=0)))
        strict          = float(np.mean(np.all(yt == yb, axis=1)))
        general         = float(accuracy_score(yt, yb))
        keras_like      = float((yt == yb).mean(axis=1).mean())
        hamming_acc     = float(1.0 - hamming_loss(yt, yb))
        return {
            "non_zero_sample": non_zero_sample,
            "non_zero_label": non_zero_label,
            "strict": strict,
            "general": general,
            "keras_like": keras_like,
            "hamming_acc": hamming_acc,
        }

    def _prf_pack(yt, yb):
        f1  = float(f1_score(yt, yb, average='micro', zero_division=0))
        rec = float(recall_score(yt, yb, average='micro', zero_division=0))
        pre = float(precision_score(yt, yb, average='micro', zero_division=0))
        return {"f1": f1, "recall": rec, "precision": pre}

    # ---------- GLOBAL ----------
    print("\n🟢 BLOCK 1: GLOBAL METRICS")

    glob_acc = _accuracy_pack(y_true, y_pred_bin)
    glob_prf = _prf_pack(y_true, y_pred_bin)
    glob_auc = _safe_auc(y_true, y_pred_prob)
    glob_ap  = _safe_ap(y_true, y_pred_prob)
    glob_ll  = _safe_logloss(y_true, y_pred_prob)

    print("\n🔹 Accuracy Metrics:")
    print(f"Non-zero accuracy per sample:  {glob_acc['non_zero_sample']:.4f}")
    print(f"Non-zero accuracy per label:   {glob_acc['non_zero_label']:.4f}")
    print(f"Strict accuracy:               {glob_acc['strict']:.4f}")
    print(f"General accuracy (sklearn):    {glob_acc['general']:.4f}")
    print(f\"Keras-like\" accuracy:         {glob_acc['keras_like']:.4f}")
    print(f"Hamming accuracy:              {glob_acc['hamming_acc']:.4f}")

    print("\n🔹 Other Metrics:")
    print(f"F1-Score:                      {glob_prf['f1']:.4f}")
    print(f"Precision:                     {glob_prf['precision']:.4f}")
    print(f"Recall:                        {glob_prf['recall']:.4f}")
    print(f"ROC-AUC (micro):               {glob_auc:.4f}")
    print(f"Average Precision (micro):     {glob_ap:.4f}")
    print(f"Log Loss:                      {glob_ll:.4f}")

    # ---------- PER-GROUP ----------
    print("\n----------------------------------")
    print("🟢 BLOCK 2: PER-MATERIAL METRICS")

    def _block(name: str, idx: list):
        if not idx:
            print(f"\n{name.upper()}: (no classes / empty indices)")
            return
        yt = y_true[:, idx]
        yp = y_pred_prob[:, idx]
        yb = y_pred_bin[:, idx]

        acc = _accuracy_pack(yt, yb)
        prf = _prf_pack(yt, yb)
        auc = _safe_auc(yt, yp)
        ll  = _safe_logloss(yt, yp)

        print(f"\n🔸 {name.upper()} Accuracy Metrics:")
        print(f"Non-zero accuracy per sample:  {acc['non_zero_sample']:.4f}")
        print(f"Non-zero accuracy per label:   {acc['non_zero_label']:.4f}")
        print(f"Strict accuracy:               {acc['strict']:.4f}")
        print(f"General accuracy (sklearn):    {acc['general']:.4f}")
        print(f\"Keras-like\" accuracy:         {acc['keras_like']:.4f}")
        print(f"Hamming accuracy:              {acc['hamming_acc']:.4f}")

        print("\n🔹 Other Metrics:")
        print(f"F1 Score:                      {prf['f1']:.4f}")
        print(f"Precision:                     {prf['precision']:.4f}")
        print(f"Recall:                        {prf['recall']:.4f}")
        print(f"ROC-AUC (micro):               {auc:.4f}")
        print(f"Log Loss:                      {ll:.4f}")

    _block("Pigments", pigment_idx)
    _block("Binders",  binder_idx)
    _block("Mixtures", mixture_idx)
