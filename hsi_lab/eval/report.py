import os
from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend without X11
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, hamming_loss, roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, log_loss
)

# --- CONFUSION MATRIX ---

def plot_confusion_matrix_by_vector(
    df,
    y_pred_prob: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = False,
):
    # --- Binarización ---
    y_pred_bin = (y_pred_prob >= float(threshold)).astype(int)
    y_true_bin = y_true.astype(int)

    # --- Mapa tuple(vector)->etiqueta legible ---
    def _row_to_label(r):
        file_ = r.get("File", "?")
        binder = r.get("Binder", r.get("binder", "?"))
        mixture = r.get("Mixture", r.get("mixture", "?"))
        return f"{file_}, {binder}, {mixture}"

    vec_to_label = {}
    if hasattr(df, "iterrows") and "Multi" in df.columns:
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

    # Orden estable según aparición en y_true
    seen = {}
    for lab in true_labels:
        if lab not in seen:
            seen[lab] = True
    labels = np.array(list(seen.keys()), dtype=object)
    if labels.size == 0:
        labels = np.unique(true_labels + pred_labels)

    cm_counts = confusion_matrix(true_labels, pred_labels, labels=labels)

    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_percent = cm_counts.astype(float) / np.maximum(row_sums, 1)
        cm_percent = np.nan_to_num(cm_percent)

    # Figura tamaño dinámico
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


def plot_confusion_matrix_for_block(
    y_true_block: np.ndarray,
    y_pred_prob_block: np.ndarray,
    class_prefix: str,
    save_path: Optional[str] = None,
    show: bool = False,
):

    import numpy as _np
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    assert y_true_block.ndim == 2 and y_pred_prob_block.ndim == 2
    assert y_true_block.shape == y_pred_prob_block.shape
    n_classes = y_true_block.shape[1]

    # Etiquetas por argmax (robusto ante 0 o >1 unos)
    y_true_cls = _np.argmax(y_true_block, axis=1)
    y_pred_cls = _np.argmax(y_pred_prob_block, axis=1)

    # Nombres de clases simples: P0.., B0.., M0..
    labels = [f"{class_prefix}{i}" for i in range(n_classes)]

    # CM de conteos
    cm_counts = confusion_matrix(y_true_cls, y_pred_cls, labels=list(range(n_classes)))

    # Normalizada por fila (%)
    with _np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_percent = cm_counts.astype(float) / _np.maximum(row_sums, 1)
        cm_percent = _np.nan_to_num(cm_percent)

    # Tamaño dinámico
    n_labels = n_classes
    fig_w = max(8, min(0.6 * n_labels, 40))
    fig_h = max(6, min(0.6 * n_labels, 40))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp = ConfusionMatrixDisplay(cm_percent, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f", colorbar=True)

    ax.set_title(f"Confusion Matrix – {class_prefix} block (row-normalized: %)", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0,  ha="right",  fontsize=9)
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


# --- METRICS ---

def print_global_and_per_group_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    y_pred_bin: np.ndarray,
    pigment_idx: list,
    binder_idx: list,
    mixture_idx: list,
    out_dir: str = "projects/HSI/outputs/other_outputs",
    report_prefix: str = "val"
):
    os.makedirs(out_dir, exist_ok=True)

    # ---------- helpers ----------
    def _safe_auc(yt, yp):
        try:    return float(roc_auc_score(yt, yp, average='micro'))
        except: return float('nan')

    def _safe_ap(yt, yp):
        try:    return float(average_precision_score(yt, yp, average='micro'))
        except: return float('nan')

    def _safe_logloss(yt, yp):
        try:    return float(log_loss(yt, yp))
        except: return float('nan')

    def _accuracy_pack(yt, yb):
        non_zero_sample = float(np.mean(np.any((yt & yb) == 1, axis=1)))
        non_zero_label  = float(np.mean(np.any((yt == yb), axis=0)))
        strict          = float(np.mean(np.all(yt == yb, axis=1)))
        general         = float(accuracy_score(yt, yb))
        keras_like      = float((yt == yb).mean(axis=1).mean())
        hamming_acc     = float(1.0 - hamming_loss(yt, yb))
        return {
            "non_zero_sample": non_zero_sample,
            "non_zero_label":  non_zero_label,
            "strict":          strict,
            "general":         general,
            "keras_like":      keras_like,
            "hamming_acc":     hamming_acc,
        }

    def _prf_pack(yt, yb):
        return {
            "f1":        float(f1_score(yt, yb, average='micro', zero_division=0)),
            "precision": float(precision_score(yt, yb, average='micro', zero_division=0)),
            "recall":    float(recall_score(yt, yb, average='micro', zero_division=0)),
        }

    # ---------- imprimir breve por consola ----------
    print("\n🟢 CLASSIFICATION REPORT")
    print("----------------------------------")
    print(classification_report(y_true, y_pred_bin, zero_division=0))

    print("\n🔹 Index ranges:")
    print(f"Pigments: {pigment_idx if pigment_idx else '[]'}")
    print(f"Binders : {binder_idx  if binder_idx  else '[]'}")
    print(f"Mixtures: {mixture_idx if mixture_idx else '[]'}")

    # ---------- GLOBAL ----------
    glob_acc = _accuracy_pack(y_true, y_pred_bin)
    glob_prf = _prf_pack(y_true, y_pred_bin)
    glob_auc = _safe_auc(y_true, y_pred_prob)
    glob_ap  = _safe_ap(y_true, y_pred_prob)
    glob_ll  = _safe_logloss(y_true, y_pred_prob)

    print("\n🟢 BLOCK 1: GLOBAL METRICS")
    print(f"Non-zero/sample: {glob_acc['non_zero_sample']:.4f} | Non-zero/label: {glob_acc['non_zero_label']:.4f} | "
          f"Strict: {glob_acc['strict']:.4f} | General: {glob_acc['general']:.4f} | "
          f"Keras-like: {glob_acc['keras_like']:.4f} | Hamming-acc: {glob_acc['hamming_acc']:.4f}")
    print(f"F1: {glob_prf['f1']:.4f} | Precision: {glob_prf['precision']:.4f} | Recall: {glob_prf['recall']:.4f} | "
          f"ROC-AUC(μ): {glob_auc:.4f} | AP(μ): {glob_ap:.4f} | LogLoss: {glob_ll:.4f}")

    global_row = {
        "scope": "GLOBAL",
        **glob_acc,
        **glob_prf,
        "roc_auc_micro": glob_auc,
        "avg_precision_micro": glob_ap,
        "log_loss": glob_ll,
    }
    df_global = pd.DataFrame([global_row])

    # ---------- PER-GROUP ----------
    print("\n----------------------------------")
    print("🟢 BLOCK 2: PER-MATERIAL METRICS")

    rows = []
    for name, idx in [("Pigments", pigment_idx), ("Binders", binder_idx), ("Mixtures", mixture_idx)]:
        if not idx:
            print(f"\n{name.upper()}: (no classes / empty indices)")
            rows.append({"scope": name, "note": "empty index"})
            continue
        yt = y_true[:, idx]
        yp = y_pred_prob[:, idx]
        yb = y_pred_bin[:, idx]

        acc = _accuracy_pack(yt, yb)
        prf = _prf_pack(yt, yb)
        auc = _safe_auc(yt, yp)
        ap  = _safe_ap(yt, yp)
        ll  = _safe_logloss(yt, yp)

        print(f"\n🔸 {name.upper()} -> "
              f"Non-zero/sample: {acc['non_zero_sample']:.4f} | Non-zero/label: {acc['non_zero_label']:.4f} | "
              f"Strict: {acc['strict']:.4f} | General: {acc['general']:.4f} | "
              f"Keras-like: {acc['keras_like']:.4f} | Hamming-acc: {acc['hamming_acc']:.4f} | "
              f"F1: {prf['f1']:.4f} | P: {prf['precision']:.4f} | R: {prf['recall']:.4f} | "
              f"ROC-AUC(μ): {auc:.4f} | AP(μ): {ap:.4f} | LogLoss: {ll:.4f}")

        rows.append({
            "scope": name,
            **acc,
            **prf,
            "roc_auc_micro": auc,
            "avg_precision_micro": ap,
            "log_loss": ll,
        })

        # --- NUEVO: classification_report por bloque a disco ---
        rep_block_dict = classification_report(yt, yb, output_dict=True, zero_division=0)
        df_rep_block = pd.DataFrame(rep_block_dict).transpose()
        safe_name = name.lower()
        p_rep_block_csv = os.path.join(out_dir, f"{report_prefix}_classification_report_{safe_name}.csv")
        p_rep_block_txt = os.path.join(out_dir, f"{report_prefix}_classification_report_{safe_name}.txt")
        df_rep_block.to_csv(p_rep_block_csv)
        with open(p_rep_block_txt, "w", encoding="utf-8") as f:
            f.write(classification_report(yt, yb, zero_division=0))
        print(f"   ↳ guardado classification_report por bloque: {p_rep_block_csv}")
        print(f"   ↳ guardado classification_report por bloque (txt): {p_rep_block_txt}")

    df_per_group = pd.DataFrame(rows)

    # ---------- classification_report GLOBAL a disco ----------
    rep_dict = classification_report(y_true, y_pred_bin, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(rep_dict).transpose()

    p_global_csv = os.path.join(out_dir, f"{report_prefix}_global_metrics.csv")
    p_group_csv  = os.path.join(out_dir, f"{report_prefix}_per_group_metrics.csv")
    p_rep_csv    = os.path.join(out_dir, f"{report_prefix}_classification_report.csv")
    p_rep_txt    = os.path.join(out_dir, f"{report_prefix}_classification_report.txt")

    df_global.to_csv(p_global_csv, index=False)
    df_per_group.to_csv(p_group_csv, index=False)
    df_report.to_csv(p_rep_csv)
    with open(p_rep_txt, "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred_bin, zero_division=0))

    print(f"\n💾 Guardado:")
    print(f" - {p_global_csv}")
    print(f" - {p_group_csv}")
    print(f" - {p_rep_csv}")
    print(f" - {p_rep_txt}")

    return df_global, df_per_group


    return df_global, df_per_group
