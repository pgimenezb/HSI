# hsi_lab/eval/report.py
import os
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

# Backend no interactivo (headless)
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, accuracy_score, hamming_loss,
    roc_auc_score, average_precision_score, log_loss,
    f1_score, precision_score, recall_score, multilabel_confusion_matrix,
    confusion_matrix, ConfusionMatrixDisplay
)

# ------------------ util guardado ------------------
def _safe_savefig(fig, path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _plot_heatmap(cm: np.ndarray, title: str, xticks: List[str], yticks: List[str],
                  save_path: str, normalize_rows: bool = True, cmap: str = "Blues", dpi: int = 150):
    cm = cm.astype(float)
    if normalize_rows:
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cm = cm / denom
        ann = cm * 100.0
        fmt = "{:.2f}%"
    else:
        ann = cm
        fmt = "{:.2f}"

    H, W = cm.shape
    fig_w = max(8, min(0.6 * max(H, W), 36))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w), dpi=dpi)
    im = ax.imshow(cm, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(W), xticks, rotation=90)
    ax.set_yticks(range(H), yticks)

    for i in range(H):
        for j in range(W):
            ax.text(j, i, fmt.format(ann[i, j]), ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row-normalized" if normalize_rows else "value")
    plt.tight_layout()
    _safe_savefig(fig, save_path, dpi=dpi)

# ------------------ indices de bloques ------------------
def split_block_indices(total_labels: int,
                        binder_mapping: dict,
                        mixture_mapping: dict) -> Tuple[List[int], List[int], List[int]]:
    """
    Devuelve (pigment_idx, binder_idx, mixture_idx) asumiendo orden [Pigments | Binders | Mixtures]
    en el vector de etiquetas.
    """
    Nb = len(binder_mapping)
    Nm = len(mixture_mapping)
    Np = total_labels - Nb - Nm
    if Np < 0:
        raise ValueError(f"total_labels={total_labels} < Nb+Nm={Nb+Nm}")
    return list(range(Np)), list(range(Np, Np + Nb)), list(range(Np + Nb, total_labels))

# ------------------ CM por bloque (P/B/M) ------------------
def plot_cm_block(y_true_bin: np.ndarray,
                  y_pred_prob: np.ndarray,
                  block_idx: List[int],
                  tick_prefix: str,
                  class_names: Optional[List[str]],
                  save_path: str):
    """
    Crea CM KxK para un bloque (pigments/binders/mixtures).
    Cada fila reparte el 100% de prob. dentro del bloque (normalización por fila).
    """
    K = len(block_idx)
    if class_names is None:
        class_names = [f"{tick_prefix}{i+1}" for i in range(K)]
    else:
        # ticks cortos P1.. / B1.. / M1..
        class_names = [f"{tick_prefix}{i+1}" for i in range(K)]
    # índice verdadero por fila dentro del bloque (asume 1-hot por bloque en y_true_bin)
    t = np.argmax(y_true_bin[:, block_idx], axis=1)
    cm = np.zeros((K, K), dtype=float)
    for i in range(y_true_bin.shape[0]):
        p = y_pred_prob[i, block_idx].astype(float)
        p = np.clip(p, 1e-9, 1.0)
        p = p / p.sum()
        cm[t[i], :] += p
    _plot_heatmap(cm, f"Confusion (per-sample %) – {tick_prefix} block", class_names, class_names, save_path)
    # leyenda CSV (tick -> nombre largo si lo pasas)
    legend = pd.DataFrame({
        "tick": [f"{tick_prefix}{i+1}" for i in range(K)],
        "name": class_names if class_names else [f"{tick_prefix}{i+1}" for i in range(K)]
    })
    legend.to_csv(os.path.splitext(save_path)[0] + "_legend.csv", index=False)


# ------------------ CM multilabel por muestra (S) ------------------

def plot_cm_samples_multilabel(
    df_grouped: pd.DataFrame,
    y_pred_prob: np.ndarray,
    y_true_bin:  np.ndarray,
    pigment_idx: list[int],
    binder_idx:  list[int],
    mixture_idx: list[int],
    pigment_names: list[str],
    binder_names:  list[str],
    mixture_names: list[str],
    save_path: str,
    file_col: str = "File",
    mixture_order: Optional[List[str]] = None,
):

    import re

    if mixture_order is None:
        mixture_order = mixture_names
    mix_ord = {name: i for i, name in enumerate(mixture_order)}

    # Verdaderos (para ordenar filas/columnas)
    t_m = np.argmax(y_true_bin[:, mixture_idx], axis=1)  # 0..(M-1)
    files = list(df_grouped[file_col].astype(str))

    def file_key(s: str):
        m = re.match(r"^(\d+)_", s)
        return (int(m.group(1)) if m else 10**9, s)

    # Orden común para filas y columnas: por File y dentro por mixture_order
    rows = list(range(len(df_grouped)))
    rows.sort(key=lambda i: (file_key(files[i]), mix_ord[mixture_names[t_m[i]]]))
    K = len(rows)
    ticks = [f"S{i+1}" for i in range(K)]

    # Columnas en el MISMO orden que filas
    col_file = [files[i] for i in rows]
    col_mix  = [t_m[i]   for i in rows]   # mixture REAL asociada a esa columna

    # (file, mixture_id) -> posición de columna
    col_pos = {(col_file[j], col_mix[j]): j for j in range(K)}

    # Matriz de CONTEOS (enteros)
    cm = np.zeros((K, K), dtype=int)
    for rpos, i in enumerate(rows):
        # predicción "dura" de mixture para ESTA fila
        p_mix = y_pred_prob[i, mixture_idx]
        pred_m = int(np.argmax(p_mix))

        # SOLO se contabiliza dentro del MISMO file
        j = col_pos.get((files[i], pred_m), None)
        if j is not None:
            cm[rpos, j] += 1

    # Guardo la matriz de confusion además del heatmap en formato CSV
    pd.DataFrame(cm, index=ticks, columns=ticks).to_csv(os.path.splitext(save_path)[0] + "_matrix.csv")

    # Pintar enteros, sin normalizar
    # (reutilizamos _plot_heatmap, pero pidiéndole NO normalizar y formato entero)
    def _plot_heatmap_counts(cm_counts: np.ndarray, title: str, xticks, yticks, path: str):
        cm_float = cm_counts.astype(float)
        H, W = cm_float.shape
        fig_w = max(8, min(0.6 * max(H, W), 36))
        fig, ax = plt.subplots(figsize=(fig_w, fig_w), dpi=150)
        im = ax.imshow(cm_float, cmap="Blues", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(W), xticks, rotation=90)
        ax.set_yticks(range(H), yticks)
        for i_ in range(H):
            for j_ in range(W):
                ax.text(j_, i_, f"{int(cm_counts[i_, j_])}", ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("counts")
        plt.tight_layout()
        _safe_savefig(fig, path, dpi=150)

    _plot_heatmap_counts(cm, "Confusion (counts) – S (File×Mixture, exact match)", ticks, ticks, save_path)

    # Leyenda S#
    legend = pd.DataFrame({
        "tick": ticks,
        "file":   [files[i] for i in rows],
        "pigment":[pigment_names[np.argmax(y_true_bin[i, pigment_idx])] for i in rows],
        "binder": [binder_names[np.argmax(y_true_bin[i, binder_idx])]  for i in rows],
        "mixture":[mixture_names[t_m[i]] for i in rows],
    })
    legend.to_csv(os.path.splitext(save_path)[0] + "_legend.csv", index=False)



# ------------------ CSVs de métricas ------------------
def save_metrics_csvs(y_true: np.ndarray,
                      y_pred_prob: np.ndarray,
                      pigment_idx: List[int],
                      binder_idx: List[int],
                      mixture_idx: List[int],
                      out_dir: str,
                      prefix: str):
    os.makedirs(out_dir, exist_ok=True)

    def _safe_auc(yt, yp):
        try:    return float(roc_auc_score(yt, yp, average="micro"))
        except: return float("nan")

    def _safe_ap(yt, yp):
        try:    return float(average_precision_score(yt, yp, average="micro"))
        except: return float("nan")

    def _safe_ll(yt, yp):
        try:    return float(log_loss(yt, yp))
        except: return float("nan")

    def _accs(yt, yb):
        return {
            "strict":      float(np.mean(np.all(yt == yb, axis=1))),
            "general":     float(accuracy_score(yt, yb)),
            "keras_like":  float((yt == yb).mean(axis=1).mean()),
            "hamming_acc": float(1.0 - hamming_loss(yt, yb)),
        }

    def _prf(yt, yb):
        return {
            "f1_micro":        float(f1_score(yt, yb, average="micro", zero_division=0)),
            "precision_micro": float(precision_score(yt, yb, average="micro", zero_division=0)),
            "recall_micro":    float(recall_score(yt, yb, average="micro", zero_division=0)),
        }

    y_true_bin = (y_true >= 0.5).astype(int)
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)

    mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin) 
    
    # GLOBAL
    g = {**_accs(y_true_bin, y_pred_bin), **_prf(y_true_bin, y_pred_bin),
         "roc_auc_micro": _safe_auc(y_true_bin, y_pred_prob),
         "avg_precision_micro": _safe_ap(y_true_bin, y_pred_prob),
         "log_loss": _safe_ll(y_true_bin, y_pred_prob)}
    pd.DataFrame([g]).to_csv(os.path.join(out_dir, f"{prefix}_global_metrics.csv"), index=False)

    # POR BLOQUE
    rows = []
    for name, idx in [("pigments", pigment_idx), ("binders", binder_idx), ("mixtures", mixture_idx)]:
        yt, yp = y_true_bin[:, idx], y_pred_prob[:, idx]
        yb = y_pred_bin[:, idx]
        rows.append({ "scope": name,
            **_accs(yt, yb), **_prf(yt, yb),
            "roc_auc_micro": _safe_auc(yt, yp),
            "avg_precision_micro": _safe_ap(yt, yp),
            "log_loss": _safe_ll(yt, yp)
        })
        # classification_report por bloque
        
        rep = classification_report(yt, yb, zero_division=0)
        with open(os.path.join(out_dir, f"{prefix}_classification_report_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(rep)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{prefix}_per_group_metrics.csv"), index=False)


