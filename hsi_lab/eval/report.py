import os
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, hamming_loss, roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, log_loss
)

# ---------------------------
# util: particionar el vector
# ---------------------------
def split_block_indices(total_labels: int, binder_mapping: dict, mixture_mapping: dict) -> Tuple[List[int], List[int], List[int]]:
    """Devuelve (pigment_idx, binder_idx, mixture_idx) sin asumir tamaños mágicos."""
    b_bits = len(next(iter(binder_mapping.keys())))
    m_bits = len(next(iter(mixture_mapping.keys())))
    n_pig = total_labels - b_bits - m_bits
    if n_pig <= 0:
        raise ValueError(
            f"Incompatibilidad de dimensiones: total={total_labels}, "
            f"binder_bits={b_bits}, mixture_bits={m_bits}"
        )
    pigment_idx = list(range(0, n_pig))
    binder_idx  = list(range(n_pig, n_pig + b_bits))
    mixture_idx = list(range(n_pig + b_bits, total_labels))
    return pigment_idx, binder_idx, mixture_idx


# ---------------------------
# matriz por muestra (general)
# ---------------------------
def _cm_per_sample(y_true_block: np.ndarray,
                   y_pred_prob_block: np.ndarray,
                   labels: List[str],
                   save_path: Optional[str],
                   title: str):
    """
    Construye una CM por muestra:
      - true class  = índice de la propia muestra
      - pred class  = muestra cuyo y_true_block es más parecido a y_pred_prob_block (argmax similitud)
    """
    # similitud (producto punto)
    sim = y_pred_prob_block @ y_true_block.T
    y_true_cls = np.arange(y_true_block.shape[0])
    y_pred_cls = np.argmax(sim, axis=1)

    cm_counts = confusion_matrix(y_true_cls, y_pred_cls, labels=list(range(len(labels))))
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_percent = np.nan_to_num(cm_counts.astype(float) / np.maximum(row_sums, 1))

    fig_w = max(8, min(0.5 * len(labels), 40))
    fig_h = fig_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp = ConfusionMatrixDisplay(cm_percent, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f", colorbar=True)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0,  ha="right",  fontsize=8)
    plt.tight_layout()

    saved_to = None
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        saved_to = save_path
    plt.close(fig)

    return {"cm_counts": cm_counts, "cm_percent": cm_percent, "saved_to": saved_to}


# ---------------------------
# 1) MATRIZ GLOBAL (S1..)
# ---------------------------
def plot_confusion_matrix_by_vector(
    df,
    y_pred_prob: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,      # no se usa, queda por compatibilidad
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Matriz de confusión global **SUAVE** por muestra (S1..Sn).
    Para cada clase real k = (p,b,m) promediamos los *scores de combinación*
    hacia cada combinación j y normalizamos la fila.
    """
    import numpy as _np
    import pandas as _pd
    import matplotlib.pyplot as _plt
    from sklearn.metrics import ConfusionMatrixDisplay

    assert y_true.ndim == 2 and y_pred_prob.shape == y_true.shape

    # --- deduce bloques (robusto, sin hardcodear 16) ---
    total_labels = y_true.shape[1]
    # Heurística estable: 4 mixtures (en tus datos), resto binder+pigment.
    # Si cambian los datos, ajusta aquí o pásame binder/mix bits desde config.
    n_mix = 4
    n_pig = max(1, total_labels - n_mix - 2)  # asegura >=2 binders
    n_bind = total_labels - n_pig - n_mix

    P0, P1 = 0, n_pig
    B0, B1 = P1, P1 + n_bind
    M0, M1 = B1, B1 + n_mix
    assert M1 == total_labels, "Los límites de bloques no suman el total"

    # --- clases reales por muestra (p,b,m) ---
    true_p = _np.argmax(y_true[:, P0:P1], axis=1)
    true_b = _np.argmax(y_true[:, B0:B1], axis=1)
    true_m = _np.argmax(y_true[:, M0:M1], axis=1)
    combos_true = list(zip(true_p, true_b, true_m))

    # clases únicas (orden de aparición)
    combos_unique = []
    for c in combos_true:
        if c not in combos_unique:
            combos_unique.append(c)
    n_classes = len(combos_unique)

    # --- scores de combinación usando la MEDIA de probs por bloque ---
    prob_p = y_pred_prob[:, P0:P1]
    prob_b = y_pred_prob[:, B0:B1]
    prob_m = y_pred_prob[:, M0:M1]

    def combo_score_j(p_idx, b_idx, m_idx):
        return (prob_p[:, p_idx] + prob_b[:, b_idx] + prob_m[:, m_idx]) / 3.0
        # Alternativa más estricta:
        # return prob_p[:, p_idx] * prob_b[:, b_idx] * prob_m[:, m_idx]

    # matriz suave: fila k = media de scores hacia cada j, normalizada
    cm_percent = _np.zeros((n_classes, n_classes), dtype=float)
    for k, (pk, bk, mk) in enumerate(combos_unique):
        mask_k = _np.array([c == (pk, bk, mk) for c in combos_true], dtype=bool)
        if not mask_k.any():
            continue
        scores_k = []
        for j, (pj, bj, mj) in enumerate(combos_unique):
            s = combo_score_j(pj, bj, mj)[mask_k]  # scores hacia j en las muestras clase k
            scores_k.append(float(s.mean()))
        row = _np.array(scores_k, dtype=float)
        ssum = row.sum()
        if ssum > 0:
            row = row / ssum
        cm_percent[k, :] = row

    # --- etiquetas S1.. y leyenda ---
    s_labels = [f"S{i+1}" for i in range(n_classes)]
    fig_w = max(8, min(0.6 * n_classes, 30))
    fig_h = max(6, min(0.6 * n_classes, 30))

    fig, ax = _plt.subplots(figsize=(fig_w, fig_h))
    disp = ConfusionMatrixDisplay(cm_percent, display_labels=s_labels)
    disp.plot(cmap=_plt.cm.Blues, ax=ax, values_format=".2f", colorbar=True)
    ax.set_title("Global Confusion (per-sample %)", fontsize=14)
    _plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=9)
    _plt.setp(ax.get_yticklabels(), rotation=0,  ha="right",  fontsize=9)
    _plt.tight_layout()

    saved_to = None
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _plt.savefig(save_path, dpi=130, bbox_inches="tight")
        saved_to = save_path
    _plt.close(fig)

    legend_rows = []
    for i, (pk, bk, mk) in enumerate(combos_unique):
        pos = combos_true.index((pk, bk, mk))
        r   = df.iloc[pos]
        legend_rows.append({
            "Sample": f"S{i+1}",
            "File":   r.get("File", "?"),
            "Binder": r.get("Binder", "?"),
            "Mixture":r.get("Mixture", "?"),
            "p_idx": int(pk), "b_idx": int(bk), "m_idx": int(mk),
        })
    legend_path = None
    if save_path:
        legend_path = os.path.splitext(save_path)[0] + "_legend.csv"
        _pd.DataFrame(legend_rows).to_csv(legend_path, index=False)

    return {
        "labels": s_labels,
        "cm_percent": cm_percent,
        "saved_to": saved_to,
        "legend": legend_path,
    }

# ---------------------------
# 2) MATRICES POR BLOQUE (P/B/M)
# ---------------------------
def plot_confusion_matrix_block_per_sample(
    y_true_grouped: np.ndarray,
    y_pred_prob_grouped: np.ndarray,
    block_idx: List[int],
    prefix: str,
    save_path: Optional[str] = None,
):
    if not block_idx:
        return {"saved_to": None}
    yt = y_true_grouped[:, block_idx]
    yp = y_pred_prob_grouped[:, block_idx]
    labels = [f"{prefix}{i+1}" for i in range(yt.shape[0])]
    return _cm_per_sample(
        y_true_block=yt,
        y_pred_prob_block=yp,
        labels=labels,
        save_path=save_path,
        title=f"Confusion (per-sample %) – {prefix} block"
    )


# ---------------------------
# 3) MÉTRICAS
# ---------------------------
def print_global_and_per_group_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    y_pred_bin: np.ndarray,
    pigment_idx: List[int],
    binder_idx: List[int],
    mixture_idx: List[int],
    out_dir: str,
    report_prefix: str = "test"
):
    os.makedirs(out_dir, exist_ok=True)

    def _safe_auc(yt, yp):
        try:    return float(roc_auc_score(yt, yp, average='micro'))
        except: return float('nan')

    def _safe_ap(yt, yp):
        try:    return float(average_precision_score(yt, yp, average='micro'))
        except: return float('nan')

    def _safe_ll(yt, yp):
        try:    return float(log_loss(yt, yp))
        except: return float('nan')

    def _accs(yt, yb):
        return {
            "strict":      float(np.mean(np.all(yt == yb, axis=1))),
            "general":     float(accuracy_score(yt, yb)),
            "keras_like":  float((yt == yb).mean(axis=1).mean()),
            "hamming_acc": float(1.0 - hamming_loss(yt, yb)),
        }

    def _prf(yt, yb):
        return {
            "f1":        float(f1_score(yt, yb, average='micro', zero_division=0)),
            "precision": float(precision_score(yt, yb, average='micro', zero_division=0)),
            "recall":    float(recall_score(yt, yb, average='micro', zero_division=0)),
        }

    # ---- GLOBAL ----
    g_acc = _accs(y_true, y_pred_bin)
    g_prf = _prf(y_true, y_pred_bin)
    g_auc = _safe_auc(y_true, y_pred_prob)
    g_ap  = _safe_ap(y_true, y_pred_prob)
    g_ll  = _safe_ll(y_true, y_pred_prob)
    df_global = pd.DataFrame([{**g_acc, **g_prf, "roc_auc_micro": g_auc, "avg_precision_micro": g_ap, "log_loss": g_ll}])
    p_global_csv = os.path.join(out_dir, f"{report_prefix}_global_metrics.csv")
    df_global.to_csv(p_global_csv, index=False)

    # ---- PER-GROUP TABLE + reports por bloque ----
    rows = []
    for name, idx in [("pigments", pigment_idx), ("binders", binder_idx), ("mixtures", mixture_idx)]:
        if not idx:
            rows.append({"scope": name, "note": "empty"}); continue
        yt, yp, yb = y_true[:, idx], y_pred_prob[:, idx], y_pred_bin[:, idx]
        acc = _accs(yt, yb); prf = _prf(yt, yb); auc = _safe_auc(yt, yp); ap = _safe_ap(yt, yp); ll = _safe_ll(yt, yp)
        rows.append({"scope": name, **acc, **prf, "roc_auc_micro": auc, "avg_precision_micro": ap, "log_loss": ll})

        rep = classification_report(yt, yb, output_dict=True, zero_division=0)
        pd.DataFrame(rep).transpose().to_csv(os.path.join(out_dir, f"{report_prefix}_classification_report_{name}.csv"))
        with open(os.path.join(out_dir, f"{report_prefix}_classification_report_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(classification_report(yt, yb, zero_division=0))

    df_group = pd.DataFrame(rows)
    p_group_csv = os.path.join(out_dir, f"{report_prefix}_per_group_metrics.csv")
    df_group.to_csv(p_group_csv, index=False)

    # ---- classification_report GLOBAL ----
    rep_global = classification_report(y_true, y_pred_bin, output_dict=True, zero_division=0)
    pd.DataFrame(rep_global).transpose().to_csv(os.path.join(out_dir, f"{report_prefix}_classification_report.csv"))
    with open(os.path.join(out_dir, f"{report_prefix}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred_bin, zero_division=0))

    print("💾 Guardado:",
          os.path.join(out_dir, f"{report_prefix}_global_metrics.csv"), "|",
          os.path.join(out_dir, f"{report_prefix}_per_group_metrics.csv"), "|",
          os.path.join(out_dir, f"{report_prefix}_classification_report*.csv/txt"))
    return df_global, df_group
