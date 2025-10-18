# hsi_lab/eval/report.py
import os
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

# backend headless
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# =============== 1) Utilidades etiquetas ===============
def binarize_probs(y_pred_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_pred_prob >= threshold).astype(int)

def _num_pigments(variables: dict) -> int:
    """Número de pigmentos según configuración."""
    return int(variables.get("num_files", 20))

def extract_pigment_idx(y_bin: np.ndarray, variables: dict) -> np.ndarray:
    """Índice [0..N_PIG-1] del pigmento (argmax de las N_PIG primeras columnas)."""
    N_PIG = _num_pigments(variables)
    return y_bin[:, :N_PIG].argmax(axis=1)

def extract_mixture_idx(y_bin: np.ndarray) -> np.ndarray:
    """Índice [0..3] de mezcla (argmax de las 4 últimas columnas)."""
    return y_bin[:, -4:].argmax(axis=1)

def mixture_idx_to_name(idx: np.ndarray, variables: dict) -> List[str]:
    # Orden canónico por posición del '1': 1000,0100,0010,0001
    mk = variables["mixture_mapping"]  # p.ej. {"1000":"Pure",...}
    order = sorted(mk.keys(), key=lambda s: s.index("1"))
    return [mk[order[int(i)]] for i in idx]

def mixture_is_pure(idx: np.ndarray) -> np.ndarray:
    """1 si la mezcla es 'Pure' (índice 0), 0 si es mezcla."""
    return (idx == 0).astype(int)  # 1 = Pure, 0 = Mix

# =============== 2) Tabla comparativa ===============
def comparative_table(df_rows: pd.DataFrame, y_true_bin: np.ndarray, y_pred_bin: np.ndarray, variables: dict) -> pd.DataFrame:
    """
    Construye una tabla con:
      File, Spectrum, Pigment True/Pred (índices),
      Mixture True/Pred (nombres), y Pure/Mix True/Pred (binario).
    """
    p_true = extract_pigment_idx(y_true_bin, variables)
    p_pred = extract_pigment_idx(y_pred_bin, variables)

    m_true_idx = extract_mixture_idx(y_true_bin)
    m_pred_idx = extract_mixture_idx(y_pred_bin)
    m_true = mixture_idx_to_name(m_true_idx, variables)
    m_pred = mixture_idx_to_name(m_pred_idx, variables)

    pure_true = mixture_is_pure(m_true_idx)  # 1=Pure, 0=Mix
    pure_pred = mixture_is_pure(m_pred_idx)

    out = pd.DataFrame({
        "File": df_rows["File"].values,
        "Spectrum": df_rows["Spectrum"].values,
        "Pigment_True": p_true,
        "Pigment_Pred": p_pred,
        "Mixture_True": m_true,
        "Mixture_Pred": m_pred,
        "PureTrue(1)/Mix(0)": pure_true,
        "PurePred(1)/Mix(0)": pure_pred,
    })
    return out

# =============== 3) Matrices de confusión ===============
def _plot_confusion(y_true_lbl: np.ndarray, y_pred_lbl: np.ndarray, label_names: List[str], title: str, save_path: Optional[str] = None):
    cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=range(len(label_names)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig_w = max(8, 0.6 * len(label_names))
    fig_h = max(6, 0.6 * len(label_names))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=label_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f")
    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def confusion_pigments(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, variables: dict, pigment_names: Optional[List[str]] = None, save_path: Optional[str] = None):
    y_true_lbl = extract_pigment_idx(y_true_bin, variables)
    y_pred_lbl = extract_pigment_idx(y_pred_bin, variables)
    if pigment_names is None:
        N_PIG = _num_pigments(variables)
        pigment_names = [f"Pigment_{i}" for i in range(N_PIG)]
    _plot_confusion(y_true_lbl, y_pred_lbl, pigment_names, "Confusion Matrix – Pigments", save_path)

def confusion_mixture4(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, variables: dict, save_path: Optional[str] = None):
    y_true_lbl = extract_mixture_idx(y_true_bin)
    y_pred_lbl = extract_mixture_idx(y_pred_bin)
    # Orden por posición del '1'
    order_names = [variables["mixture_mapping"][k] for k in sorted(variables["mixture_mapping"].keys(), key=lambda s: s.index("1"))]
    _plot_confusion(y_true_lbl, y_pred_lbl, order_names, "Confusion Matrix – Mixture (4 classes)", save_path)

def confusion_pure_vs_mix(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, save_path: Optional[str] = None):
    """Binario: Pure (1) vs Mix (0)."""
    y_true_lbl = mixture_is_pure(extract_mixture_idx(y_true_bin))  # 1/0
    y_pred_lbl = mixture_is_pure(extract_mixture_idx(y_pred_bin))
    _plot_confusion(y_true_lbl, y_pred_lbl, ["Mix", "Pure"], "Confusion Matrix – Pure vs Mixture", save_path)

# =============== 4) Guardados auxiliares ===============
def save_comparative_csv(df_comp: pd.DataFrame, out_dir: str, name: str = "comparative_predictions.csv"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df_comp.to_csv(path, index=False)
    return path
