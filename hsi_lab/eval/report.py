# ==== IMPORTS NECESARIOS ====
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =========================================================
# === PLOTEO PURO (no toca datos, solo visualización) ====
# =========================================================
def plot_confusion_matrix(cm, classes, title, out_path,
                   annotate_percent=False,      # anota como 0.00–1.00 si False, o 0.00–100.00 si True
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

    # Asumimos cm ya normalizada a [0,1] si así lo deseas; fijamos vmin/vmax visuales
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
# === CONSTRUCTORES DE MATRICES (NO TRANSFORMAN y_true / y_pred) ========
# ========================================================================
def _labels_to_indices(y_true_labels, y_pred_labels, classes=None):

    y_true_labels = np.asarray(y_true_labels)
    y_pred_labels = np.asarray(y_pred_labels)

    if classes is None:
        # orden: primero únicas de y_true en orden de aparición, luego añadimos las de y_pred que falten
        seen = []
        for x in list(y_true_labels) + list(y_pred_labels):
            if x not in seen:
                seen.append(x)
        classes = list(seen)

    idx = {c: i for i, c in enumerate(classes)}

    # Máscara: descarta pares donde la etiqueta no esté en 'classes' (no alteramos los valores, solo filtramos filas inválidas)
    valid_true = np.array([lbl in idx for lbl in y_true_labels])
    valid_pred = np.array([lbl in idx for lbl in y_pred_labels])
    mask = valid_true & valid_pred

    ti = np.array([idx[lbl] for lbl in y_true_labels[mask]])
    pi = np.array([idx[lbl] for lbl in y_pred_labels[mask]])

    return ti, pi, classes


def confusion_from_labels(y_true_labels, y_pred_labels, classes=None, normalize='true'):

    ti, pi, classes = _labels_to_indices(y_true_labels, y_pred_labels, classes=classes)
    labels_idx = list(range(len(classes)))
    cm = sk_confusion_matrix(ti, pi, labels=labels_idx, normalize=normalize)
    return cm, classes


# ====================================================================================
# === WRAPPERS ESPECÍFICOS (esperan etiquetas ya preparadas por el llamador) ========
# ====================================================================================
def cm_pigment_mix2N_VISUAL(y_true_lbl, y_pred_lbl, out_path, classes=None):
    cm, used_classes = confusion_from_labels(y_true_lbl, y_pred_lbl, classes=classes, normalize='true')
    plot_confusion_matrix(cm, used_classes, "Global (Pigment + Pure/Mixture)", out_path)


def cm_pigment_mix4N_VISUAL(y_true_lbl, y_pred_lbl, out_path, classes=None, present_only=False):
    # Si el usuario quiere limitar a presentes y no proporcionó classes, las derivamos de y_true
    if present_only and classes is None:
        classes = []
        for lab in y_true_lbl:
            if lab not in classes:
                classes.append(lab)

    cm, used_classes = confusion_from_labels(y_true_lbl, y_pred_lbl, classes=classes, normalize='true')
    plot_confusion_matrix(cm, used_classes, "Global 4-casos (Pure/M1/M2/M3 por pigmento)", out_path)


def cm_mix_global2_VISUAL(y_true_lbl, y_pred_lbl, out_path, classes=("Pure", "Mixture")):
    cm, used_classes = confusion_from_labels(y_true_lbl, y_pred_lbl, classes=list(classes), normalize='true')
    plot_confusion_matrix(cm, used_classes, "Mixture Only (Pure vs Mixture)", out_path)
