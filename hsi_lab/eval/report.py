# report.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, ConfusionMatrixDisplay  
from hsi_lab.data.config import variables
from mpl_toolkits.axes_grid1 import make_axes_locatable

def convert_materials(binary, pigment_map=None):
    s = ''.join(map(str, binary))    # Normalizes into string
    n = int(variables["num_files"])  # nº bits pigment
    mmap = variables["mixture_mapping"]
    pigments = s[:n]
    # binder = s[n:n+2]  # oculto
    mixture  = s[n:n+4]  # 4 bits justo tras pigment
    pigment_name = (pigment_map or {}).get(pigments, "Pigment unknown")
    mixture_name = mmap.get(mixture, "Mixture unknown")
    return f"{pigment_name}, {mixture_name}"

def confusion_matrix(y_pred, y_test):
    pred_labels = [convert_materials(val) for val in y_pred.astype(int)]
    test_labels = [convert_materials(val) for val in y_test.astype(int)]
    conf_matrix = sk_confusion_matrix(test_labels, pred_labels, labels=np.unique(test_labels))  # ← usa alias
    conf_matrix_percent = conf_matrix.astype(float) / conf_matrix.sum(axis=1)[:, np.newaxis]
    print("Confusion matrix (normalized in %):")
    print(np.round(conf_matrix_percent * 100, 2))

def confusion_matrix_per_material(y_pred, y_true, pigment_names, num_files, save_dir=None, prefix="model"):
    classes = ["Pigment"] * num_files + ["Mixture"] * 4  # binder oculto
    groups = {
        "Pigment": range(num_files),
        "Mixture": range(num_files, num_files + 4)       # ← FIX: eran indices erróneos
    }
    labels = {
        "Pigment": pigment_names,
        "Mixture": ["Pure", "Mixture1", "Mixture2", "Mixture3"]
    }
    for name_group, index in groups.items():
        y_true_grp = y_true[:, index]
        y_pred_grp = y_pred[:, index]
        y_true_labels = y_true_grp.argmax(axis=1)
        y_pred_labels = y_pred_grp.argmax(axis=1)
        out_path = None
        if save_dir:
            out_path = os.path.join(save_dir, f"{prefix}_confusion_matrix_{name_group.lower()}.png")
        plot_confusion_matrix(
            y_true_labels,
            y_pred_labels,
            [labels[name_group][i] for i in range(len(index))],
            title=f"Confusion matrix per {name_group}",
            out_path=out_path
        )

# === plot_confusion_matrix (1 decimal, muestra 0.0, más espacio) ===
def plot_confusion_matrix(cm, classes, title, out_path,
                   annotate_percent=False,      # mantenemos decimales 0.00–1.00
                   cmap_name="Blues",
                   figsize=None,
                   min_font=5, max_font=11):    # rango auto del texto

    n = len(classes)

    # 1) Tamaño de figura a partir de nº de clases (celda ~0.35")
    if figsize is None:
        cell = 0.35                     # ancho/alto por celda en pulgadas
        side = max(8, min(36, cell * n))
        figsize = (side, side)

    # 2) Figura compacta (constrained_layout elimina blancos)
    fig, ax = plt.subplots(figsize=figsize, dpi=180, constrained_layout=True)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap_name, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=13, pad=8)

    # ticks y etiquetas
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=90, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.tick_params(axis="x", which="major", pad=6)
    ax.tick_params(axis="y", which="major", pad=4)

    # 3) Colorbar estrecho pegado a la matriz (sin comer área)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.3)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=9)

    # 4) Celdas cuadradas y texto adaptativo
    ax.set_aspect("equal", adjustable="box")
    # Tamaño del texto: decrece con n, pero dentro de [min_font, max_font]
    font_size = max(min_font, min(max_font, int(200 / max(n, 1))))
    # Anotar TODAS las celdas con 2 decimales (sin símbolo %)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = (cm[i, j] * 100.0) if annotate_percent else cm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=font_size)

    # 5) Guardar recortando el extra
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[OK] {title} -> {out_path}")


# === Helpers de decodificación desde el vector Multi ===
def _decode_pigment_and_group(y_like: np.ndarray, n_p: int):
    pig = np.argmax(y_like[:, :n_p], axis=1)
    mix_idx = np.argmax(y_like[:, n_p:n_p+4], axis=1)  # 0=Pure, 1..3=Mixtures
    group = np.where(mix_idx==0, "Pure", "Mixture")
    return np.array([f"P{p+1:02d}_{g}" for p,g in zip(pig, group)])

def _decode_pigment_and_mix4(y_like: np.ndarray, n_p: int):
    pig = np.argmax(y_like[:, :n_p], axis=1)
    mix_idx = np.argmax(y_like[:, n_p:n_p+4], axis=1)  # 0=Pure,1=M1,2=M2,3=M3
    names = np.array(["Pure","M1","M2","M3"])
    return np.array([f"P{p+1:02d}_{names[m]}" for p,m in zip(pig, mix_idx)])

def _decode_mix_group(y_like: np.ndarray, n_p: int):
    mix_idx = np.argmax(y_like[:, n_p:n_p+4], axis=1)
    return np.where(mix_idx==0, "Pure", "Mixture")

# === 2N: Pxx_{Pure,Mixture} ===
def cm_pigment_mix2N(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    n_p = int(variables["num_files"])
    y_true_lbl = _decode_pigment_and_group((y_true > 0.5).astype(int), n_p)
    y_pred_lbl = _decode_pigment_and_group(y_pred, n_p)
    classes = [f"P{i+1:02d}_Pure" for i in range(n_p)] + [f"P{i+1:02d}_Mixture" for i in range(n_p)]
    idx = {c:i for i,c in enumerate(classes)}
    ti = np.array([idx.get(x,-1) for x in y_true_lbl]); pi = np.array([idx.get(x,-1) for x in y_pred_lbl])
    mask = (ti>=0)&(pi>=0)
    cm = sk_confusion_matrix(ti[mask], pi[mask], labels=list(range(len(classes))), normalize='true')
    plot_confusion_matrix(cm, classes, "Global (Pigment + Pure/Mixture)", out_path)

# === 4N: Pxx_{Pure,M1,M2,M3} ===
def cm_pigment_mix4N(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, present_only: bool=False):
    n_p = int(variables["num_files"])
    y_true_lbl = _decode_pigment_and_mix4((y_true > 0.5).astype(int), n_p)
    y_pred_lbl = _decode_pigment_and_mix4(y_pred, n_p)
    classes_full = [f"P{i+1:02d}_{s}" for i in range(n_p) for s in ["Pure","M1","M2","M3"]]
    vc_true = (lambda s: s.reindex(classes_full, fill_value=0))( 
        __import__("pandas").Series(y_true_lbl).value_counts()
    )
    classes = [c for c in classes_full if vc_true[c] > 0] if present_only else classes_full
    idx = {c:i for i,c in enumerate(classes_full)}
    ti = np.array([idx.get(x,-1) for x in y_true_lbl]); pi = np.array([idx.get(x,-1) for x in y_pred_lbl])
    mask = (ti>=0)&(pi>=0)
    cm = sk_confusion_matrix(ti[mask], pi[mask], labels=[idx[c] for c in classes], normalize='true')
    plot_confusion_matrix(cm, classes, "Global 4-casos (Pure/M1/M2/M3 por pigmento)", out_path)

# === 2: Pure vs Mixture (global) ===
def cm_mix_global2(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    n_p = int(variables["num_files"])
    y_true_lbl = _decode_mix_group((y_true > 0.5).astype(int), n_p)
    y_pred_lbl = _decode_mix_group(y_pred, n_p)
    classes = ["Pure","Mixture"]; idx = {"Pure":0,"Mixture":1}
    ti = np.array([idx[x] for x in y_true_lbl]); pi = np.array([idx[x] for x in y_pred_lbl])
    cm = sk_confusion_matrix(ti, pi, labels=[0,1], normalize='true')
    plot_confusion_matrix(cm, classes, "Mixture Only (Pure vs Mixture)", out_path)


# ahora añado una funcion que me guarde las matrices de confusion en PNG:


# --- MINI helper para la GLOBAL (Pigment + Mixture) → PNG, reusa tu plot_confusion_matrix
def save_confusion_pngs(y_true, y_pred, pigment_names, out_dir, prefix="model", threshold=None):
    # 1) Respeta el tipo de y_pred; si piden umbral, aplícalo
    if threshold is not None and y_pred.dtype.kind in "fc":
        y_pred_use = (y_pred >= threshold).astype(int)
    else:
        y_pred_use = y_pred.astype(int)

    # 2) GLOBAL: etiquetas solo desde y_true (como tu función original)
    true_labels = [convert_materials(v) for v in y_true.astype(int)]
    pred_labels = [convert_materials(v) for v in y_pred_use.astype(int)]
    labels_true = sorted(set(true_labels))  # = np.unique(test_labels)

    lab2idx = {lab: i for i, lab in enumerate(labels_true)}
    yt_idx = np.array([lab2idx[x] for x in true_labels])
    # OJO: si hay una etiqueta predicha que no está en y_true, la descartamos para mantener la forma
    yp_idx = np.array([lab2idx[x] for x in pred_labels if x in lab2idx])
    # Alinea longitudes si hiciste filtrado (opcionalmente descarta esas muestras en yt_idx también)
    mask = np.array([x in lab2idx for x in pred_labels])
    yt_idx = yt_idx[mask]

    os.makedirs(out_dir, exist_ok=True)
    plot_confusion_matrix(
        yt_idx, yp_idx, labels=labels_true,  # <- display_labels
        title="Global (Pigment + Mixture)",
        out_path=os.path.join(out_dir, f"{prefix}_confusion_matrix_global.png")
    )
