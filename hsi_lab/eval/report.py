import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, ConfusionMatrixDisplay
from hsi_lab.data.config import variables
import pandas as pd
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, log_loss, hamming_loss
)

def _split_heads(y: np.ndarray, n_pig: int):
    """Separa bloques: pigmentos (N, n_pig) y mixtures (N, 4) desde y (N, n_pig+4)."""
    y = np.asarray(y)
    pig = y[:, :n_pig]
    mix = y[:, n_pig:n_pig+4]
    return pig, mix

def convert_materials(binary, pigment_map=None):
    s = ''.join(map(str, binary))            # Normaliza a string
    n = int(variables["num_files"])          # nº bits de pigmento
    mmap = variables["mixture_mapping"]      # {'1000': 'Pure', ...}

    pigments = s[:n]
    mixture  = s[n:n+4]                      # 4 bits justo tras pigment

    pigment_name = (pigment_map or {}).get(pigments, "Pigment unknown")
    mixture_name = mmap.get(mixture, "Mixture unknown")
    return f"{pigment_name}, {mixture_name}"

def confusion_matrix(y_pred, y_test):
    pred_labels = [convert_materials(val) for val in y_pred.astype(int)]
    test_labels = [convert_materials(val) for val in y_test.astype(int)]
    cm = sk_confusion_matrix(test_labels, pred_labels, labels=np.unique(test_labels))
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    print("Confusion matrix (normalized in %):")
    print(np.round(cm_pct * 100, 2))

def _plot_confusion_matrix(y_true_idx, y_pred_idx, class_names, title, out_path=None, ax=None):
    K = len(class_names)
    cm = sk_confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(K)))
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)

    disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=80)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()




def _mixture_labels_from_config():
    mmap = variables["mixture_mapping"]
    order = ["1000", "0100", "0010", "0001"]
    default = ["Mixture1", "Mixture2", "Mixture3", "Mixture4"]
    labels = []
    for i, b in enumerate(order):
        labels.append(mmap.get(b, default[i]))
    return labels

def confusion_matrix_per_material(y_pred, y_true, pigment_names, num_files, save_dir=None, prefix="model", legend_lines=None):
    pig_true = y_true[:, :num_files]
    pig_pred = y_pred[:, :num_files]
    y_true_pig_idx = pig_true.argmax(axis=1)
    y_pred_pig_idx = pig_pred.argmax(axis=1)

    out_path = os.path.join(save_dir, f"{prefix}_cm_pigment.png") if save_dir else None
    _plot_confusion_matrix(y_true_pig_idx, y_pred_pig_idx, pigment_names,
                           "Confusion matrix per Pigment", out_path, legend_lines=legend_lines)

    mix_true = y_true[:, num_files : num_files + 4]
    mix_pred = y_pred[:, num_files : num_files + 4]
    mix_names = ["Pure", "Mixture1", "Mixture2", "Mixture3"]

    y_true_mix_idx = mix_true.argmax(axis=1)
    y_pred_mix_idx = mix_pred.argmax(axis=1)

    out_path = os.path.join(save_dir, f"{prefix}_cm_mixture.png") if save_dir else None
    _plot_confusion_matrix(y_true_mix_idx, y_pred_mix_idx, mix_names,
                           "Confusion matrix per Mixture", out_path, legend_lines=legend_lines)



def _derive_labels(y_mat, n_pig):
    """
    y_mat: (N, n_pig+4) con probabilidades o one-hot por bloques.
    Devuelve:
      yt_pig, yp_pig : indices [0..n_pig-1]
      yt_mix, yp_mix : indices [0..3]
      yt_global, yp_global : indices [0..2*n_pig-1] donde global = pig*2 + is_mix
    """
    n = y_mat.shape[1]
    assert n == n_pig + 4, f"Esperaba n_pig+4 columnas, recibí {n}."

    # y_mat puede ser “true”/one-hot o probabilidades predichas.
    # Para TRUE usamos argmax igual (one-hot -> índice correcto).
    pig = y_mat[:, :n_pig]
    mix = y_mat[:, n_pig:n_pig+4]

    idx_pig = pig.argmax(axis=1)
    idx_mix = mix.argmax(axis=1)
    is_mix = (idx_mix != 0).astype(int)   # 0: Pure, 1: {Mixture1,2,3}
    global_idx = idx_pig * 2 + is_mix
    return idx_pig, idx_mix, global_idx

def _ensure_concat(y_pred_out, n_pig):
    """Acepta (pig_prob, mix_prob) o ya-concatenado y devuelve (N, n_pig+4)."""
    if isinstance(y_pred_out, (list, tuple)):
        pig_prob, mix_prob = y_pred_out
        return np.concatenate([pig_prob, mix_prob], axis=1)
    return y_pred_out

def save_confusion_pngs(y_true, y_pred, pigment_names, out_dir, prefix="model"):
    n_pig = int(variables["num_files"])
    pig_true, mix_true = _split_heads(y_true, n_pig)
    pig_pred, mix_pred = _split_heads(y_pred, n_pig)

    yt_pig = pig_true.argmax(axis=1)
    yp_pig = pig_pred.argmax(axis=1)
    yt_mix = mix_true.argmax(axis=1)
    yp_mix = mix_pred.argmax(axis=1)

    # Global: Pure (idx 0) vs Mixture (idx 1/2/3)
    yt_is_mix = (yt_mix != 0).astype(int)
    yp_is_mix = (yp_mix != 0).astype(int)
    yt_global = yt_pig * 2 + yt_is_mix
    yp_global = yp_pig * 2 + yp_is_mix

    # Ticks Pxx_pure / Pxx_mixture
    class_names = []
    for i in range(n_pig):
        ptag = f"P{(i+1):02d}"
        class_names.append(f"{ptag}_pure")
        class_names.append(f"{ptag}_mixture")

    os.makedirs(out_dir, exist_ok=True)
    _plot_confusion_matrix(
        yt_global, yp_global, class_names=class_names,
        title="Global (Pigment × {Pure/Mixture})",
        out_path=os.path.join(out_dir, f"{prefix}_cm_global.png")
    )

    # Per Pigment
    _plot_confusion_matrix(
        yt_pig, yp_pig, class_names=[f"P{(i+1):02d}" for i in range(n_pig)],
        title="Confusion matrix per Pigment",
        out_path=os.path.join(out_dir, f"{prefix}_cm_pigment.png")
    )

    # Per Mixture
    mix_names = _mixture_labels_from_config()  # ["Pure","Mixture1","Mixture2","Mixture3"]
    _plot_confusion_matrix(
        yt_mix, yp_mix, class_names=mix_names,
        title="Confusion matrix per Mixture",
        out_path=os.path.join(out_dir, f"{prefix}_cm_mixture.png")
    )

# ---------- NUEVO: CSV amplio con TRUE vs PRED (pesos) para Pxx_pure / Pxx_mixture ----------
def export_global_table(y_true, y_pred, pigment_names, out_csv):
    """
    Crea una tabla por muestra con:
    - true_pig / pred_pig
    - true_mix (Pure/Mix1/Mix2/Mix3) y flag true_is_mix / pred_is_mix
    """
    n_pig = int(variables["num_files"])
    mix_names = _mixture_labels_from_config()

    pig_t, mix_t = _split_heads(y_true, n_pig)
    pig_p, mix_p = _split_heads(y_pred, n_pig)

    tp = pig_t.argmax(axis=1)
    pp = pig_p.argmax(axis=1)
    tm = mix_t.argmax(axis=1)
    pm = mix_p.argmax(axis=1)

    true_is_mix = (tm != 0).astype(int)
    pred_is_mix = (pm != 0).astype(int)

    rows = []
    for i in range(len(tp)):
        rows.append({
            "idx": i,
            "true_pig_idx": int(tp[i]),
            "pred_pig_idx": int(pp[i]),
            "true_pig_name": pigment_names[int(tp[i])] if 0 <= tp[i] < len(pigment_names) else f"P{tp[i]}",
            "pred_pig_name": pigment_names[int(pp[i])] if 0 <= pp[i] < len(pigment_names) else f"P{pp[i]}",
            "true_mix_idx": int(tm[i]),
            "pred_mix_idx": int(pm[i]),
            "true_mix_name": mix_names[int(tm[i])],
            "pred_mix_name": mix_names[int(pm[i])],
            "true_is_mix": int(true_is_mix[i]),
            "pred_is_mix": int(pred_is_mix[i]),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df

def evaluation_metrics(y_true, y_pred, pigment_names, threshold=0.5,
                       save_table_path=None, print_report=False):
    """
    No imprime; devuelve dict con métricas globales y guarda (si se pide) tabla por muestra.
    """
    n_pig = int(variables["num_files"])
    mix_names = _mixture_labels_from_config()

    pig_t, mix_t = _split_heads(y_true, n_pig)
    pig_p, mix_p = _split_heads(y_pred, n_pig)

    tp = pig_t.argmax(axis=1)
    pp = pig_p.argmax(axis=1)
    tm = mix_t.argmax(axis=1)
    pm = mix_p.argmax(axis=1)

    true_is_mix = (tm != 0).astype(int)
    pred_is_mix = (pm != 0).astype(int)

    # One-hot “pred” duros para métricas multilabel (concat)
    y_true_bin = np.concatenate([pig_t, mix_t], axis=1)
    y_pred_bin = np.concatenate([
        np.eye(n_pig)[pp],            # pigmento 1-de-N
        np.eye(4)[pm]                 # mixture 1-de-4
    ], axis=1)

    # Probabilidades para AUC/AP
    try:
        roc_auc = roc_auc_score(y_true_bin, np.concatenate([pig_p, mix_p], axis=1), average='micro')
    except ValueError:
        roc_auc = float('nan')
    try:
        avg_precision = average_precision_score(y_true_bin, np.concatenate([pig_p, mix_p], axis=1), average='micro')
    except ValueError:
        avg_precision = float('nan')
    try:
        ll = log_loss(y_true_bin, np.concatenate([pig_p, mix_p], axis=1))
    except ValueError:
        ll = float('nan')

    f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    prec = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    ham_acc = 1 - hamming_loss(y_true_bin, y_pred_bin)
    acc_exact = accuracy_score(y_true_bin, y_pred_bin)

    # Tabla por muestra (si se pide)
    if save_table_path:
        rows = []
        for i in range(len(tp)):
            rows.append({
                "idx": i,
                "true_pig_idx": int(tp[i]),
                "pred_pig_idx": int(pp[i]),
                "true_pig_name": pigment_names[int(tp[i])] if 0 <= tp[i] < len(pigment_names) else f"P{tp[i]}",
                "pred_pig_name": pigment_names[int(pp[i])] if 0 <= pp[i] < len(pigment_names) else f"P{pp[i]}",
                "true_mix_idx": int(tm[i]),
                "pred_mix_idx": int(pm[i]),
                "true_mix_name": mix_names[int(tm[i])],
                "pred_mix_name": mix_names[int(pm[i])],
                "true_is_mix": int(true_is_mix[i]),
                "pred_is_mix": int(pred_is_mix[i]),
                "pig_correct": int(tp[i] == pp[i]),
                "mix_correct": int(tm[i] == pm[i]),
                "global_correct": int((tp[i] == pp[i]) and ((tm[i] == 0) == (pm[i] == 0))),
            })
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(save_table_path), exist_ok=True)
        df.to_csv(save_table_path, index=False)

    return {
        "f1_micro": float(f1),
        "precision_micro": float(prec),
        "recall_micro": float(rec),
        "roc_auc_micro": float(roc_auc),
        "avg_precision_micro": float(avg_precision),
        "hamming_accuracy": float(ham_acc),
        "exact_match_accuracy": float(acc_exact),
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(y_true.shape[1]),
    }