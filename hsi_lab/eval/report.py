import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    y_true_bin = (y_true.astype(int))

    # --- Construcción de etiquetas legibles por vector ---
    # Creamos un mapa: tuple(vector) -> "File, Binder, Mixture"
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
                # por si hay filas sin 'Multi' válida
                continue

    # Función de ayuda: convierte vector -> etiqueta legible
    def vector_to_label(vec):
        key = tuple(vec.tolist() if hasattr(vec, "tolist") else vec)
        return vec_to_label.get(key, "Unknown")

    # Construimos las secuencias de etiquetas (una por muestra)
    pred_labels = [vector_to_label(v) for v in y_pred_bin]
    true_labels = [vector_to_label(v) for v in y_true_bin]

    # --- Clases presentes: usa el conjunto de labels verdaderos (orden estable) ---
    # Para ordenar de forma estable, tomamos el orden de aparición en true_labels.
    seen = {}
    for lab in true_labels:
        if lab not in seen:
            seen[lab] = True
    labels = np.array(list(seen.keys()), dtype=object)
    if labels.size == 0:
        # Fallback: si no hubo etiquetas válidas en true_labels, usa unión de ambos
        labels = np.unique(true_labels + pred_labels)

    # --- Matriz de confusión (conteos) ---
    cm_counts = confusion_matrix(true_labels, pred_labels, labels=labels)

    # --- Normalización por fila (porcentaje) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_percent = cm_counts.astype(float) / np.maximum(row_sums, 1)
        cm_percent = np.nan_to_num(cm_percent)

    # --- Dibujo ---
    n_labels = len(labels)
    # tamaño dinámico (limita algo para no generar figuras gigantescas)
    fig_w = max(10, min(0.6 * n_labels, 40))
    fig_h = max(8,  min(0.6 * n_labels, 40))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp = ConfusionMatrixDisplay(cm_percent, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f", colorbar=True)

    ax.set_title("Confusion Matrix (row-normalized: %)", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    # Ejes legibles
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0,  ha="right",  fontsize=9)

    # Márgenes y layout
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
