import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, hamming_loss, roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, log_loss
)

def plot_confusion_matrix_by_vector(df, y_pred_prob, y_true, threshold=0.5):
    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    y_true_bin = y_true.astype(int)

    # Crea etiquetas legibles a partir de df["Multi"]
    def vector_to_label(vec):
        row = df[df['Multi'].apply(lambda x: tuple(x)==tuple(vec))]
        if row.empty: return "Unknown"
        r = row.iloc[0]
        return f"{r.get('File','?')}, {r.get('Binder','?')}, {r.get('Mixture','?')}"

    pred_labels = [vector_to_label(v) for v in y_pred_bin]
    true_labels = [vector_to_label(v) for v in y_true_bin]

    labels = np.unique(true_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm_pct = np.nan_to_num(cm.astype(float)/cm.sum(axis=1, keepdims=True))

    fig_w = max(18, 0.6*len(labels))
    fig_h = max(12, 0.6*len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ConfusionMatrixDisplay(cm_pct, display_labels=labels).plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title("Confusion Matrix (pred vs true)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=90, ha='center'); plt.yticks(rotation=0, ha='right')
    plt.subplots_adjust(left=0.30, bottom=0.1)
    plt.tight_layout()
    plt.show()

def print_global_and_per_group_metrics(y_true, y_pred_prob, y_pred_bin,
                                       pigment_idx, binder_idx, mixture_idx):
    print("\n🟢 CLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred_bin, zero_division=0))

    roc_auc = roc_auc_score(y_true, y_pred_prob, average='micro')
    avg_precision = average_precision_score(y_true, y_pred_prob, average='micro')
    f1 = f1_score(y_true, y_pred_bin, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred_bin, average='micro', zero_division=0)
    precision = precision_score(y_true, y_pred_bin, average='micro', zero_division=0)
    try: ll = log_loss(y_true, y_pred_prob)
    except ValueError: ll = float('nan')

    print("\n🔹 Global:")
    print(f"F1={f1:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  ROC-AUC={roc_auc:.4f}  AP={avg_precision:.4f}  LogLoss={ll:.4f}")

    def block(name, idx):
        if not idx: 
            print(f"\n{name}: (sin clases)"); return
        yt = y_true[:, idx]; yb = y_pred_bin[:, idx]; yp = y_pred_prob[:, idx]
        try: auc = roc_auc_score(yt, yp, average='micro')
        except ValueError: auc = float('nan')
        f1b = f1_score(yt, yb, average='micro', zero_division=0)
        rb  = recall_score(yt, yb, average='micro', zero_division=0)
        pb  = precision_score(yt, yb, average='micro', zero_division=0)
        try: llb = log_loss(yt, yp)
        except ValueError: llb = float('nan')
        print(f"\n🔸 {name}: F1={f1b:.4f}  Prec={pb:.4f}  Rec={rb:.4f}  AUC={auc:.4f}  LogLoss={llb:.4f}")

    block("Pigments", pigment_idx)
    block("Binders",  binder_idx)
    block("Mixtures", mixture_idx)
