import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from hsi_lab.config import variables
from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.models.cnn import tune_and_train
from hsi_lab.models.metrics import strict_accuracy
import tensorflow as tf
import h5py

from hsi_lab.eval.report import (
    print_global_and_per_group_metrics,
    plot_confusion_matrix_by_vector,
)

def ensure_dirs(cfg):
    os.makedirs(cfg["outputs_dir"], exist_ok=True)
    os.makedirs(cfg["models_dir"], exist_ok=True)

def prepare_Xy(df):
    # columnas espectrales (ajusta a tus prefijos reales)
    spec_cols = [c for c in df.columns if c.startswith(("vis_", "swir_", "val_vis_", "val_swir_"))]
    if not spec_cols:
        raise ValueError("No spectral columns found starting with 'vis_/swir_' or 'val_vis_/val_swir_'")
    spec_cols = sorted(spec_cols, key=lambda c: (0 if c.startswith(("vis_", "val_vis_")) else 1, c))

    X = df[spec_cols].astype(np.float32).fillna(0.0).values
    y = np.array([np.array(label) for label in df["Multi"]])
    # añadir canal
    X = X[..., np.newaxis]
    return X, y

def save_artifacts(model, study, best_params, cfg):
    # modelo
    h5_path = os.path.join(cfg["models_dir"], "Model_Test2.h5")
    keras_path = os.path.join(cfg["models_dir"], "Model_Test2.keras")
    model.save(h5_path)
    model.save(keras_path)

    # hiperparámetros
    pd.DataFrame([best_params]).to_csv(os.path.join(cfg["outputs_dir"], "best_params.csv"), index=False)

    # resumen del modelo
    with open(os.path.join(cfg["outputs_dir"], "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))

    # árbol H5
    with h5py.File(h5_path, "r") as h5:
        lines = []
        def visitor(name, obj):
            import h5py as _h5
            if isinstance(obj, _h5.Dataset):
                lines.append(f"[DATASET] {name} shape={obj.shape} dtype={obj.dtype}")
            else:
                lines.append(f"[GROUP]   {name}")
        h5.visititems(lambda n,o: visitor(n,o))
    with open(os.path.join(cfg["outputs_dir"], "model_h5_tree.txt"), "w") as f:
        f.write("\n".join(lines))

    # trials
    try:
        df_trials = study.trials_dataframe(attrs=("number","value","state","params","datetime_start","datetime_complete"))
        df_trials.to_csv(os.path.join(cfg["outputs_dir"], "trials_summary.csv"), index=False)
    except Exception as e:
        print("No se pudo exportar trials_summary.csv:", e)

def main():
    cfg = variables.copy()
    ensure_dirs(cfg)

    # datos
    processor = HSIDataProcessor(cfg)
    processor.load_h5_files()
    df = processor.dataframe()

    # split
    X, y = prepare_Xy(df)
    X_temp, X_test, y_temp, y_test = train_test_split(X[...,0], y, test_size=0.3, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
    # reañadir canal
    X_train = X_train[..., np.newaxis]; X_val = X_val[..., np.newaxis]; X_test = X_test[..., np.newaxis]

    # entrenar
    model, study, best = tune_and_train(
        X_train, y_train, X_val, y_val,
        input_len=X_train.shape[1],
        num_classes=y.shape[1],
        trials=30, epochs=50
    )

    # evaluar
    results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    print(f"🧪 Test Strict Accuracy: {results.get('strict_accuracy', float('nan')):.4f}")

    # === BLOQUE NUEVO: reportes ===
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred_bin  = (y_pred_prob >= 0.5).astype(int)

    num_pigments = variables["num_files"]
    total_labels = y_pred_bin.shape[1]

    pigment_idx = list(range(0, min(num_pigments, total_labels)))
    binder_idx  = list(range(num_pigments, min(num_pigments + 2, total_labels)))

    has_mixture = True
    sel_regions = variables.get("selected_regions", [])
    if sel_regions:
        has_mixture = all(r not in [5, 6] for r in sel_regions)

    mixture_idx = (
        list(range(num_pigments + 2, min(num_pigments + 4, total_labels)))
        if has_mixture else []
    )

    print_global_and_per_group_metrics(
        y_true=y_test,
        y_pred_prob=y_pred_prob,
        y_pred_bin=y_pred_bin,
        pigment_idx=pigment_idx,
        binder_idx=binder_idx,
        mixture_idx=mixture_idx,
    )

    try:
        plot_confusion_matrix_by_vector(df, y_pred_prob, y_test)
    except Exception as e:
        print("No se pudo dibujar la matriz de confusión por vector:", e)

    # guardar artefactos
    save_artifacts(model, study, best, cfg)


if __name__ == "__main__":
    main()
