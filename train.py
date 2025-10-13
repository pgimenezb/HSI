import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import h5py
import tensorflow as tf

from hsi_lab.config import variables
from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.models.cnn import tune_and_train
from hsi_lab.models.metrics import strict_accuracy
from hsi_lab.eval.report import (
    print_global_and_per_group_metrics,
    plot_confusion_matrix_by_vector,
)


def ensure_dirs(cfg: dict) -> None:
    """
    Crea las carpetas de salida configuradas.
    models_dir y outputs_dir pueden ser rutas anidadas, os.makedirs crea intermedias.
    """
    os.makedirs(cfg["outputs_dir"], exist_ok=True)
    os.makedirs(cfg["models_dir"], exist_ok=True)
    # por comodidad, una subcarpeta para figuras
    os.makedirs(os.path.join(cfg["outputs_dir"], "figures"), exist_ok=True)


def prepare_Xy(df: pd.DataFrame):
    """
    Extrae X (espectros) y y (multilabel) del DataFrame unificado.
    Acepta prefijos vis_/swir_ o val_vis_/val_swir_.
    """
    spec_cols = [c for c in df.columns if c.startswith(("vis_", "swir_", "val_vis_", "val_swir_"))]
    if not spec_cols:
        raise ValueError("No spectral columns found starting with 'vis_/swir_' or 'val_vis_/val_swir_'")

    # Ordena: primero VIS luego SWIR
    spec_cols = sorted(spec_cols, key=lambda c: (0 if c.startswith(("vis_", "val_vis_")) else 1, c))

    X = df[spec_cols].astype(np.float32).fillna(0.0).values  # (N, C)
    y = np.array([np.array(label) for label in df["Multi"]])  # (N, L)

    # Añade eje canal para la CNN: (N, C, 1)
    X = X[..., np.newaxis]
    return X, y


def save_artifacts(model, study, best_params: dict, cfg: dict) -> None:
    """
    Guarda pesos/arquitectura y artefactos ligeros de inspección.
    - Pesos: cfg["models_dir"]/Model_Test2.h5 y .keras
    - CSV/MD/TXT: cfg["outputs_dir"]
    """
    # Rutas de modelos
    h5_path = os.path.join(cfg["models_dir"], "Model_Test2.h5")
    keras_path = os.path.join(cfg["models_dir"], "Model_Test2.keras")
    model.save(h5_path)
    model.save(keras_path)

    # Hiperparámetros (CSV)
    pd.DataFrame([best_params]).to_csv(os.path.join(cfg["outputs_dir"], "best_params.csv"), index=False)

    # Resumen del modelo (TXT)
    with open(os.path.join(cfg["outputs_dir"], "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))

    # Árbol del archivo HDF5 (TXT)
    with h5py.File(h5_path, "r") as h5:
        lines = []

        def visitor(name, obj):
            import h5py as _h5
            if isinstance(obj, _h5.Dataset):
                lines.append(f"[DATASET] {name}  shape={obj.shape}  dtype={obj.dtype}")
            else:
                lines.append(f"[GROUP]   {name}")

        h5.visititems(lambda n, o: visitor(n, o))

    with open(os.path.join(cfg["outputs_dir"], "model_h5_tree.txt"), "w") as f:
        f.write("\n".join(lines))

    # Resumen de trials (CSV)
    try:
        df_trials = study.trials_dataframe(
            attrs=("number", "value", "state", "params", "datetime_start", "datetime_complete")
        )
        df_trials.to_csv(os.path.join(cfg["outputs_dir"], "trials_summary.csv"), index=False)
    except Exception as e:
        print("No se pudo exportar trials_summary.csv:", e)


def _to_py_scalars(d: dict) -> dict:
    """
    Convierte posibles tipos de numpy (np.int64, np.float32, etc.) a tipos nativos de Python
    para que JSON no se queje al volcar best_params.
    """
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            try:
                out[k] = v.item()
                continue
            except Exception:
                pass
        out[k] = v
    return out


def main():
    # Carga config y asegura carpetas
    cfg = variables.copy()
    ensure_dirs(cfg)

    # === DATOS ===
    processor = HSIDataProcessor(cfg)
    processor.load_h5_files()
    df = processor.dataframe()

    # Split (70/15/15). Quitamos eje canal y se lo volvemos a añadir tras el split.
    X, y = prepare_Xy(df)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X[..., 0], y, test_size=0.3, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    )
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # === ENTRENAR (tuning + fit con mejores params) ===
    model, study, best = tune_and_train(
        X_train,
        y_train,
        X_val,
        y_val,
        input_len=X_train.shape[1],
        num_classes=y.shape[1],
        trials=30,
        epochs=50,
    )

    # === EVALUAR ===
    results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    print(f"🧪 Test Strict Accuracy: {results.get('strict_accuracy', float('nan')):.4f}")

    # === REPORTES (métricas agregadas + matriz de confusión) ===
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)

    num_pigments = int(cfg["num_files"])
    total_labels = y_pred_bin.shape[1]

    pigment_idx = list(range(0, min(num_pigments, total_labels)))
    binder_idx = list(range(num_pigments, min(num_pigments + 2, total_labels)))

    sel_regions = cfg.get("selected_regions", [])
    has_mixture = all(r not in [5, 6] for r in sel_regions) if sel_regions else True
    mixture_idx = list(range(num_pigments + 2, min(num_pigments + 4, total_labels))) if has_mixture else []

    print_global_and_per_group_metrics(
        y_true=y_test,
        y_pred_prob=y_pred_prob,
        y_pred_bin=y_pred_bin,
        pigment_idx=pigment_idx,
        binder_idx=binder_idx,
        mixture_idx=mixture_idx,
    )

    # Dibuja matriz de confusión por vector y guárdala
    fig_saved = False
    try:
        plot_confusion_matrix_by_vector(df, y_pred_prob, y_test)
        fig_path = os.path.join(cfg["outputs_dir"], "figures", "confusion_matrix.png")
        plt.savefig(fig_path, dpi=130, bbox_inches="tight")
        plt.close()
        print("✅ Guardada figura:", fig_path)
        fig_saved = True
    except Exception as e:
        print("No se pudo dibujar/guardar la matriz de confusión:", e)

    # === ARTEFACTOS ===
    save_artifacts(model, study, best, cfg)

    # === RESUMEN LIGERO EN JSON/MD EN other_outputs ===
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strict_accuracy_test": float(results.get("strict_accuracy", "nan")),
        "best_params": _to_py_scalars(best),
        "n_test": int(y_test.shape[0]),
        "n_labels": int(y_test.shape[1]),
        "confusion_matrix_png": (os.path.join("figures", "confusion_matrix.png") if fig_saved else None),
    }

    metrics_path = os.path.join(cfg["outputs_dir"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    report_md = os.path.join(cfg["outputs_dir"], "reports.md")
    with open(report_md, "a") as f:
        f.write(f"\n## Run {summary['timestamp']}\n")
        f.write(f"- Strict Accuracy (test): **{summary['strict_accuracy_test']:.4f}**\n")
        f.write(f"- Best params: `{summary['best_params']}`\n")
        if fig_saved:
            f.write(f"- Confusion matrix: `{summary['confusion_matrix_png']}`\n")


if __name__ == "__main__":
    main()
