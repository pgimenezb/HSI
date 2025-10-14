# hsi_lab/train.py
import os, json, importlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import h5py

from hsi_lab.config import variables
from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.eval.report import (
    print_global_and_per_group_metrics,
    plot_confusion_matrix_by_vector,
)

OPTUNA_STORAGE = f"sqlite:///{os.path.abspath('outputs/optuna.db')}"

# ---------------------------
# utilidades comunes
# ---------------------------
def ensure_dirs(cfg: dict) -> None:
    os.makedirs(cfg["outputs_dir"], exist_ok=True)
    os.makedirs(cfg["models_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["outputs_dir"], "figures"), exist_ok=True)
    os.makedirs("outputs", exist_ok=True)  # para optuna.db

def prepare_Xy(df: pd.DataFrame):
    spec_cols = [c for c in df.columns if c.startswith(("vis_", "swir_", "val_vis_", "val_swir_"))]
    if not spec_cols:
        raise ValueError("No spectral columns starting with vis_/swir_ o val_vis_/val_swir_")
    spec_cols = sorted(spec_cols, key=lambda c: (0 if c.startswith(("vis_", "val_vis_")) else 1, c))
    X = df[spec_cols].astype(np.float32).fillna(0.0).values[..., np.newaxis]
    y = np.array([np.array(v) for v in df["Multi"]])
    return X, y

def _to_py_scalars(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            try:
                out[k] = v.item(); continue
            except Exception:
                pass
        out[k] = v
    return out

def _save_h5_tree(h5_path: str, out_txt: str):
    with h5py.File(h5_path, "r") as h5:
        lines = []
        def visitor(name, obj):
            import h5py as _h5
            if isinstance(obj, _h5.Dataset):
                lines.append(f"[DATASET] {name}  shape={obj.shape}  dtype={obj.dtype}")
            else:
                lines.append(f"[GROUP]   {name}")
        h5.visititems(lambda n, o: visitor(n, o))
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))

def _save_artifacts_keras_like(model, study, best_params: dict, cfg: dict, tag: str):
    # guarda solo si el modelo expone .save (Keras). Para sklearn-like se omite.
    if hasattr(model, "save"):
        h5_path = os.path.join(cfg["models_dir"], f"{tag}.h5")
        keras_path = os.path.join(cfg["models_dir"], f"{tag}.keras")
        try:
            model.save(h5_path)
            model.save(keras_path)
            _save_h5_tree(h5_path, os.path.join(cfg["outputs_dir"], f"{tag}_h5_tree.txt"))
        except Exception as e:
            print(f"(info) No se pudo guardar como Keras/H5: {e}")

    # hiperparámetros
    pd.DataFrame([best_params]).to_csv(os.path.join(cfg["outputs_dir"], f"{tag}_best_params.csv"), index=False)

    # resumen de study
    try:
        df_trials = study.trials_dataframe(
            attrs=("number", "value", "state", "params", "datetime_start", "datetime_complete")
        )
        df_trials.to_csv(os.path.join(cfg["outputs_dir"], f"{tag}_trials_summary.csv"), index=False)
    except Exception as e:
        print("(info) No se pudo exportar trials_summary.csv:", e)

# ---------------------------
# core: ejecutar un modelo
# ---------------------------
def run_one_model(model_name: str, df: pd.DataFrame, cfg: dict,
                  trials: int, epochs: int, optuna_n_jobs: int,
                  do_reports: bool = False):
    print(f"\n=== ▶ {model_name} ===")

    # split único y estable
    X, y = prepare_Xy(df)
    X_temp, X_test, y_temp, y_test = train_test_split(X[...,0], y, test_size=0.3, random_state=42, shuffle=True)
    X_train, X_val,  y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
    X_train = X_train[...,np.newaxis]; X_val = X_val[...,np.newaxis]; X_test = X_test[...,np.newaxis]

    # import dinámico
    mod = importlib.import_module(f"hsi_lab.models.{model_name}")
    tune = getattr(mod, "tune_and_train")

    # cada modelo reanuda su propio estudio en la misma DB
    study_name = f"{model_name}_study"
    model, study, best = tune(
        X_train, y_train, X_val, y_val,
        input_len=X_train.shape[1], num_classes=y.shape[1],
        trials=trials, epochs=epochs,
        storage=OPTUNA_STORAGE, study_name=study_name, n_jobs=optuna_n_jobs
    )

    # evaluación homogénea
    results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    strict_acc = float(results.get("strict_accuracy", float("nan")))
    print(f"✓ {model_name}: strict_acc_test={strict_acc:.4f}")

    # reportes completos (gráficas, CM) solo si quieres un modelo "principal"
    if do_reports:
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
            y_true=y_test, y_pred_prob=y_pred_prob, y_pred_bin=y_pred_bin,
            pigment_idx=pigment_idx, binder_idx=binder_idx, mixture_idx=mixture_idx,
        )
        try:
            plot_confusion_matrix_by_vector(df, y_pred_prob, y_test)
            fig_path = os.path.join(cfg["outputs_dir"], "figures", f"{model_name}_confusion_matrix.png")
            plt.savefig(fig_path, dpi=130, bbox_inches="tight")
            plt.close()
            print("✅ Guardada figura:", fig_path)
        except Exception as e:
            print("(info) No se pudo dibujar/guardar la matriz de confusión:", e)

    # artefactos ligeros siempre
    tag = f"{model_name}"
    _save_artifacts_keras_like(model, study, _to_py_scalars(best), cfg, tag)

    # resumen JSON
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "strict_accuracy_test": strict_acc,
        "best_params": _to_py_scalars(best),
        "n_test": int(y_test.shape[0]),
        "n_labels": int(y_test.shape[1]),
    }
    with open(os.path.join(cfg["outputs_dir"], f"{model_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary

# ---------------------------
# main
# ---------------------------
def main():
    cfg = variables.copy()
    ensure_dirs(cfg)

    # datos 1 vez + caché
    processor = HSIDataProcessor(cfg)
    processor.load_h5_files()
    df = processor.dataframe_cached()

    # ¿uno o varios modelos?
    model_list = cfg.get("model_list", ["cnn"])
    trials = int(cfg.get("trials", 50))
    epochs = int(cfg.get("epochs", 50))
    optuna_n_jobs = int(cfg.get("optuna_n_jobs", 4))
    n_jobs_models = int(cfg.get("n_jobs_models", 1))

    # si solo hay uno, hacemos el flujo “completo” (con reportes)
    if len(model_list) == 1:
        summary = run_one_model(model_list[0], df, cfg, trials, epochs, optuna_n_jobs, do_reports=True)
        print("\n=== RESUMEN ===")
        print(f"- {summary['model']}: strict_acc_test={summary['strict_accuracy_test']:.4f}")
        return

    # si hay varios, los ejecutamos (en serie o paralelo) con la MISMA partición
    # (misma df, y el split se hace dentro de run_one_model con random_state fijo)
    results = Parallel(n_jobs=n_jobs_models)(
        delayed(run_one_model)(m, df, cfg, trials, epochs, optuna_n_jobs, do_reports=False)
        for m in model_list
    )

    print("\n=== RESUMEN ===")
    for r in results:
        print(f"- {r['model']}: strict_acc_test={r['strict_accuracy_test']:.4f}")

if __name__ == "__main__":
    main()
