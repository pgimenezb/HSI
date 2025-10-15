# hsi_lab/train.py
import os, json, importlib, sys, contextlib
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
    plot_confusion_matrix_by_vector,
    print_global_and_per_group_metrics,
)

OPTUNA_STORAGE = None

# ------------- Tee de logs por modelo -------------
class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

@contextlib.contextmanager
def tee_to_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old_out, old_err = sys.stdout, sys.stderr
    with open(path, "a") as f:
        sys.stdout = _TeeIO(old_out, f)
        sys.stderr  = _TeeIO(old_err, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

# ------------- Utilidades comunes -------------
def ensure_dirs(cfg: dict) -> None:
    os.makedirs(cfg["outputs_dir"], exist_ok=True)
    os.makedirs(cfg["models_dir"],  exist_ok=True)
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
        h5_path    = os.path.join(cfg["models_dir"], f"{tag}.h5")
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

# ------------- CLI & ENV overrides -------------
import argparse as _argparse
import os as _os

def _parse_args():
    p = _argparse.ArgumentParser()
    p.add_argument("--models", type=str, default="", help="Lista separada por comas (p.ej. cnn_baseline,dnn_wide)")
    p.add_argument("--trials", type=int, default=None, help="Optuna trials (override)")
    p.add_argument("--epochs", type=int, default=None, help="Epochs (override)")
    p.add_argument("--optuna-n-jobs", type=int, default=None, help="Paralelismo interno de Optuna")
    p.add_argument("--n-jobs-models", type=int, default=None, help="Paralelismo entre modelos (Joblib)")
    p.add_argument("--reports", action="store_true", help="Guardar figuras/CM")
    p.add_argument("--run-id", type=str, default=None, help="Identificador de la ejecución (carpeta)")
    p.add_argument("--group-by", type=str, default=None,
                   help="Columna(s) para agrupar (p.ej. Subregion o 'Subregion,Pigment').")
    p.add_argument("--per-group-limit", type=int, default=None,
                   help="Máximo de filas por grupo definido por --group-by.")
    p.add_argument("--limit-rows", type=int, default=None,
                   help="Usa solo N filas totales (muestra aleatoria, reproducible).")
    p.add_argument("--sample-frac", type=float, default=None,
                   help="Usa una fracción del dataframe (0-1). Ignorado si --limit-rows está presente.")
    return p.parse_args()

def _resolve_list(value_from_cli: str, value_from_env: str, value_from_cfg):
    if value_from_cli:
        return [x.strip() for x in value_from_cli.split(",") if x.strip()]
    if value_from_env:
        return [x.strip() for x in value_from_env.split(",") if x.strip()]
    return value_from_cfg

# ------------- Core: ejecutar un modelo -------------
def run_one_model(model_name: str, df: pd.DataFrame, cfg: dict,
                  trials: int, epochs: int, optuna_n_jobs: int,
                  do_reports: bool = False):
    # log por modelo
    logs_dir = cfg.get("logs_dir", cfg["outputs_dir"])
    log_path = os.path.join(logs_dir, f"{model_name}.log")

    with tee_to_file(log_path):
        print(f"\n=== ▶ {model_name} ===")

        # split único y estable
        X, y = prepare_Xy(df)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X[..., 0], y, test_size=0.3, random_state=42, shuffle=True
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
        )
        X_train = X_train[..., np.newaxis]
        X_val   = X_val[..., np.newaxis]
        X_test  = X_test[..., np.newaxis]

        # import dinámico del modelo
        mod  = importlib.import_module(f"hsi_lab.models.{model_name}")
        tune = getattr(mod, "tune_and_train")

        # estudio Optuna único por lanzamiento+modelo
        study_name = f"{model_name}_study_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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

        # reportes completos (opcionales)
        if do_reports:
            try:
                # --- Predicciones ---
                y_pred_prob = model.predict(X_test, verbose=0)
                y_pred_bin  = (y_pred_prob >= 0.5).astype(int)

                # --- Confusion Matrix por vector completo ---
                fig_path = os.path.join(cfg["outputs_dir"], "figures", f"{model_name}_confusion_matrix.png")
                plot_confusion_matrix_by_vector(df, y_pred_prob, y_test, save_path=fig_path, show=False)
                print("✅ Guardada figura:", fig_path)

                # --- Índices de bloques (20 pigmentos, B binders, 4 mixtures) ---
                total_labels = y_test.shape[1]
                n_pigments   = 20
                n_mixtures   = 4
                n_binders    = total_labels - n_pigments - n_mixtures
                if n_binders < 0:
                    raise ValueError(f"Dimensiones de y incompatibles: total={total_labels}, esperado >= {n_pigments + n_mixtures}")

                pigment_idx = list(range(0, n_pigments))
                binder_idx  = list(range(n_pigments, n_pigments + n_binders))
                mixture_idx = list(range(n_pigments + n_binders, n_pigments + n_binders + n_mixtures))

                # --- Métricas globales y por bloque (tablas + classification_report a disco) ---
                other_out_dir = "projects/HSI/outputs/other_outputs"
                report_prefix = f"{model_name}_test"

                df_global, df_groups = print_global_and_per_group_metrics(
                    y_true      = y_test.astype(int),
                    y_pred_prob = y_pred_prob,
                    y_pred_bin  = y_pred_bin,
                    pigment_idx = pigment_idx,
                    binder_idx  = binder_idx,
                    mixture_idx = mixture_idx,
                    out_dir     = other_out_dir,
                    report_prefix = report_prefix
                )

                print("✅ Tablas de métricas guardadas en:", other_out_dir)

            except Exception as e:
                print("(info) No se pudieron generar reportes/figuras/metricas:", e)

        # artefactos
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
            "log_file": log_path,
        }
        with open(os.path.join(cfg["outputs_dir"], f"{model_name}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"📝 Log guardado en: {log_path}")
        return summary

# ------------- main -------------
def main():
    # === CONFIG + CLI/ENV ===
    cfg  = variables.copy()
    args = _parse_args()

    # RUN_ID (compartido por todos los modelos de este lanzamiento)
    run_id = args.run_id or _os.getenv("HSI_RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg["run_id"] = run_id

    # redirige outputs a subcarpetas por ejecución
    cfg["outputs_dir"] = _os.path.join(cfg["outputs_dir"], "runs", run_id)
    cfg["models_dir"]  = _os.path.join(cfg["models_dir"],  run_id)
    ensure_dirs(cfg)
    logs_dir = os.path.join(cfg["outputs_dir"], "logs")
    os.makedirs(logs_dir, exist_ok=True)
    cfg["logs_dir"] = logs_dir  # para usarla en run_one_model

    # OPTUNA_STORAGE (DB por ejecución)
    global OPTUNA_STORAGE
    OPTUNA_STORAGE = f"sqlite:///{os.path.abspath(os.path.join(cfg['outputs_dir'], 'optuna.db'))}"

    # datos 1 vez (usa caché si la tienes implementada)
    processor = HSIDataProcessor(cfg)
    processor.load_h5_files()
    df = processor.dataframe_cached()  # o processor.dataframe()

    # ---- limitar tamaño del df para pruebas rápidas ----
    group_by_env = _os.getenv("HSI_GROUP_BY")
    per_group_limit_env = _os.getenv("HSI_PER_GROUP_LIMIT")

    group_by_arg = args.group_by or group_by_env
    per_group_limit = (
        args.per_group_limit
        if args.per_group_limit is not None
        else (int(per_group_limit_env) if per_group_limit_env else None)
    )

    if group_by_arg and per_group_limit and per_group_limit > 0:
        group_cols = [c.strip() for c in group_by_arg.split(",") if c.strip()]
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            print(f"(warn) --group-by contiene columnas inexistentes: {missing}. Se ignora muestreo por grupo.")
        else:
            orig_n = len(df)
            # muestreo reproducible por grupo
            def _sample_g(g):
                n = min(per_group_limit, len(g))
                return g.sample(n=n, random_state=42)
            df = (df.groupby(group_cols, group_keys=False).apply(_sample_g).reset_index(drop=True))
            n_groups = df[group_cols].drop_duplicates().shape[0]
            print(f"(info) per-group limit={per_group_limit} por {group_cols} → df: {orig_n} → {len(df)} filas "
                  f"({n_groups} grupos)")

    # Selección automática (CLI > ENV > config)
    env_models = _os.getenv("HSI_MODELS", "")
    model_list = _resolve_list(args.models, env_models, cfg.get("model_list", ["cnn_baseline"]))

    trials        = args.trials        if args.trials        is not None else int(cfg.get("trials", 50))
    epochs        = args.epochs        if args.epochs        is not None else int(cfg.get("epochs", 50))
    optuna_n_jobs = args.optuna_n_jobs if args.optuna_n_jobs is not None else int(cfg.get("optuna_n_jobs", 4))
    n_jobs_models = args.n_jobs_models if args.n_jobs_models is not None else int(cfg.get("n_jobs_models", 1))

    # si solo hay uno, hacemos el flujo “completo” (con reportes)
    if len(model_list) == 1:
        summary = run_one_model(model_list[0], df, cfg, trials, epochs, optuna_n_jobs, do_reports=True)
        print("\n=== RESUMEN ===")
        print(f"- {summary['model']}: strict_acc_test={summary['strict_accuracy_test']:.4f}")
        return

    # si hay varios, los ejecutamos (en serie o en paralelo) con la MISMA partición
    results = Parallel(n_jobs=n_jobs_models)(
        delayed(run_one_model)(m, df, cfg, trials, epochs, optuna_n_jobs, do_reports=args.reports)
        for m in model_list
    )

    print("\n=== RESUMEN ===")
    for r in results:
        print(f"- {r['model']}: strict_acc_test={r['strict_accuracy_test']:.4f}")

if __name__ == "__main__":
    main()
