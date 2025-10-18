# train.py
import os, json, importlib, sys, contextlib
from datetime import datetime
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import h5py
import traceback

# backend headless
import matplotlib
if os.environ.get("DISPLAY","")=="":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hsi_lab.data.config import variables
from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.eval.report import (
    split_block_indices,
    plot_cm_block,
    plot_cm_samples_multilabel,
    save_metrics_csvs,
)

OPTUNA_STORAGE = None

# --- logging a fichero + consola ---
class _TeeIO:
    def __init__(self, *streams): self.streams = streams
    def write(self, data): [s.write(data) or s.flush() for s in self.streams]
    def flush(self): [s.flush() for s in self.streams]

@contextlib.contextmanager
def tee_to_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old_out, old_err = sys.stdout, sys.stderr
    with open(path, "a") as f:
        sys.stdout = _TeeIO(old_out, f); sys.stderr = _TeeIO(old_err, f)
        try: yield
        finally: sys.stdout, sys.stderr = old_out, old_err

# --- util ---
def ensure_dirs(cfg): [os.makedirs(p, exist_ok=True) for p in (cfg["outputs_dir"], cfg["logs_dir"], "outputs")]

def prepare_Xy(df: pd.DataFrame):
    spec_cols = [c for c in df.columns if c.startswith(("vis_", "swir_", "val_vis_", "val_swir_"))]
    if not spec_cols: raise ValueError("No spectral columns (vis_/swir_ o val_vis_/val_swir_)")
    spec_cols = sorted(spec_cols, key=lambda c: (0 if c.startswith(("vis_", "val_vis_")) else 1, c))
    X = df[spec_cols].astype(np.float32).fillna(0.0).values[..., np.newaxis]
    y = np.array([np.array(v) for v in df["Multi"]])
    return X, y

def _to_py_scalars(d):
    out={}
    for k,v in d.items():
        if hasattr(v,"item"):
            try: out[k]=v.item(); continue
            except: pass
        out[k]=v
    return out

def _save_h5_tree(h5_path: str, out_txt: str):
    with h5py.File(h5_path, "r") as h5:
        lines=[]
        def visitor(name, obj):
            import h5py as _h5
            lines.append(f"[DATASET] {name}  shape={obj.shape}  dtype={obj.dtype}" if isinstance(obj,_h5.Dataset) else f"[GROUP]   {name}")
        h5.visititems(lambda n,o: visitor(n,o))
    with open(out_txt,"w") as f: f.write("\n".join(lines))

def _save_artifacts_keras_like(model, study, best_params, cfg, tag):
    if hasattr(model, "save"):
        h5_path = os.path.join(cfg["models_dir"], f"{tag}.h5")
        keras_path = os.path.join(cfg["models_dir"], f"{tag}.keras")
        try:
            model.save(h5_path); model.save(keras_path)
            _save_h5_tree(h5_path, os.path.join(cfg["outputs_dir"], f"{tag}_h5_tree.txt"))
        except Exception as e:
            print(f"(info) No se pudo guardar como Keras/H5: {e}")
    pd.DataFrame([best_params]).to_csv(os.path.join(cfg["outputs_dir"], f"{tag}_best_params.csv"), index=False)
    try:
        df_trials = study.trials_dataframe(attrs=("number","value","state","params","datetime_start","datetime_complete"))
        df_trials.to_csv(os.path.join(cfg["outputs_dir"], f"{tag}_trials_summary.csv"), index=False)
    except Exception as e:
        print("(info) No se pudo exportar trials_summary.csv:", e)

# --- CLI ---
import argparse as _argparse, os as _os
def _parse_args():
    p=_argparse.ArgumentParser()
    p.add_argument("--models", type=str, default="")
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--optuna-n-jobs", type=int, default=None)
    p.add_argument("--n-jobs-models", type=int, default=None)
    p.add_argument("--reports", action="store_true")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--group-by", type=str, default=None)
    p.add_argument("--per-group-limit", type=int, default=None)
    return p.parse_args()

def _resolve_list(cli, env, cfg_value):
    if cli: return [x.strip() for x in cli.split(",") if x.strip()]
    if env: return [x.strip() for x in env.split(",") if x.strip()]
    return cfg_value

from collections import defaultdict

def prepare_Xy(df: pd.DataFrame):
    # columnas espectrales
    spec_cols = [c for c in df.columns if c.startswith(("vis_", "swir_"))]
    if not spec_cols:
        raise ValueError("No spectral columns found starting with 'vis_' or 'swir_'.")
    # vis primero, luego swir (orden estable)
    spec_cols = sorted(spec_cols, key=lambda c: (0 if c.startswith("vis_") else 1, c))

    # X con canal (L,1)
    X = df[spec_cols].astype(np.float32).fillna(0.0).values
    X = X[..., np.newaxis]

    # y multilabel desde columna "Multi"
    y = np.array([np.array(v) for v in df["Multi"]], dtype=np.float32)
    return X, y

# --- núcleo por modelo ---
from collections import defaultdict

""" 
# índices de bloque
total_labels = y_test.shape[1]
pig_idx, bind_idx, mix_idx = split_block_indices(
    total_labels, variables["binder_mapping"], variables["mixture_mapping"]
)

# matriz de confusión de CASOS (File×Mixture) con orden y ticks S#
p_S = os.path.join(cfg["outputs_dir"], f"{model_name}_cm_S.png")
cm_S, classes_S, ticks_S = plot_confusion_matrix_cases_simple(
    df_test=df_test,
    y_test=y_test,
    y_pred_prob=y_pred_prob,
    binder_mapping=variables["binder_mapping"],
    mixture_mapping=variables["mixture_mapping"],
    normalize="true",            # o None si quieres conteos
    save_path=p_S,
    file_col="File",
    pigment_idx=pig_idx,
    binder_idx=bind_idx,
    mixture_idx=mix_idx,
)
print("✅ S matrix:", p_S) """


def run_one_model(model_name, df, cfg, trials, epochs, optuna_n_jobs, do_reports=False):
    logs_dir = cfg.get("logs_dir", cfg["outputs_dir"])
    log_path = os.path.join(logs_dir, f"{model_name}.log")

    with tee_to_file(log_path):
        print(f"\n=== ▶ {model_name} ===")

        # ===== PREPARA X, y =====
        X, y = prepare_Xy(df)

        # ===== SPLIT por combinación para asegurar 1 (File, Mixture) en test =====
        group_cols = [c for c in ["File", "Mixture"] if c in df.columns]
        if len(group_cols) < 2:
            raise ValueError("Necesito columnas 'File' y 'Mixture' en df para garantizar 4×4 en test.")

        comb_key = df[group_cols].astype(str).agg(" | ".join, axis=1).values
        by_group = defaultdict(list)
        for i, k in enumerate(comb_key):
            by_group[k].append(i)

        rng = np.random.default_rng(42)
        idx_test = []
        idx_pool = []
        for _, idxs in by_group.items():
            pick = idxs[0]                # o rng.choice(idxs) si prefieres
            idx_test.append(pick)
            idx_pool.extend([j for j in idxs if j != pick])

        # añade extra para aproximar test_size ≈ 30%
        desired_test = int(0.30 * len(df))
        need_extra = max(0, min(desired_test - len(idx_test), len(idx_pool)))
        if need_extra > 0:
            extra = rng.choice(idx_pool, size=need_extra, replace=False)
            idx_test = np.unique(np.concatenate([idx_test, extra])).tolist()

        mask = np.ones(len(df), dtype=bool)
        mask[idx_test] = False
        idx_temp = np.where(mask)[0]

        # 50/50 del resto → train/val (partimos SOLO indices, no pasamos None)
        idx_train, idx_val = train_test_split(
            idx_temp, test_size=0.5, random_state=42, shuffle=True
        )

        # Subconjuntos finales
        X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
        df_test = df.iloc[idx_test].reset_index(drop=True)

        # ===== ENTRENAR =====
        mod  = importlib.import_module(f"hsi_lab.models.{model_name}")
        tune = getattr(mod, "tune_and_train")

        study_name = f"{model_name}_study_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        model, study, best = tune(
            X_train, y_train, X_val, y_val,
            input_len=X_train.shape[1], num_classes=y.shape[1],
            trials=trials, epochs=epochs, storage=OPTUNA_STORAGE,
            study_name=study_name, n_jobs=optuna_n_jobs
        )

        # ===== PREDICCIONES EN EL MISMO TEST =====
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred_bin  = (y_pred_prob >= 0.5).astype(int)
        results     = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
        strict_acc  = float(results.get("strict_accuracy", float("nan")))
        print(f"✓ {model_name}: strict_acc_test={strict_acc:.4f}")

        # ===== REPORTES =====
        if do_reports:
            try:
                print("\n📊 Generando figuras y tablas...")

                # meta de test con índice 0..n_test-1 que casa con y_test / y_pred_prob
                df_gmeta = df_test[group_cols].assign(_row=np.arange(len(df_test)))

                # agrupar por combinación
                grouped = df_gmeta.groupby(group_cols, sort=False)["_row"].apply(list).reset_index()

                # agregación por muestra (media de probs por combinación)
                y_true_g, y_pred_g, rows_meta = [], [], []
                for _, r in grouped.iterrows():
                    idxs = r["_row"]
                    yt = (y_test[idxs].mean(axis=0) >= 0.5).astype(int)
                    yp = y_pred_prob[idxs].mean(axis=0)
                    y_true_g.append(yt)
                    y_pred_g.append(yp)
                    rows_meta.append({k: r[k] for k in group_cols})

                y_true_g = np.vstack(y_true_g)
                y_pred_g = np.vstack(y_pred_g)
                df_grouped = pd.DataFrame(rows_meta)

                # índices y nombres
                from hsi_lab.eval.report import (
                    split_block_indices,
                    plot_cm_block,
                    plot_cm_samples_multilabel,
                    save_metrics_csvs,
                )
                total_labels = y_true_g.shape[1]
                pigment_idx, binder_idx, mixture_idx = split_block_indices(
                    total_labels, variables["binder_mapping"], variables["mixture_mapping"]
                )

                pigment_names = [f"Pigment {i+1}" for i in range(len(pigment_idx))]
                binder_names  = [variables["binder_mapping"]["10"], variables["binder_mapping"]["01"]]
                mixture_names = [
                    variables["mixture_mapping"]["1000"],  # Pure
                    variables["mixture_mapping"]["0100"],  # 80:20
                    variables["mixture_mapping"]["0010"],  # 50:50
                    variables["mixture_mapping"]["0001"],  # 20:80
                ]

                # --- S (multietiqueta por muestra) => S1.. orden por File y Pure→80:20→50:50→20:80 ---
                p_S = os.path.join(cfg["outputs_dir"], f"{model_name}_cm_S.png")
                plot_cm_samples_multilabel(
                    df_grouped=df_grouped,
                    y_pred_prob=y_pred_g,
                    y_true_bin=y_true_g,
                    pigment_idx=pigment_idx,
                    binder_idx=binder_idx,
                    mixture_idx=mixture_idx,
                    pigment_names=pigment_names,
                    binder_names=binder_names,
                    mixture_names=mixture_names,
                    save_path=p_S,
                    file_col="File",
                    mixture_order=mixture_names,
                )
                print("✅ S matrix:", p_S)

                # --- P ---
                p_P = os.path.join(cfg["outputs_dir"], f"{model_name}_cm_P.png")
                plot_cm_block(
                    y_true_bin=y_true_g.astype(int),
                    y_pred_prob=y_pred_g,
                    block_idx=pigment_idx,
                    tick_prefix="P",
                    class_names=pigment_names,
                    save_path=p_P,
                )
                print("✅ P matrix:", p_P)

                # --- B ---
                p_B = os.path.join(cfg["outputs_dir"], f"{model_name}_cm_B.png")
                plot_cm_block(
                    y_true_bin=y_true_g.astype(int),
                    y_pred_prob=y_pred_g,
                    block_idx=binder_idx,
                    tick_prefix="B",
                    class_names=binder_names,
                    save_path=p_B,
                )
                print("✅ B matrix:", p_B)

                # --- M ---
                p_M = os.path.join(cfg["outputs_dir"], f"{model_name}_cm_M.png")
                plot_cm_block(
                    y_true_bin=y_true_g.astype(int),
                    y_pred_prob=y_pred_g,
                    block_idx=mixture_idx,
                    tick_prefix="M",
                    class_names=mixture_names,
                    save_path=p_M,
                )
                print("✅ M matrix:", p_M)

                # --- CSVs métricas (global + por bloque) ---
                save_metrics_csvs(
                    y_true=y_true_g.astype(int),
                    y_pred_prob=y_pred_g,
                    pigment_idx=pigment_idx,
                    binder_idx=binder_idx,
                    mixture_idx=mixture_idx,
                    out_dir=cfg["outputs_dir"],
                    prefix=f"{model_name}_test"
                )
                print("✅ CSVs de métricas guardados en:", cfg["outputs_dir"])

            except Exception as e:
                print("(info) Error generando reportes:", e)
                traceback.print_exc()

        # ===== ARTEFACTOS + RESUMEN =====
        tag = f"{model_name}"
        _save_artifacts_keras_like(model, study, _to_py_scalars(best), cfg, tag)

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


# --- main ---
def main():
    cfg  = variables.copy()
    args = _parse_args()

    run_id = args.run_id or _os.getenv("HSI_RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg["run_id"]      = run_id
    cfg["outputs_dir"] = os.path.join(cfg["outputs_dir"], "runs", run_id)
    cfg["models_dir"]  = cfg["outputs_dir"]
    cfg["logs_dir"]    = cfg["outputs_dir"]
    ensure_dirs(cfg)

    global OPTUNA_STORAGE
    OPTUNA_STORAGE = f"sqlite:///{os.path.abspath(os.path.join(cfg['outputs_dir'], 'optuna.db'))}"

    processor = HSIDataProcessor(cfg)
    processor.load_h5_files()
    df = processor.dataframe_cached()

    # muestreo opcional por grupo
    if args.group_by and args.per_group_limit:
        cols = [c.strip() for c in args.group_by.split(",") if c.strip()]
        if all(c in df.columns for c in cols):
            rng = np.random.RandomState(42)
            df = (
                df.assign(_rnd=rng.rand(len(df)))
                  .sort_values("_rnd")
                  .groupby(cols, group_keys=False)
                  .head(args.per_group_limit)
                  .drop(columns="_rnd")
                  .reset_index(drop=True)
            )

    model_list    = _resolve_list(args.models, _os.getenv("HSI_MODELS",""), cfg.get("model_list", ["cnn_baseline"]))
    trials        = args.trials        if args.trials        is not None else int(cfg.get("trials", 50))
    epochs        = args.epochs        if args.epochs        is not None else int(cfg.get("epochs", 50))
    optuna_n_jobs = args.optuna_n_jobs if args.optuna_n_jobs is not None else int(cfg.get("optuna_n_jobs", 4))
    n_jobs_models = args.n_jobs_models if args.n_jobs_models is not None else int(cfg.get("n_jobs_models", 1))

    if len(model_list) == 1:
        summary = run_one_model(model_list[0], df, cfg, trials, epochs, optuna_n_jobs, do_reports=True)
        print("\n=== RESUMEN ===")
        print(f"- {summary['model']}: strict_acc_test={summary['strict_accuracy_test']:.4f}")
        return

    results = Parallel(n_jobs=n_jobs_models)(
        delayed(run_one_model)(m, df, cfg, trials, epochs, optuna_n_jobs, do_reports=args.reports)
        for m in model_list
    )
    print("\n=== RESUMEN ===")
    for r in results:
        print(f"- {r['model']}: strict_acc_test={r['strict_accuracy_test']:.4f}")


if __name__ == "__main__":
    main()
