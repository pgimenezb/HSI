# train.py
import os, json, importlib, argparse, traceback
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from hsi_lab.data.config import variables
from hsi_lab.data.processor import HSIDataProcessor

# intenta usar helpers del repo si existen
try:
    from hsi_lab.eval import report as rpt
except Exception:
    rpt = None


# ------------------------------- #
# Utils CLI
# ------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento HSI genérico (modelos, agrupado y límites por grupo).")
    p.add_argument("--models", type=str, required=True,
                   help="Módulos en hsi_lab.models (coma-separados). Ej: cnn_baseline,cnn_residual")
    p.add_argument("--group-by", type=str, default=None,
                   help="Columna del DataFrame para limitar por grupo (ej: Region, Subregion, Mixture, File)")
    p.add_argument("--per-group-limit", type=int, default=0,
                   help="Máximo de filas por grupo antes del split. 0 = sin límite")
    p.add_argument("--per-group-limit-map", type=str, default=None,
                   help="Mapa de límites por grupo. Ej: '1=300,2=100' o JSON '{\"1\":300,\"2\":100}'")
    return p.parse_args()


def parse_limit_map(s: str):
    if not s:
        return None
    s = s.strip()
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        raw = json.loads(s)
        if not isinstance(raw, dict):
            raise ValueError("--per-group-limit-map JSON debe ser un objeto {grupo: limite}")
        return {str(k): int(v) for k, v in raw.items()}
    out = {}
    for part in s.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Formato inválido en '{part}'. Usa 'grupo=limite'.")
        k, v = part.split("=", 1)
        out[str(k).strip()] = int(v)
    return out


def limit_per_group(df, group_col, default_limit=0, limit_map=None, seed=42):
    if not group_col or group_col not in df.columns:
        return df
    rng = np.random.default_rng(seed)
    parts = []
    for g, dfg in df.groupby(group_col, dropna=False):
        key = str(g)
        lim = None
        if isinstance(limit_map, dict) and key in limit_map:
            lim = int(limit_map[key])
        elif default_limit and default_limit > 0:
            lim = int(default_limit)

        if not lim or lim <= 0 or len(dfg) <= lim:
            parts.append(dfg)
        else:
            idx = rng.choice(dfg.index.values, size=lim, replace=False)
            parts.append(dfg.loc[sorted(idx)])

    df_out = pd.concat(parts, ignore_index=False).sort_values(["File","Y","X"], kind="mergesort").reset_index(drop=True)
    return df_out


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# ------------------------------- #
# Helpers locales (fallback)
# ------------------------------- #
def _first_one_idx(s: str) -> int:
    pos = [i for i, c in enumerate(s) if c == "1"]
    return pos[0] if pos else 0


def binarize_probs(probs, threshold=0.5):
    return (probs >= threshold).astype(int)


def _mixture_idx_to_name(variables):
    mk = variables["mixture_mapping"]  # {"1000":"Pure",...}
    keys_sorted = sorted(mk.keys(), key=_first_one_idx)
    return [mk[k] for k in keys_sorted]  # 0..3


def comparative_table(df_test, y_true, y_pred, variables):
    mix_names = _mixture_idx_to_name(variables)
    pig_true = y_true[:, :20].argmax(axis=1)
    pig_pred = y_pred[:, :20].argmax(axis=1)
    mix_true_idx = y_true[:, 20:].argmax(axis=1)
    mix_pred_idx = y_pred[:, 20:].argmax(axis=1)

    mix_true = [mix_names[i] for i in mix_true_idx]
    mix_pred = [mix_names[i] for i in mix_pred_idx]

    is_pure_true = [1 if name == "Pure" else 0 for name in mix_true]
    is_pure_pred = [1 if name == "Pure" else 0 for name in mix_pred]

    out = pd.DataFrame({
        "File": df_test["File"].values,
        "Spectrum": df_test["Spectrum"].values,
        "Pigment_True": pig_true,
        "Pigment_Pred": pig_pred,
        "Mixture_True": mix_true,
        "Mixture_Pred": mix_pred,
        "PureTrue(1)/Mix(0)": is_pure_true,
        "PurePred(1)/Mix(0)": is_pure_pred,
    })
    return out


def save_comparative_csv(df, out_dir):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "comparative.csv")
    df.to_csv(path, index=False)
    return path


def confusion_pure_vs_mix(y_true, y_pred, save_path=None):
    t = (y_true[:, 20:].argmax(axis=1) == 0).astype(int)  # 1 si Pure
    p = (y_pred[:, 20:].argmax(axis=1) == 0).astype(int)
    cm = confusion_matrix(t, p, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Mix", "Pure"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title("Confusion Matrix: Pure vs Mix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def confusion_mixture4(y_true, y_pred, variables, save_path=None):
    labels = _mixture_idx_to_name(variables)
    t = y_true[:, 20:].argmax(axis=1)
    p = y_pred[:, 20:].argmax(axis=1)
    cm = confusion_matrix(t, p, labels=range(4))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title("Confusion Matrix: Mixture (4 classes)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def confusion_pigments(y_true, y_pred, pigment_names=None, save_path=None):
    n = 20
    labels = pigment_names or [f"Pig_{i}" for i in range(n)]
    t = y_true[:, :n].argmax(axis=1)
    p = y_pred[:, :n].argmax(axis=1)
    cm = confusion_matrix(t, p, labels=range(n))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=90)
    ax.set_title("Confusion Matrix: Pigments")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ------------------------------- #
# Main
# ------------------------------- #
def main():
    args = parse_args()
    model_names = [s.strip() for s in args.models.split(",") if s.strip()]
    group_by = args.group_by
    per_group_limit = int(args.per_group_limit or 0)
    limit_map = parse_limit_map(args.per_group_limit_map) if args.per_group_limit_map else None

    # 1) Datos
    processor = HSIDataProcessor(variables)
    processor.load_h5_files()

    df = processor.dataframe()
    if group_by:
        if group_by not in df.columns:
            raise ValueError(f"--group-by '{group_by}' no existe en el DataFrame. Columnas disponibles: {list(df.columns)}")
        df = limit_per_group(df, group_by, default_limit=per_group_limit, limit_map=limit_map, seed=variables.get("seed", 42))
        vc = df[group_by].value_counts(dropna=False).sort_index()
        print(f"✓ Límites aplicados por '{group_by}':")
        print(vc.to_string())
        print(f"Total ahora: {len(df)}")

    # X, y
    make_xy_fn = getattr(processor, "make_Xy", None) or getattr(processor, "make_xy", None)
    if make_xy_fn is None:
        raise AttributeError("HSIDataProcessor no expone make_Xy/make_xy. Actualiza processor.py con el método.")
    X, _, _, y_multi = make_xy_fn(df)

    # split 70/15/15
    from sklearn.model_selection import train_test_split
    n = len(X)
    idx_all = np.arange(n)
    idx_tmp, idx_test, y_tmp, y_test = train_test_split(idx_all, y_multi, test_size=0.30,
                                                        random_state=variables.get("seed", 42), shuffle=True)
    idx_train, idx_val, y_train, y_val = train_test_split(idx_tmp, y_tmp, test_size=0.50,
                                                          random_state=variables.get("seed", 42), shuffle=True)
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    df_test = df.iloc[idx_test].reset_index(drop=True)

    # 2) Ejecuta cada modelo
    for module_name in model_names:
        print(f"\n=== Modelo: {module_name} ===")
        out_dir_base = ensure_dir(os.path.join(
            variables["outputs_dir"],
            f"{module_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        ))

        try:
            mod  = importlib.import_module(f"hsi_lab.models.{module_name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No encuentro el módulo 'hsi_lab.models.{module_name}'. "
                f"Asegúrate de que exista 'hsi_lab/models/{module_name}.py'."
            ) from e

        if not hasattr(mod, "tune_and_train"):
            raise AttributeError(f"El módulo '{module_name}' no expone 'tune_and_train'.")

        tune = getattr(mod, "tune_and_train")

        # hiperparámetros de entrenamiento desde config
        trials         = variables.get("trials", 10)
        epochs         = variables.get("epochs", 10)
        OPTUNA_STORAGE = None
        optuna_n_jobs  = variables.get("optuna_n_jobs", 1)

        study_name = f"{module_name}_study_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            model, study, best = tune(
                X_train, y_train, X_val, y_val,
                input_len=X_train.shape[1], num_classes=y_train.shape[1],
                trials=trials, epochs=epochs, storage=OPTUNA_STORAGE,
                study_name=study_name, n_jobs=optuna_n_jobs
            )

            # 3) Predicción + tabla comparativa
            y_pred_prob = model.predict(X_test, verbose=0)
            _bin = getattr(rpt, "binarize_probs", binarize_probs) if rpt else binarize_probs
            y_pred_bin  = rpt.binarize_probs(y_pred_prob, threshold=0.5)

            _comp = getattr(rpt, "comparative_table", comparative_table) if rpt else comparative_table
            comp_df = rpt.comparative_table(df_test, y_test.astype(int), y_pred_bin, variables)
            print("\n🔎 Predichos vs Reales (primeras filas):")
            print(comp_df.head())

            # matrices
            rpt.confusion_pure_vs_mix(y_test.astype(int), y_pred_bin,
                                    save_path=os.path.join(out_dir_base, "cm_pure_vs_mix.png"))

            rpt.confusion_mixture4(y_test.astype(int), y_pred_bin, variables,
                                save_path=os.path.join(out_dir_base, "cm_mixture_4class.png"))

            # si no pasas pigment_names, se generan automáticamente con tamaño variables["num_files"]
            rpt.confusion_pigments(y_test.astype(int), y_pred_bin, variables,
                                save_path=os.path.join(out_dir_base, "cm_pigments.png"))

            _save_comp = getattr(rpt, "save_comparative_csv", save_comparative_csv) if rpt else save_comparative_csv
            comp_csv = _save_comp(comp_df, out_dir_base)

            # 5) Guardados
            model_path = os.path.join(out_dir_base, "model.h5")
            model.save(model_path)

            if study is not None:
                import joblib
                joblib.dump(study, os.path.join(out_dir_base, "optuna_study.pkl"))

            summary = {
                "module": module_name,
                "best_hyperparameters": best,
                "args": {
                    "group_by": group_by,
                    "per_group_limit": per_group_limit,
                    "per_group_limit_map": limit_map,
                },
                "artifacts": {
                    "model": model_path,
                    "comparative_csv": comp_csv,
                    "cm_pure_vs_mix": os.path.join(out_dir_base, "cm_pure_vs_mix.png"),
                    "cm_mixture_4class": os.path.join(out_dir_base, "cm_mixture_4class.png"),
                    "cm_pigments": os.path.join(out_dir_base, "cm_pigments.png"),
                },
            }
            with open(os.path.join(out_dir_base, "summary.json"), "w") as f:
                json.dump(summary, f, indent=4)

            print(f"✓ Guardado en: {out_dir_base}")

        except Exception as e:
            print(f"✗ Error con el modelo '{module_name}': {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
