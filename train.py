import os
import argparse
import importlib
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from hsi_lab.data.config import variables
from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.eval.report import plot_confusion_matrix
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Carga dinámica del trainer
# ─────────────────────────────────────────────────────────────────────────────
def import_model_trainer(name: str):
    """
    Carga una función `tune_and_train` desde:
      - el módulo exacto si `name` contiene puntos (p.ej. "pkg.subpkg.mod"), o
      - "hsi_lab.models.<name>" si es un alias corto (p.ej. "DNN").
    """
    candidates = [name] if "." in name else []
    candidates.append(f"hsi_lab.models.{name}")
    last_err = None
    for mod_path in candidates:
        try:
            mod = importlib.import_module(mod_path)
            return getattr(mod, "tune_and_train")
        except (ModuleNotFoundError, AttributeError) as e:
            last_err = e
    raise ImportError(
        f"No se pudo cargar `tune_and_train` para '{name}'. Probé: {candidates}. "
        f"Último error: {last_err}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Features/targets
# ─────────────────────────────────────────────────────────────────────────────
def build_Xy(df: pd.DataFrame):
    """
    X: columnas espectrales 'vis_*' y 'swir_*' (vis primero), float32, shape (N, F, 1)
    y: vector Multi tal cual (pigmentos + 4 mixtures), float32
    """
    spec_cols = [c for c in df.columns if c.startswith(("vis_", "swir_"))]
    if not spec_cols:
        raise ValueError("No spectral columns starting with 'vis_' or 'swir_'.")
    spec_cols = sorted(spec_cols, key=lambda c: (0 if c.startswith("vis_") else 1, c))
    X = df[spec_cols].astype(np.float32).fillna(0.0).values[..., np.newaxis]
    y = np.array([np.array(v) for v in df["Multi"]], dtype=np.float32)
    return X, y, X.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# Split estratificado 70/15/15 por pigmento
# ─────────────────────────────────────────────────────────────────────────────
def pigment_ids(df: pd.DataFrame, vars_: dict) -> np.ndarray:
    n_p = int(vars_["num_files"])
    pig = []
    for v in df["Multi"]:
        a = np.asarray(v, dtype=np.float32)
        pig.append(int(np.argmax(a[:n_p])))
    return np.array(pig, dtype=int)

def balanced_test_split_by_pigment_mixture(
    df: pd.DataFrame,
    vars_: dict,
    per_mix: int = 2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1) Selecciona un TEST balanceado: exactamente `k` filas por cada (Pigment Index, Mixture),
       con k = min(per_mix, min_count_por_grupo). Si k < 2, avisa pero sigue.
    2) Del resto hace split Train/Val con proporción 70/15 respecto al total (test ya fijado).
    Estratifica Train/Val por pigmento.
    """
    rng = np.random.default_rng(seed)

    # --- índices por (Pigment, Mixture) ---
    grp = df.groupby(["Pigment Index", "Mixture"], sort=False).indices
    counts = {k: len(v) for k, v in grp.items()}

    if not counts:
        raise ValueError("No hay grupos (Pigment Index, Mixture) para construir el test.")

    k_common = min(per_mix, min(counts.values()))
    if k_common <= 0:
        raise ValueError(
            "Algún (Pigment, Mixture) no tiene filas. Revisa cuotas/filtros en processor."
        )
    if k_common < per_mix:
        print(f"[WARN] Bajando test_per_mixture de {per_mix} a {k_common} "
              f"por falta de filas en algún grupo.")

    # --- SAMPLE TEST FIJO Y BALANCEADO ---
    test_idx_list = []
    for (p, m), idxs in grp.items():
        idxs = np.array(list(idxs))
        if len(idxs) <= k_common:
            test_idx_list.append(idxs)
        else:
            test_idx_list.append(rng.choice(idxs, size=k_common, replace=False))
    idx_test = np.concatenate(test_idx_list)
    idx_test.sort()

    # --- RESTO -> TRAIN/VAL (70/15 del total) ---
    all_idx = np.arange(len(df))
    mask = np.ones(len(df), dtype=bool)
    mask[idx_test] = False
    idx_rest = all_idx[mask]

    # Queremos que al final aprox. 70/15/15 del TOTAL.
    # Si TEST ocupa T, en el resto (R) usamos val_ratio = 0.15 / (0.70+0.15)
    val_ratio = 0.15 / 0.85
    y_rest_pig = pigment_ids(df.iloc[idx_rest], vars_)
    idx_train, idx_val = train_test_split(
        idx_rest, test_size=val_ratio, random_state=seed, stratify=y_rest_pig
    )

    return np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test)

# ─────────────────────────────────────────────────────────────────────────────
# Decoders SIN threshold (solo argmax para etiquetar)
# ─────────────────────────────────────────────────────────────────────────────
def decode_pigment_and_group(y_like: np.ndarray, n_p: int):
    pig = np.argmax(y_like[:, :n_p], axis=1)
    mix_idx = np.argmax(y_like[:, n_p:n_p+4], axis=1)  # 0=Pure, 1..3=Mixtures
    group = np.where(mix_idx == 0, "Pure", "Mixture")
    return np.array([f"P{p+1:02d}_{g}" for p, g in zip(pig, group)])

def decode_pigment_and_mix4(y_like: np.ndarray, n_p: int):
    pig = np.argmax(y_like[:, :n_p], axis=1)
    mix_idx = np.argmax(y_like[:, n_p:n_p+4], axis=1)  # 0=Pure,1=M1,2=M2,3=M3
    names = np.array(["Pure", "M1", "M2", "M3"])
    return np.array([f"P{p+1:02d}_{names[m]}" for p, m in zip(pig, mix_idx)])

def decode_mix_group(y_like: np.ndarray, n_p: int):
    mix_idx = np.argmax(y_like[:, n_p:n_p+4], axis=1)
    return np.where(mix_idx == 0, "Pure", "Mixture")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train HSI models + export CMs (4N, 2N, 2). No thresholds; split estratificado 70/15/15."
    )
    p.add_argument("--outputs-dir", type=str, default=None)
    p.add_argument("--models", type=str, required=True,
                   help="Lista separada por comas (ej: DNN,cnn_baseline o rutas de módulo completas)")
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = args.outputs_dir or variables.get("outputs_dir") or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Datos desde el processor (si éste filtra/ordena, ya viene aplicado)
    pr = HSIDataProcessor(variables)
    pr.load_h5_files()
    df = pr.dataframe()

    # Guardar dataframe usado
    datasets_dir = os.path.join(out_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    df_used_path = os.path.join(datasets_dir, "dataframe_used.csv")
    df.to_csv(df_used_path, index=False)
    print(f"[SAVE] DataFrame usado -> {df_used_path}")

    # 2) Split
    if variables.get("balance_test_by_mixture", True):
        idx_train, idx_val, idx_test = balanced_test_split_by_pigment_mixture(
            df,
            variables,
            per_mix=int(variables.get("test_per_mixture", 2)),
            seed=variables.get("seed", 42),
        )
    else:
        idx_train, idx_val, idx_test = stratified_split_70_15_15(
            df, variables, seed=variables.get("seed", 42)
        )


    # 3) X/y
    X, y, input_len = build_Xy(df)
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    print(f"[DATA] input_len={input_len} | X_train={X_train.shape} | "
          f"X_val={X_val.shape} | X_test={X_test.shape}")

    # 4) Entrenar y evaluar
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in model_names:
        print(f"[TRAIN] {name}")
        tune = import_model_trainer(name)

        res = tune(
            X_train, y_train, X_val, y_val,
            input_len=input_len, num_classes=y.shape[1],
            trials=args.trials if args.trials is not None else variables.get("trials"),
            epochs=args.epochs if args.epochs is not None else variables.get("epochs"),
            n_jobs=variables.get("optuna_n_jobs", 1),
            seed=variables.get("seed", 42),
        )
        model = res[0] if isinstance(res, tuple) else res

        # Predicciones de test (probabilidades); no tocamos ni umbralizamos
        y_pred_prob = model.predict(X_test, verbose=0)
        if not isinstance(y_pred_prob, np.ndarray):
            y_pred_prob = np.concatenate(y_pred_prob, axis=1)

        # === Decodificación SOLO con argmax (sin threshold) ===
        n_p = int(variables["num_files"])
        y_true_2N = decode_pigment_and_group(y_test, n_p)
        y_pred_2N = decode_pigment_and_group(y_pred_prob, n_p)

        y_true_4N = decode_pigment_and_mix4(y_test, n_p)
        y_pred_4N = decode_pigment_and_mix4(y_pred_prob, n_p)

        y_true_mix = decode_mix_group(y_test, n_p)
        y_pred_mix = decode_mix_group(y_pred_prob, n_p)

        # === Matrices de confusión (normalize='true' solo para visual) ===
        sub_out = os.path.join(out_dir, name, "conf_mats")
        os.makedirs(sub_out, exist_ok=True)

        # 4N
        classes_4N = [f"P{i+1:02d}_{s}" for i in range(n_p) for s in ["Pure","M1","M2","M3"]]
        idx_4N = {c: i for i, c in enumerate(classes_4N)}
        ti_4N = np.array([idx_4N[x] for x in y_true_4N])
        pi_4N = np.array([idx_4N[x] for x in y_pred_4N])
        cm_4N = sk_confusion_matrix(ti_4N, pi_4N,
                                    labels=list(range(len(classes_4N))), normalize='true')
        plot_confusion_matrix(
            cm_4N, classes_4N,
            "Global 4-casos (Pure/M1/M2/M3 por pigmento)",
            os.path.join(sub_out, f"{name}_cm_GLOBAL_4CASES.png")
        )

        # 2N
        classes_2N = [f"P{i+1:02d}_Pure" for i in range(n_p)] + \
                     [f"P{i+1:02d}_Mixture" for i in range(n_p)]
        idx_2N = {c: i for i, c in enumerate(classes_2N)}
        ti_2N = np.array([idx_2N[x] for x in y_true_2N])
        pi_2N = np.array([idx_2N[x] for x in y_pred_2N])
        cm_2N = sk_confusion_matrix(ti_2N, pi_2N,
                                    labels=list(range(len(classes_2N))), normalize='true')
        plot_confusion_matrix(
            cm_2N, classes_2N,
            "Global (Pigment + Pure/Mixture)",
            os.path.join(sub_out, f"{name}_cm_GLOBAL.png")
        )

        # 2 clases (Pure vs Mixture)
        classes_mix = ["Pure", "Mixture"]
        idx_mix = {"Pure": 0, "Mixture": 1}
        ti_m = np.array([idx_mix[x] for x in y_true_mix])
        pi_m = np.array([idx_mix[x] for x in y_pred_mix])
        cm_m = sk_confusion_matrix(ti_m, pi_m, labels=[0, 1], normalize='true')
        plot_confusion_matrix(
            cm_m, classes_mix,
            "Mixture Only (Pure vs Mixture)",
            os.path.join(sub_out, f"{name}_cm_MIXTURE.png")
        )


if __name__ == "__main__":
    main()

