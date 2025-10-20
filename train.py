import os
import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hsi_lab.data.config import variables
from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.eval.report import cm_pigment_mix2N, cm_pigment_mix4N, cm_mix_global2


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades mínimas
# ─────────────────────────────────────────────────────────────────────────────
def parse_region_quota(s: str) -> dict[int, int]:
    """Convierte '1:300,2:100' en {1:300, 2:100}."""
    if not s:
        return {}
    out: dict[int, int] = {}
    for tok in s.split(","):
        m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", tok.strip())
        if not m:
            raise ValueError(f"Formato inválido en --region-quota: {tok}")
        out[int(m.group(1))] = int(m.group(2))
    return out


def sample_per_region(
    df: pd.DataFrame,
    per_region: int | None = None,
    quota_map: dict[int, int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Muestrea por 'Region' SIN reequilibrar nada.
    - Si quota_map: cantidades por región, ej: {1:300,2:100}
    - Si per_region: misma cantidad para todas las regiones presentes
    Solo recorta filas y baraja; no toca columnas ni etiquetas.
    """
    if "Region" not in df.columns:
        return df

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []

    if quota_map:
        for r in sorted(quota_map.keys()):
            sub = df[df["Region"] == r]
            n = min(len(sub), int(quota_map[r]))
            if n > 0:
                idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
                parts.append(df.loc[idx])
    elif per_region is not None:
        for _, sub in df.groupby("Region"):
            n = min(len(sub), int(per_region))
            if n > 0:
                idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
                parts.append(df.loc[idx])
    else:
        return df

    out = pd.concat(parts, axis=0) if parts else df.iloc[0:0]
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Construcción X/y (no toca representación)
# ─────────────────────────────────────────────────────────────────────────────
def build_Xy(df: pd.DataFrame):
    """
    X: columnas espectrales 'vis_*' y 'swir_*' ordenadas (vis primero).
    y: vector Multi tal cual (pigmentos + 4 mixtures).
    """
    spec_cols = [c for c in df.columns if c.startswith(("vis_", "swir_"))]
    if not spec_cols:
        raise ValueError("No spectral columns starting with 'vis_' or 'swir_'.")
    spec_cols = sorted(spec_cols, key=lambda c: (0 if c.startswith("vis_") else 1, c))
    X = df[spec_cols].astype(np.float32).fillna(0.0).values[..., np.newaxis]
    y = np.array([np.array(v) for v in df["Multi"]], dtype=np.float32)
    return X, y, X.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# Split por (File, Mixture) + estratificación por pigmento
# ─────────────────────────────────────────────────────────────────────────────
def pigment_ids(df: pd.DataFrame, vars_: dict) -> np.ndarray:
    n_p = int(vars_["num_files"])
    pig = []
    for v in df["Multi"]:
        a = np.asarray(v, dtype=np.float32)
        pig.append(int(np.argmax(a[:n_p])))
    return np.array(pig, dtype=int)


def grouped_split_indices(df: pd.DataFrame, vars_: dict, test_frac=0.30, seed=42):
    """
    Garantiza al menos una muestra por (File, Mixture) en TEST y ~test_frac total.
    El resto se divide en train/val con estratificación por pigmento.
    """
    if not {"File", "Mixture"}.issubset(df.columns):
        raise ValueError("Necesito columnas 'File' y 'Mixture' para split por grupos.")

    keys = df[["File", "Mixture"]].astype(str).agg(" | ".join, axis=1).values
    by_group = defaultdict(list)
    for i, k in enumerate(keys):
        by_group[k].append(i)

    rng = np.random.default_rng(seed)
    idx_test, pool = [], []
    for _, idxs in by_group.items():
        idx_test.append(idxs[0])
        pool.extend([j for j in idxs if j != idxs[0]])

    desired = int(test_frac * len(df))
    need_extra = max(0, min(desired - len(idx_test), len(pool)))
    if need_extra > 0:
        idx_test = np.unique(
            np.concatenate([idx_test, rng.choice(pool, size=need_extra, replace=False)])
        ).tolist()

    mask = np.ones(len(df), dtype=bool)
    mask[idx_test] = False
    idx_rest = np.where(mask)[0]

    y_pig = pigment_ids(df, vars_)
    y_rest = y_pig[idx_rest]

    idx_train_rel, idx_val_rel = train_test_split(
        np.arange(len(idx_rest)), test_size=0.2, random_state=seed, stratify=y_rest
    )
    idx_train = idx_rest[idx_train_rel]
    idx_val = idx_rest[idx_val_rel]
    return np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────
def import_model_trainer(name: str):
    """
    Devuelve la función tune_and_train para el modelo 'name'.
    Debe tener firma:
      tune_and_train(X_train, y_train, X_val, y_val, *, input_len, num_classes, ...)
    y devolver (model, study, best) o solo model.
    """
    if name == "cnn_baseline":
        return tune_and_train_default  # si tienes un baseline in-tree
    mod = __import__(f"hsi_lab.models.{name}", fromlist=["tune_and_train"])
    return getattr(mod, "tune_and_train")


# ─────────────────────────────────────────────────────────────────────────────
# CLI & main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train HSI models y guardar CMs (4N, 2N y 2) sin cambiar representación."
    )
    p.add_argument("--outputs-dir", type=str, default=None)
    p.add_argument("--models", type=str, required=True,
                   help="Lista separada por comas (ej: cnn_baseline,otro)")
    # Selección rápida por región
    p.add_argument("--per-region", type=int, default=None,
                   help="Tomar N filas por cada región (muestreo uniforme).")
    p.add_argument("--region-quota", type=str, default=None,
                   help='Cupos por región, ej: "1:300,2:100,4:50"')
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.outputs_dir or variables.get("outputs_dir") or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Cargar DF
    pr = HSIDataProcessor(variables)
    pr.load_h5_files()
    df = pr.dataframe()

    # 2) (Opcional) muestreo por región SIN reequilibrar
    quota_map = parse_region_quota(args.region_quota) if args.region_quota else None
    if args.per_region is not None or quota_map:
        df = sample_per_region(
            df,
            per_region=args.per_region,
            quota_map=quota_map,
            seed=variables.get("seed", 42),
        )
    df.to_csv(os.path.join(out_dir, "dataframe_used.csv"), index=False)

    # 3) Split
    idx_train, idx_val, idx_test = grouped_split_indices(
        df, variables, test_frac=0.30, seed=variables.get("seed", 42)
    )

    # 4) X/y
    X, y, input_len = build_Xy(df)
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    print(f"[DATA] input_len={input_len} | X_train={X_train.shape} | "
          f"X_val={X_val.shape} | X_test={X_test.shape}")

    # 5) Entrenar y evaluar
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

        # Predicciones de test (probabilidades)
        y_pred_prob = model.predict(X_test, verbose=0)
        if not isinstance(y_pred_prob, np.ndarray):
            y_pred_prob = np.concatenate(y_pred_prob, axis=1)

        # Directorio de salidas por modelo
        sub_out = os.path.join(out_dir, name)
        os.makedirs(os.path.join(sub_out, "conf_mats"), exist_ok=True)

        # 4N, 2N y 2 vistas (formato bonito definido en report.py)
        cm_pigment_mix4N(
            y_true=y_test,
            y_pred=y_pred_prob,
            out_path=os.path.join(sub_out, "conf_mats", f"{name}_cm_GLOBAL_4CASES.png"),
        )
        cm_pigment_mix2N(
            y_true=y_test,
            y_pred=y_pred_prob,
            out_path=os.path.join(sub_out, "conf_mats", f"{name}_cm_GLOBAL.png"),
        )
        cm_mix_global2(
            y_true=y_test,
            y_pred=y_pred_prob,
            out_path=os.path.join(sub_out, "conf_mats", f"{name}_cm_MIXTURE.png"),
        )


if __name__ == "__main__":
    main()
