import os
import pandas as pd

# ───────────────────────────────────────────────
# CONFIGURACIÓN DE RUTAS
# ───────────────────────────────────────────────
BASE_DIR = "/home/pgimenez/projects/HSI/hsi_lab/data"
PURE_PATH = os.path.join(BASE_DIR, "processor_pure_pigments.csv")
SYNTH_PATH = os.path.join(BASE_DIR, "processor_synthetic_mixtures.csv")
OUT_PATH = os.path.join(BASE_DIR, "processor_concat_pure_synthetic.csv")

# ───────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ───────────────────────────────────────────────
def concat_pure_and_synthetic(pure_path=PURE_PATH, synth_path=SYNTH_PATH, out_path=OUT_PATH):
    print(f"[INFO] Loading:")
    print(f"       PURE  -> {pure_path}")
    print(f"       SYNTH -> {synth_path}")

    if not os.path.exists(pure_path):
        raise FileNotFoundError(f"Missing file: {pure_path}")
    if not os.path.exists(synth_path):
        raise FileNotFoundError(f"Missing file: {synth_path}")

    df_pure = pd.read_csv(pure_path)
    df_synth = pd.read_csv(synth_path)

    print(f"[INFO] Pure pigments shape: {df_pure.shape}")
    print(f"[INFO] Synthetic mixtures shape: {df_synth.shape}")

    # Armonizar columnas
    common_cols = list(set(df_pure.columns) & set(df_synth.columns))
    missing_in_synth = [c for c in df_pure.columns if c not in df_synth.columns]
    missing_in_pure = [c for c in df_synth.columns if c not in df_pure.columns]

    if missing_in_synth:
        print(f"[WARN] Columns missing in synthetic: {missing_in_synth}")
    if missing_in_pure:
        print(f"[WARN] Columns missing in pure: {missing_in_pure}")

    # Añadir columnas faltantes
    for col in missing_in_synth:
        df_synth[col] = pd.NA
    for col in missing_in_pure:
        df_pure[col] = pd.NA

    # Ordenar columnas igual que en df_pure
    df_synth = df_synth[df_pure.columns]

    # Concatenar
    df_concat = pd.concat([df_pure, df_synth], ignore_index=True)
    print(f"[OK] Concatenated shape: {df_concat.shape}")
    print(f"[INFO] Total rows: {len(df_concat)}")

    # Guardar
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_concat.to_csv(out_path, index=False)
    print(f"[SAVE] Combined CSV -> {out_path}")
    return df_concat


# ───────────────────────────────────────────────
# EJECUCIÓN DESDE TERMINAL
# ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[INIT] Concatenating pure pigments + synthetic mixtures...")
    df_final = concat_pure_and_synthetic()
    print("[DONE] Combined CSV successfully generated.")

    """
    To execute from terminal:
      cd ~/projects/HSI
      python -m hsi_lab.data.processor_concat_pure_synthetic
    """
