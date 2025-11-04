# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
import os
import csv
import numpy as np        
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix 
from scipy.optimize import nnls
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import argparse
from hsi_lab.data.processor_pure import HSIDataProcessor
from hsi_lab.data.config import variables
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, hamming_loss, log_loss, average_precision_score
)
from report1 import (
    parse_args,
    build_Xy,
    stratified_balanced_split_by_file_pigment_mixture,
    stratified_split_70_15_15,
    import_model_trainer,
    generate_combined_report,
    generate_combined_report_pigments_only,
    reflectance_to_ks,
    ks_to_reflectance,
    estimate_k_s_per_pigment,
    reflectance_to_ks,
    km_unmix_nnls,
    apply_km_unmixing_nnls,
    generate_synthetic_mixtures,
    get_wavelengths_and_labels,
    apply_per_file_region_quotas,
    equalize_across_files_by_pigment,
    save_region_subregion_usage,
    export_splits_csv,
    plot_comparative_spectra,
    write_conclusions,
    soft_confusion_matrix,
    plot_confusion_matrix,
    confusion_from_labels,
    soft_confusion_matrix,
    compute_metrics,
    compute_detailed_metrics,
    compute_detailed_metrics_pigments_only,
    export_matrix_model_diff,
    export_matrix_metrics_full,
    compute_reflectance_ks_covariance
)

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # === 1️⃣ Configurar salida principal del modelo ===
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in model_names:
        print(f"\n🚀 Starting pipeline for model: {name}")

        # Carpeta raíz del modelo dentro de outputs/
        main_dir = os.path.join("outputs", name)
        os.makedirs(main_dir, exist_ok=True)
        print(f"[SETUP] Main directory created → {main_dir}")

        out_dir = main_dir  # ruta base de salida

        # ====================================================================== #
        # 1️⃣ CARGA Y PREPROCESADO DE DATOS
        # ====================================================================== #
        pr = HSIDataProcessor(variables)
        pr.load_h5_files()
        df_raw = pr.dataframe()

        # Aplicar cuotas regionales si procede
        if variables.get("apply_region_quotas", True):
            df_after_quotas = apply_per_file_region_quotas(df_raw, variables)
        else:
            df_after_quotas = df_raw

        # Igualar tamaños entre archivos si procede
        if variables.get("equalize_across_files", True):
            df_used = equalize_across_files_by_pigment(df_after_quotas)
        else:
            df_used = df_after_quotas

        # División train/val/test balanceada
        if variables.get("balance_test_by_mixture", True):
            idx_train, idx_val, idx_test = stratified_balanced_split_by_file_pigment_mixture(
                df_used, variables,
                per_mix=int(variables.get("test_per_mixture", 2)),
                seed=variables.get("seed", 42)
            )
        else:
            idx_train, idx_val, idx_test = stratified_split_70_15_15(
                df_used, variables, seed=variables.get("seed", 42)
            )

        # Construcción de X, y
        X, y, input_len = build_Xy(df_used)

        idx_train = np.array(idx_train, dtype=int)
        idx_val   = np.array(idx_val, dtype=int)
        idx_test  = np.array(idx_test, dtype=int)

        X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

        num_classes = y_train.shape[1]

        print(f"[DATA] input_len={input_len} | X_train={X_train.shape} | y_train={y_train.shape}")

        # ====================================================================== #
        # 2️⃣ ENTRENAMIENTO DIRECTO CON DATOS DEL DATAFRAME
        # ====================================================================== #
        print(f"\n[TRAINING] Training model '{name}' directly on real dataset...")

        tune = import_model_trainer(name)

        res = tune(
            X_train, y_train,
            X_val, y_val,
            input_len=input_len,
            num_classes=num_classes,
            trials=args.trials or variables.get("trials"),
            epochs=args.epochs or variables.get("epochs"),
            batch_size=args.batch_size or variables.get("batch_size"),
            n_jobs=variables.get("optuna_n_jobs", 1),
            seed=variables.get("seed", 42),
        )

        # Algunos modelos devuelven (model, history) o (model, study)
        model = res[0] if isinstance(res, tuple) else res
        print(f"[OK] Model '{name}' trained successfully on real dataset.")

        # ====================================================================== #
        # 3️⃣ PREDICCIONES Y MÉTRICAS
        # ====================================================================== #
        print("\n[TEST] Evaluating model on test set...")

        y_pred_prob = model.predict(X_test)
        y_pred_bin = (y_pred_prob > 0.5).astype(int)

        # Guardar métricas detalladas
        metrics, desc = compute_metrics(y_test, y_pred_bin, y_pred_prob)

        metrics_path = os.path.join(out_dir, f"{name}_metrics.csv")
        pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": [float(v) if not isinstance(v, (dict, tuple)) else np.nan for v in metrics.values()],
            "Description": [desc[k] for k in metrics.keys()]
        }).to_csv(metrics_path, index=False)
        print(f"[SAVE] Metrics summary -> {metrics_path}")

        print("\n--- Global Metrics ---")
        for k, v in metrics.items():
            print(f"{k:20s}: {v:.4f}")

        # ====================================================================== #
        # 4️⃣ OPCIONAL: MATRICES DE CONFUSIÓN Y GUARDADO EXTRA
        # ====================================================================== #
        report_dir = os.path.join(out_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        print(f"[SETUP] Report directory created → {report_dir}")

        # === Guardar DataFrame usado ===
        datasets_dir = os.path.join(out_dir, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)

        df_used_path = os.path.join(datasets_dir, f"{name}_dataframe_used.csv")
        df_used.to_csv(df_used_path, index=False)
        print(f"[SAVE] DataFrame used -> {df_used_path}")


        print(f"\n✅ Pipeline finished for model '{name}'")




# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()