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

from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.data.config import variables
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
# MAIN PIPELINE
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

        # Asignar esta ruta como base de todos los outputs
        out_dir = main_dir

    # ====================================================================== #
    # 1️⃣ CARGA Y PREPROCESADO DE DATOS
    # ====================================================================== #
    pr = HSIDataProcessor(variables)
    pr.load_h5_files()
    df_raw = pr.dataframe(mode="raw")

    if variables.get("apply_region_quotas", True):
        df_after_quotas = apply_per_file_region_quotas(df_raw, variables)
    else:
        df_after_quotas = df_raw

    if variables.get("equalize_across_files", True):
        df_used = equalize_across_files_by_pigment(df_after_quotas)
    else:
        df_used = df_after_quotas

    if variables.get("balance_test_by_mixture", True):
        idx_train, idx_val, idx_test = stratified_balanced_split_by_file_pigment_mixture(
            df_used, variables, per_mix=int(variables.get("test_per_mixture", 2)),
            seed=variables.get("seed", 42)
        )
    else:
        idx_train, idx_val, idx_test = stratified_split_70_15_15(
            df_used, variables, seed=variables.get("seed", 42)
        )

    X, y, input_len = build_Xy(df_used)
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    num_classes = y_train.shape[1]

    print(f"[DATA] input_len={input_len} | X_train={X_train.shape} | y_train={y_train.shape}")


    # ====================================================================== #
    # 2️⃣ ESPECTROS BASE
    # ====================================================================== #
    spectra_dir = os.path.join(out_dir, "spectra_dataset")
    plot_comparative_spectra(df_used, base_out=spectra_dir, name="dataset_per_pigment", per_file=True)
    plot_comparative_spectra(df_used, base_out=spectra_dir, name="dataset_full", per_file=False)


# ====================================================================== #
    # 3️⃣ MODELO EN REFLECTANCIA
    # ====================================================================== #
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in model_names:
        print(f"\n[TRAIN - REFLECTANCE] {name}")
        tune = import_model_trainer(name)
        res = tune(
            X_train, y_train, X_val, y_val,
            input_len=input_len, num_classes=num_classes,
            trials=args.trials or variables.get("trials"),
            epochs=args.epochs or variables.get("epochs"),
            batch_size=args.batch_size or variables.get("batch_size"),
            n_jobs=variables.get("optuna_n_jobs", 1),
            seed=variables.get("seed", 42),
        )
        model = res[0] if isinstance(res, tuple) else res

        y_pred_prob = model.predict(X_test, verbose=0)
        base_ref = os.path.join(out_dir, "Reflectance", name)
        os.makedirs(base_ref, exist_ok=True)

        df_used.to_csv(os.path.join(base_ref, f"{name}_dataframe_used.csv"), index=False)
        cov_ref_dir = os.path.join(base_ref, "Covariance")
        os.makedirs(cov_ref_dir, exist_ok=True)

        compute_reflectance_ks_covariance(
            df=df_used,
            base_out=cov_ref_dir,
            name=f"{name}_Reflectance",
            n_pigments=int(variables["num_files"])
        )

        report_ref_dir = os.path.join(base_ref, "Evaluation")
        generate_combined_report(
            y_true=y_test, y_pred_prob=y_pred_prob,
            n_pigments=int(variables["num_files"]),
            output_dir=report_ref_dir, name=f"{name}_Reflectance"
        )

        # ====================================================================== #
        # 4️⃣ MODELO EN K/S
        # ====================================================================== #
        print(f"\n[TRAIN - K/S SPACE] {name}")
        X_train_ks = reflectance_to_ks(X_train)
        X_val_ks = reflectance_to_ks(X_val)
        X_test_ks = reflectance_to_ks(X_test)

        res_ks = tune(
            X_train_ks, y_train, X_val_ks, y_val,
            input_len=input_len, num_classes=num_classes,
            trials=args.trials or variables.get("trials"),
            epochs=args.epochs or variables.get("epochs"),
            batch_size=args.batch_size or variables.get("batch_size"),
        )
        model_ks = res_ks[0] if isinstance(res_ks, tuple) else res_ks

        y_pred_prob_ks = model_ks.predict(X_test_ks, verbose=0)
        base_ks = os.path.join(out_dir, "KS", name)
        os.makedirs(base_ks, exist_ok=True)

        df_used.to_csv(os.path.join(base_ks, f"{name}_dataframe_used.csv"), index=False)

        cov_ks_dir = os.path.join(base_ks, "Covariance")
        os.makedirs(cov_ks_dir, exist_ok=True)
        compute_reflectance_ks_covariance(df=df_used, base_out=cov_ks_dir, name=f"{name}_KS")

        report_ks_dir = os.path.join(base_ks, "Evaluation")
        os.makedirs(report_ks_dir, exist_ok=True)
        generate_combined_report(y_true=y_test, y_pred_prob=y_pred_prob_ks,
                                 n_pigments=int(variables["num_files"]),
                                 output_dir=report_ks_dir, name=f"{name}_KS")

        # ====================================================================== #
        # 5️⃣ TRANSFERENCIA REFLECTANCIA → MEZCLAS K/S (PIGMENT ANALYSIS)
        # ====================================================================== #
        print(f"\n[TRANSFER - SYNTHETIC MIXTURE TEST (KM)] {name}")

        km_dir = os.path.join(out_dir, "KM_Transfer", name)
        os.makedirs(km_dir, exist_ok=True)

        vis_cols = [c for c in df_used.columns if c.lower().startswith("vis_")]
        swir_cols = [c for c in df_used.columns if c.lower().startswith("swir_")]
        all_cols = vis_cols + swir_cols

        R_pigments = []
        for i in range(int(variables["num_files"])):
            dfp = df_used[df_used["Pigment Index"] == i]
            if len(dfp) > 0:
                R_pigments.append(dfp[all_cols].mean(axis=0).to_numpy())
        R_pigments = np.array(R_pigments, dtype=float)
        n_pigments = R_pigments.shape[0]

        mixtures, alphas_true, components, y_true_synth = generate_synthetic_mixtures(
            R_pigments, n_samples=2000, n_mix=(2, 3)
        )

        mixtures_ks = reflectance_to_ks(mixtures)
        np.save(os.path.join(km_dir, "mixtures.npy"), mixtures)
        np.save(os.path.join(km_dir, "alphas_true.npy"), np.array(alphas_true, dtype=object))

        print("[KM] Predicting pigment activations on synthetic mixtures...")
        y_pred_km = model.predict(mixtures_ks[..., np.newaxis], verbose=0)
        np.save(os.path.join(km_dir, "alphas_pred.npy"), y_pred_km)

        report_km_dir = os.path.join(km_dir, "Pigment_Analysis")
        os.makedirs(report_km_dir, exist_ok=True)

        y_true_aligned = np.array(y_true_synth, dtype=np.float32)
        y_pred_pig = y_pred_km[:, :y_true_aligned.shape[1]]

        generate_combined_report_pigments_only(
            y_true=y_true_aligned,
            y_pred_prob=y_pred_pig,
            n_pigments=n_pigments,
            output_dir=report_km_dir,
            name=f"{name}_KMTransfer_Pigments"
        )

        # ====================================================================== #
        # 5️⃣ TER — RANDOM SAMPLING CLASSIFICATION (KM)
        # ====================================================================== #
        print(f"\n[TRANSFER - RANDOM SAMPLING TEST (KM, Pigments vs Pigments+Mix)] {name}")

        km_random_dir = os.path.join(out_dir, "KM_Transfer_Random", name)
        os.makedirs(km_random_dir, exist_ok=True)

        n_random = 10000
        n_mix_classes = 4
        total_labels = n_pigments + n_mix_classes
        KS_pigments = reflectance_to_ks(R_pigments)

        mixtures = []
        y_true_pig = np.zeros((n_random, n_pigments))
        y_true_full = np.zeros((n_random, total_labels))
        alphas_true = []

        for i in range(n_random):
            idx_pigs = np.random.choice(n_pigments, 2, replace=False)
            alpha = np.random.rand(2)
            alpha /= np.sum(alpha)
            mix_bits = np.zeros(n_mix_classes)
            mix_indices = np.random.choice(n_mix_classes, 2, replace=True)
            mix_bits[mix_indices] = 1
            KS_mix = np.dot(alpha, KS_pigments[idx_pigs])
            R_mix = ks_to_reflectance(KS_mix)
            mixtures.append(R_mix)
            alphas_true.append(alpha)
            y_true_pig[i, idx_pigs] = 1
            y_true_full[i, idx_pigs] = 1
            y_true_full[i, -n_mix_classes:] = np.maximum(y_true_full[i, -n_mix_classes:], mix_bits)

        mixtures = np.array(mixtures, dtype=float)
        mixtures_ks = reflectance_to_ks(mixtures)
        y_pred_prob = model.predict(mixtures_ks[..., np.newaxis], verbose=0)

        # --- Clasificación 1: Solo pigmentos ---
        report_pig_dir = os.path.join(km_random_dir, "Classification_PigmentsOnly")
        os.makedirs(report_pig_dir, exist_ok=True)
        generate_combined_report_pigments_only(
            y_true=y_true_pig, y_pred_prob=y_pred_prob,
            n_pigments=n_pigments, output_dir=report_pig_dir,
            name=f"{name}_KMTransfer_PigmentsOnly"
        )

        # --- Clasificación 2: Pigments + mixtures ---
        report_full_dir = os.path.join(km_random_dir, "Classification_PigmentsMix")
        os.makedirs(report_full_dir, exist_ok=True)
        generate_combined_report(
            y_true=y_true_full, y_pred_prob=y_pred_prob,
            n_pigments=n_pigments, output_dir=report_full_dir,
            name=f"{name}_KMTransfer_PigmentsMix"
        )

        # ====================================================================== #
        # 6️⃣ REGRESIÓN (PREDICCIÓN DE PROPORCIONES)
        # ====================================================================== #
        print("[KM] Evaluating pigment proportion regression...")

        regmix_dir = os.path.join(km_random_dir, "Regression")
        os.makedirs(regmix_dir, exist_ok=True)

        alphas_true_mat = np.zeros((n_random, n_pigments))
        for i in range(n_random):
            idx_pigs = np.where(y_true_pig[i, :] == 1)[0]
            alphas_true_mat[i, idx_pigs] = alphas_true[i]

        y_pred_alphas = y_pred_prob[:, :n_pigments]
        rmse = np.sqrt(np.mean((alphas_true_mat - y_pred_alphas) ** 2, axis=0))

        df_regmix = pd.DataFrame({
            "Pigment": [f"P{i+1:02d}" for i in range(n_pigments)],
            "RMSE": rmse,
            "Mean_True": np.mean(alphas_true_mat, axis=0),
            "Mean_Pred": np.mean(y_pred_alphas, axis=0)
        })
        csv_regmix = os.path.join(regmix_dir, "KMTransfer_RegressionSummary.csv")
        df_regmix.to_csv(csv_regmix, index=False)
        print(f"[SAVE] Regression summary -> {csv_regmix}")

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.bar(df_regmix["Pigment"], df_regmix["RMSE"], color="tab:orange")
        ax.set_title(f"{name} – RMSE per pigment (KMTransfer Regression)")
        ax.set_ylabel("RMSE")
        plt.xticks(rotation=60)
        plt.tight_layout()
        plt.savefig(os.path.join(regmix_dir, f"{name}_KMTransfer_RMSE.png"), dpi=300)
        plt.close()
        print(f"[SAVE] RMSE plot -> {regmix_dir}")


        # ====================================================================== #
        # 7️⃣ VISUALIZACIÓN, MATRICES Y EXPORTACIÓN DE ARRAYS (TRUE vs PRED)
        # ====================================================================== #
        print("[KM] Generating covariance / coactivation / confusion matrices for random mixtures...")
        viz_dir = os.path.join(km_random_dir, "Visualization")
        os.makedirs(viz_dir, exist_ok=True)

        # === 1️⃣ Guardar arrays numéricos ===
        np.save(os.path.join(viz_dir, "mixtures_reflectance.npy"), mixtures)
        np.save(os.path.join(viz_dir, "alphas_true_matrix.npy"), alphas_true_mat)
        np.save(os.path.join(viz_dir, "alphas_pred_matrix.npy"), y_pred_alphas)

        # También como CSV legibles
        pd.DataFrame(mixtures).to_csv(os.path.join(viz_dir, "mixtures_reflectance.csv"), index=False)
        pd.DataFrame(alphas_true_mat, columns=[f"P{i+1:02d}" for i in range(n_pigments)]).to_csv(
            os.path.join(viz_dir, "alphas_true_matrix.csv"), index=False)
        pd.DataFrame(y_pred_alphas, columns=[f"P{i+1:02d}" for i in range(n_pigments)]).to_csv(
            os.path.join(viz_dir, "alphas_pred_matrix.csv"), index=False)

        print(f"[SAVE] Arrays exported -> {viz_dir}")

        # ====================================================================== #
        # 2️⃣ MATRICES DE EVALUACIÓN (solo pigmentos)
        # ====================================================================== #
        print("[KM] Generating confusion / coactivation / covariance matrices (random mixtures)...")

        generate_combined_report_pigments_only(
            y_true=alphas_true_mat,
            y_pred_prob=y_pred_alphas,
            n_pigments=n_pigments,
            output_dir=viz_dir,
            name=f"{name}_KMTransfer_Random_Pigments"
        )

        # ====================================================================== #
        # 3️⃣ RECONSTRUCCIÓN ESPECTRAL (True vs Predicho)
        # ====================================================================== #
        print("[KM] Reconstructing spectra from predicted proportions...")

        K_set = reflectance_to_ks(R_pigments)
        reconstructed = []
        for alpha in y_pred_alphas:
            alpha = np.array(alpha)
            if np.sum(alpha) > 0:
                alpha /= np.sum(alpha)
            KS_recon = np.dot(alpha, K_set)
            R_recon = ks_to_reflectance(KS_recon)
            reconstructed.append(R_recon)
        reconstructed = np.array(reconstructed)

        # === Guardar reconstrucciones ===
        np.save(os.path.join(viz_dir, "reconstructed_reflectance.npy"), reconstructed)
        pd.DataFrame(reconstructed).to_csv(os.path.join(viz_dir, "reconstructed_reflectance.csv"), index=False)

        # === Calcular error espectral medio (RMSE entre espectros) ===
        spectral_rmse = np.sqrt(np.mean((mixtures - reconstructed) ** 2, axis=1))
        pd.DataFrame({"Spectral_RMSE": spectral_rmse}).to_csv(
            os.path.join(viz_dir, "Spectral_RMSE_per_sample.csv"), index=False
        )

        # === 4️⃣ Graficar 5 ejemplos ===
        n_show = min(5, len(mixtures))
        wls, _, _ = get_wavelengths_and_labels(total_bands=reconstructed.shape[1], data_type=["vis", "swir"])
        colors = plt.cm.tab10(np.linspace(0, 1, n_show))

        plt.figure(figsize=(12, 6))
        for i in range(n_show):
            a_true = alphas_true_mat[i]
            a_pred = y_pred_alphas[i]
            pigments_true = np.argsort(-a_true)[:3]
            pigments_pred = np.argsort(-a_pred)[:3]

            label_true = ", ".join([f"P{p+1}:{a_true[p]:.2f}" for p in pigments_true])
            label_pred = ", ".join([f"P{p+1}:{a_pred[p]:.2f}" for p in pigments_pred])

            plt.plot(wls, mixtures[i], color=colors[i], lw=2,
                     label=f"True {i+1} — {label_true}")
            plt.plot(wls, reconstructed[i], "--", color=colors[i], lw=2,
                     label=f"Pred {i+1} — {label_pred}")

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title(f"{name} – Random Mixtures: True vs Reconstructed (KM_Transfer_Random)")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        spectra_png = os.path.join(viz_dir, f"{name}_KMTransfer_Random_Reconstruction.png")
        plt.savefig(spectra_png, dpi=300)
        plt.close()
        print(f"[SAVE] True vs reconstructed spectra -> {spectra_png}")

        print("\n✅ KM_Transfer random evaluation complete.")



    print("\n✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
