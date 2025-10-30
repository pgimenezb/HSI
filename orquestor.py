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

from hsi_lab.data.processor import HSIDataProcessor
from hsi_lab.data.config import variables
from report import (
    parse_args,
    build_Xy,
    stratified_balanced_split_by_file_pigment_mixture,
    stratified_split_70_15_15,
    import_model_trainer,
    generate_combined_report,
    compute_metrics,
    reflectance_to_ks,
    apply_km_unmixing,
    fast_km_unmixing,
    generate_synthetic_mixtures,
    get_wavelengths_and_labels,
    match_spectral_resolution,
    apply_per_file_region_quotas,
    equalize_across_files_by_pigment,
    save_region_subregion_usage,
    export_splits_csv,
    summarize_model,
    per_pigment_metrics,
    top_confusions,
    plot_comparative_spectra,
    write_conclusions,
    summarize_confusion_matrices,
    soft_confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = args.outputs_dir or variables.get("outputs_dir") or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1️⃣ Cargar datos
    pr = HSIDataProcessor(variables)
    pr.load_h5_files()
    df_raw = pr.dataframe(mode="raw")

    # 2️⃣ Aplicar cuotas por región/subregión
    APPLY_QUOTAS = variables.get("apply_region_quotas", True)
    region_quotas = variables.get("region_row_quota", {}) or {}
    df_after_quotas = (
        apply_per_file_region_quotas(df_raw, variables)
        if (APPLY_QUOTAS and region_quotas)
        else df_raw
    )

    # 3️⃣ Igualado global por pigmento
    DO_EQUALIZE_GLOBAL = variables.get("equalize_across_files", True)
    df_used = (
        equalize_across_files_by_pigment(df_after_quotas)
        if DO_EQUALIZE_GLOBAL
        else df_after_quotas
    )

    # 4️⃣ Split del dataset
    if variables.get("balance_test_by_mixture", True):
        idx_train, idx_val, idx_test = stratified_balanced_split_by_file_pigment_mixture(
            df_used,
            variables,
            per_mix=int(variables.get("test_per_mixture", 2)),
            seed=variables.get("seed", 42),
        )
    else:
        idx_train, idx_val, idx_test = stratified_split_70_15_15(
            df_used, variables, seed=variables.get("seed", 42)
        )

    
    # 5️⃣ Construir X/y
    X, y, input_len = build_Xy(df_used)
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    print(f"[DATA] input_len={input_len} | X_train={X_train.shape} | X_val={X_val.shape} | X_test={X_test.shape}")

    # === Calcular tamaño real de entrada y salida === ✅
    input_len = X_train.shape[1]
    num_classes = y_train.shape[1]
    print(f"[INFO] Model input length adjusted to {input_len} bands ({num_classes} outputs)")

    # 5️⃣bis Kubelka–Munk Synthetic Mixing & Unmixing (auto-scaled VIS/SWIR wavelengths)
    DO_KM_MIX = variables.get("do_km_mixing", True)
    if DO_KM_MIX:
        print("\n[KM] Generating synthetic mixtures and performing Kubelka–Munk unmixing...")

        # === Compute mean spectrum per pigment (average over one-hot samples)
        pigment_means = []
        for i in range(y_train.shape[1]):
            mask = y_train[:, i] > 0
            if np.sum(mask) > 0:
                spec = np.mean(X_train[mask], axis=0)
                pigment_means.append(spec.squeeze())
        pigment_means = np.array(pigment_means).reshape(len(pigment_means), -1)
        print(f"[KM] {len(pigment_means)} pigments detected. Shape={pigment_means.shape}")

        # === Generate synthetic mixtures
        mixtures, alphas_true, components = generate_synthetic_mixtures(
            pigment_means,
            n_samples=int(variables.get("km_n_samples", 2000)),
            n_mix=tuple(variables.get("km_n_mix", (2, 3)))
        )

        # === Convert reflectance → K/S
        KS_pure = reflectance_to_ks(pigment_means)
        k_set = KS_pure
        s_set = np.ones_like(KS_pure)

        # === Apply unmixing
        if variables.get("km_method", "minimize") == "lstsq":
            alphas_pred = fast_km_unmixing(mixtures, k_set, s_set)
        else:
            alphas_pred = apply_km_unmixing(mixtures, k_set, s_set)

        # === Compute mean MSE
        errors = []
        for a_true, a_pred in zip(alphas_true, alphas_pred):
            if len(a_pred) > len(a_true):
                a_pred = a_pred[np.argsort(a_pred)[-len(a_true):]]
            errors.append(np.mean((np.array(a_true) - np.array(a_pred)) ** 2))
        mean_mse = np.mean(errors)
        print(f"[KM] Mean unmixing error (MSE): {mean_mse:.6f}")

        # === Output folder
        km_dir = os.path.join(out_dir, f"{args.models}", "KM_unmixing_results")
        os.makedirs(km_dir, exist_ok=True)

        # === Save data arrays
        np.save(os.path.join(km_dir, "mixtures.npy"), mixtures)
        np.save(os.path.join(km_dir, "alphas_true.npy"), np.array(alphas_true, dtype=object))
        np.save(os.path.join(km_dir, "alphas_pred.npy"), np.array(alphas_pred, dtype=object))

        # === Generate wavelength axis (auto-scaled VIS/SWIR)
        wls, tick_positions, tick_labels = get_wavelengths_and_labels(
            total_bands=mixtures.shape[1],
            data_type=variables.get("data_type", ["vis"])
        )

        # Separar VIS y SWIR (si aplica)
        vis_mask = wls < 1000
        vis = wls[vis_mask]
        swir = wls[~vis_mask]

        print(f"[DEBUG] Wavelength range: {wls[0]:.1f}–{wls[-1]:.1f} nm | {len(wls)} bands")

        # === Save spectra as CSV
        df_mix = pd.DataFrame(mixtures, columns=[f"{wl:.1f}nm" for wl in wls])
        df_mix.to_csv(os.path.join(km_dir, "synthetic_mixtures.csv"), index=False)

        df_pure = pd.DataFrame(pigment_means, columns=[f"{wl:.1f}nm" for wl in wls])
        df_pure.to_csv(os.path.join(km_dir, "pigment_means.csv"), index=False)

        # === Export α_true / α_pred as readable CSVs
        max_len_true = max(len(a) for a in alphas_true)
        max_len_pred = max(len(a) for a in alphas_pred)
        max_len = max(max_len_true, max_len_pred)

        alphas_true_padded = [np.pad(a, (0, max_len - len(a))) for a in alphas_true]
        alphas_pred_padded = [np.pad(a, (0, max_len - len(a))) for a in alphas_pred]

        alphas_true_df = pd.DataFrame(alphas_true_padded, columns=[f"alpha_true_{i+1}" for i in range(max_len)])
        alphas_pred_df = pd.DataFrame(alphas_pred_padded, columns=[f"alpha_pred_{i+1}" for i in range(max_len)])
        alphas_true_df.to_csv(os.path.join(km_dir, "alphas_true.csv"), index=False)
        alphas_pred_df.to_csv(os.path.join(km_dir, "alphas_pred.csv"), index=False)

        print(f"[SAVE] α_true proportions -> {os.path.join(km_dir, 'alphas_true.csv')}")
        print(f"[SAVE] α_pred proportions -> {os.path.join(km_dir, 'alphas_pred.csv')}")

        # === Reconstruct mixtures from predicted α (with pigment info)
        print("[KM] Reconstructing mixtures from predicted alphas...")

        def ks_to_reflectance(KS):
            KS = np.maximum(KS, 1e-6)  # evita valores muy pequeños o negativos
            R = (1 + KS - np.sqrt(KS**2 + 2*KS))
            return np.clip(R, 0, 1)

        reconstructed = []
        for alpha in alphas_pred:
            KS_recon = np.dot(alpha, k_set) / np.dot(alpha, s_set)
            R_recon = ks_to_reflectance(KS_recon)
            reconstructed.append(R_recon)
        reconstructed = np.array(reconstructed)

        df_reconstructed = pd.DataFrame(reconstructed, columns=[f"{wl:.1f}nm" for wl in wls])
        df_reconstructed.to_csv(os.path.join(km_dir, "reconstructed_mixtures.csv"), index=False)

        # === Plot example synthetic spectra ===
        plt.figure(figsize=(12, 5))
        n_show = min(5, len(mixtures))
        for i in range(n_show):
            a_true = np.array(alphas_true[i])
            pigments_used = np.argsort(-a_true)[:3]
            mix_label = ", ".join([f"P{p+1}:{a_true[p]:.2f}" for p in pigments_used])
            plt.plot(wls, mixtures[i], lw=2, label=f"Mix {i+1} — {mix_label}")

        plt.xlabel("Data type, channel and wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title("Example Synthetic Mixtures (Kubelka–Munk)")
        plt.xticks(tick_positions, tick_labels, rotation=65, ha="right", fontsize=8)
        plt.legend(loc="upper right", fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(km_dir, "synthetic_mixtures_examples.png"), dpi=300)
        plt.close()

        # === Plot comparison between true and reconstructed mixtures ===
        plt.figure(figsize=(12, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, n_show))
        for i in range(n_show):
            a_true = np.array(alphas_true[i])
            a_pred = np.array(alphas_pred[i])
            pigments_true = np.argsort(-a_true)[:3]
            pigments_pred = np.argsort(-a_pred)[:3]
            label_true = ", ".join([f"P{p+1}:{a_true[p]:.2f}" for p in pigments_true])
            label_pred = ", ".join([f"P{p+1}:{a_pred[p]:.2f}" for p in pigments_pred])
            plt.plot(wls, mixtures[i], color=colors[i], lw=2, label=f"True {i+1} — {label_true}")
            plt.plot(wls, reconstructed[i], "--", color=colors[i], lw=2, label=f"Pred {i+1} — {label_pred}")

        plt.xlabel("Data type, channel and wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title("True vs Reconstructed Mixtures (Kubelka–Munk)")
        plt.xticks(tick_positions, tick_labels, rotation=65, ha="right", fontsize=8)
        plt.legend(loc="upper right", fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(km_dir, "reconstructed_mixtures_comparison.png"), dpi=300)
        plt.close()

        print(f"[KM] Plots saved in {km_dir}")

    # 6️⃣ TRAINING  
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    for name in model_names:
        print(f"\n[TRAIN] {name}")
        tune = import_model_trainer(name)

        res = tune(
            X_train, y_train, X_val, y_val,
            input_len=input_len,
            num_classes=num_classes,
            trials=args.trials or variables.get("trials"),
            epochs=args.epochs or variables.get("epochs"),
            batch_size=args.batch_size or variables.get("batch_size"),
            n_jobs=variables.get("optuna_n_jobs", 1),
            seed=variables.get("seed", 42),
        )

        if isinstance(res, tuple):
            model = res[0]
        else:
            model = res

        print(f"[EVAL] Predicting test set with {name}...")
        y_pred_prob = model.predict(X_test, verbose=0)

        base_out = os.path.join(out_dir, name)
        os.makedirs(base_out, exist_ok=True)


        # === Evaluar en el conjunto de test ===
        print(f"[EVAL] Predicting test set with {name}...")
        y_pred_prob = model.predict(X_test, verbose=0)

        # === Crear carpeta de salida para este modelo ===
        base_out = os.path.join(out_dir, name)
        os.makedirs(base_out, exist_ok=True)


        # === Guardar DataFrame usado ===
        df_used_path = os.path.join(base_out, f"{name}_dataframe_used.csv")
        df_used.to_csv(df_used_path, index=False)
        print(f"[SAVE] DataFrame -> {df_used_path}")

        # === Guardar uso de región/subregión ===
        region_usage_path = os.path.join(base_out, f"{name}_region_subregion_usage.csv")
        save_region_subregion_usage(df_raw, df_used, region_usage_path)

        # === Guardar splits ===
        y_pred_prob_full = np.full_like(y, np.nan, dtype=np.float32)
        y_pred_prob_full[idx_test] = y_pred_prob
        split_paths = export_splits_csv(
            df_used, y, y_pred_prob_full,
            idx_train, idx_val, idx_test,
            base_out, variables
        )
        for k, p in split_paths.items():
            print(f"[SAVE] Split CSV ({k}) -> {p}")

        # === UNIFIED REPORT ===
        eval_dir = os.path.join(base_out, "evaluation")
        generate_combined_report(
            y_true=y_test,
            y_pred_prob=y_pred_prob,
            n_pigments=int(variables["num_files"]),
            output_dir=eval_dir,
            name=name
        )

        # === Calcular matrices de coactivación soft ===
        n_p = int(variables["num_files"])
        y_pig_true, y_mix_true = y_test[:, :n_p], y_test[:, n_p:n_p+4]
        y_pig_pred, y_mix_pred = y_pred_prob[:, :n_p], y_pred_prob[:, n_p:n_p+4]
        classes_pig = [f"P{i+1:02d}" for i in range(n_p)]
        classes_mix = ["Pure", "M1", "M2", "M3"]

        cm_pig_soft = soft_confusion_matrix(y_pig_true, y_pig_pred, classes_pig)
        cm_mix_soft = soft_confusion_matrix(y_mix_true, y_mix_pred, classes_mix)

        # === Otras matrices opcionales ===
        cm_puremix_soft = soft_confusion_matrix(y_test[:, :n_p+4], y_pred_prob[:, :n_p+4], class_names=None)
        cm_soft_perPigMix = np.dot(y_test.T, y_pred_prob)

        # === Diccionario para resumen ===
        cm_dict = {
            "pigments": {"matrix": cm_pig_soft, "classes": classes_pig, "desc": "Pigment coactivation"},
            "mixtures": {"matrix": cm_mix_soft, "classes": classes_mix, "desc": "Mixture coactivation"},
            "pure_mix": {"matrix": cm_puremix_soft, "classes": ["Pure/Mix"], "desc": "Pure vs Mix"},
            "perPigMix": {"matrix": cm_soft_perPigMix, "classes": [f"C{i+1}" for i in range(y_test.shape[1])],
                          "desc": "Per pigment × mixture coactivation"},
        }

        # === MÉTRICAS DETALLADAS ===
        detailed_results = {"global": compute_metrics(y_test, (y_pred_prob > 0.5).astype(int), y_pred_prob)}

        csv_out_dir = os.path.join(base_out, "datasets")
        os.makedirs(csv_out_dir, exist_ok=True)
        metrics_csv = os.path.join(csv_out_dir, f"{name}_metrics_summary.csv")

        header = ["model_name", "scope", "metric_name", "value", "description"]
        write_header = not os.path.exists(metrics_csv)
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            metrics, descriptions = detailed_results["global"]
            for key, value in metrics.items():
                writer.writerow([name, "global", key, round(value, 6), descriptions.get(key, "")])

        print(f"[SAVE] Detailed metrics report -> {metrics_csv}")

        # === SUMMARY (coactivación vs métricas globales) ===
        summarize_confusion_matrices(cm_dict, detailed_results, name, base_out)

        # === Extra summaries ===
        summarize_model(name, variables, X_train, y_train, model, base_out)
        per_pigment_metrics(y_test, y_pred_prob, variables, base_out, name)
        top_confusions(cm_soft_perPigMix, [f"P{i+1}" for i in range(y_test.shape[1])], base_out, name)
        plot_comparative_spectra(X_test, y_test, y_pred_prob, base_out, name, pigment_indices=[0, 5, 10])

        # === Conclusiones finales ===
        write_conclusions(detailed_results, base_out, name)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
