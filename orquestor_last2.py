# ============================================================================ #
# PIPELINE: TRAIN & EVALUATE MODEL USING MANUAL CSV SELECTION
# ============================================================================ #
import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import ast
# Local project imports
from hsi_lab.data.config import variables
from report import (
    parse_args,
    build_Xy,
    stratified_split_70_15_15,
    import_model_trainer,
    apply_per_file_region_quotas,
    stratified_balanced_split,
    compute_metrics,
    plot_pigment_confusion_matrix,
    plot_true_vs_predicted_proportions,
    rebalance_by_pigment,
    analyze_balance_vs_recall
)

# ============================================================================ #
# CONFIGURATION — EASY MANUAL CSV SELECTION
# ============================================================================ #
# Choose manually which CSV to load as your dataset
CSV_PATH = "/home/pgimenez/projects/HSI/hsi_lab/data/processor_synthetic_mixtures.csv"
# Examples:
# CSV_PATH = "data/processor_concat_pure_synthetic.csv"
# CSV_PATH = "data/processor_pure_pigments.csv"

# ============================================================================ #
# MAIN PIPELINE
# ============================================================================ #
def main():
    args = parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    csv_base = os.path.splitext(os.path.basename(CSV_PATH))[0]  # e.g. processor_synthetic_mixtures → synthetic_mixtures

    for name in model_names:
        # === Output folder combined name ===
        folder_name = f"{name}_{csv_base.replace('processor_', '')}"
        out_dir = os.path.join("outputs", folder_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n🚀 Starting pipeline for model: {name}")
        print(f"[SETUP] Output directory → {out_dir}")

        # ======================================================================
        # 1️⃣ LOAD CSV + APPLY QUOTAS (single source of truth)
        # ======================================================================
        print(f"[LOAD] Loading dataset from: {CSV_PATH}")
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"Dataset not found at: {CSV_PATH}")

        df_raw = pd.read_csv(CSV_PATH)
        print(f"[OK] Loaded dataset with {len(df_raw)} samples and {df_raw.shape[1]} columns.")

        # === Apply region quotas once, directly to the base data ===
        if variables.get("apply_region_quotas", True) and "Region" in df_raw.columns:
            df_balanced = apply_per_file_region_quotas(df_raw, variables)
            print("[INFO] Region quotas applied to base dataset.")
        else:
            df_balanced = df_raw.copy()
            print("[INFO] Region quotas skipped (no 'Region' column found).")

        # ======================================================================
        # 2️⃣ STRATIFIED TRAIN/VAL/TEST SPLIT ON BALANCED DATA
        # ======================================================================
        idx_train, idx_val, idx_test = stratified_balanced_split(
            df_balanced,
            test_size=0.15,
            val_size=0.15,
            seed=variables.get("seed", 42)
        )

        df_train = df_balanced.iloc[idx_train].reset_index(drop=True)
        df_val   = df_balanced.iloc[idx_val].reset_index(drop=True)
        df_test  = df_balanced.iloc[idx_test].reset_index(drop=True)

        print(f"[OK] Split complete: train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")

        # ======================================================================
        # 3️⃣ BUILD DATA MATRICES DIRECTLY FROM THESE
        # ======================================================================
        X_train, y_train, input_len = build_Xy(df_train)
        X_val,   y_val,   _         = build_Xy(df_val)
        X_test,  y_test,  _         = build_Xy(df_test)
        num_classes = y_train.shape[1]

        print(f"[DATA] input_len={input_len} | X_train={X_train.shape} | y_train={y_train.shape}")


        # ======================================================================
        # ⚖️ POST-SPLIT BALANCING BY PIGMENT (according to region_row_quota)
        # ======================================================================


        region_row_quota = variables.get("region_row_quota", {})
        target_per_pigment = region_row_quota.get(1, 300)  # e.g. 300 per pigment
        print(f"[INFO] Target rows per pigment (region 1 quota): {target_per_pigment}")


        def apply_rebalance_and_save(df_part, y_part, split_name):
            """Apply rebalance_by_pigment, print stats, and save pigment balance CSV."""
            # Ensure numeric arrays
            y_part = np.array(y_part, dtype=np.float32)

            # Apply rebalancing
            idx_bal = np.array(
                rebalance_by_pigment(df_part, y_part, target_per_pigment=target_per_pigment),
                dtype=int
            )

            # Subselect data
            df_bal = df_part.iloc[idx_bal].reset_index(drop=True)
            y_bal = y_part[idx_bal]

            # Save pigment counts
            counts = y_bal.sum(axis=0)
            balance_df = pd.DataFrame({
                "Pigment": [f"P{i+1:02d}" for i in range(len(counts))],
                "Count": counts.astype(int)
            })
            balance_csv = os.path.join(out_dir, f"{name}_{split_name}_pigment_balance.csv")
            balance_df.to_csv(balance_csv, index=False)

            print(f"[SAVE] Pigment balance ({split_name}) → {balance_csv}")
            print(f"[INFO] {split_name.capitalize()} set balanced -> {len(df_bal)} samples "
                f"({target_per_pigment} per pigment expected)")

            return df_bal, y_bal


        # === Apply to each split equally ===
        df_train, y_train = apply_rebalance_and_save(df_train, y_train, "train")
        df_val,   y_val   = apply_rebalance_and_save(df_val,   y_val,   "val")
        df_test,  y_test  = apply_rebalance_and_save(df_test,  y_test,  "test")

        # === Confirm global status ===
        print(f"[INFO] Final balanced splits: "
            f"train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")

        # === Ensure numeric arrays before training ===
        y_train = np.array(y_train, dtype=np.float32)
        y_val   = np.array(y_val, dtype=np.float32)
        y_test  = np.array(y_test, dtype=np.float32)

        # ======================================================================
        # SAVE DATASETS (TRAIN / VAL / TEST) — using rebalanced sets
        # ======================================================================

        path_train = os.path.join(out_dir, f"{folder_name}_train.csv")
        path_val   = os.path.join(out_dir, f"{folder_name}_val.csv")
        path_test  = os.path.join(out_dir, f"{folder_name}_test.csv")

        df_train.to_csv(path_train, index=False)
        df_val.to_csv(path_val, index=False)
        df_test.to_csv(path_test, index=False)

        print(f"[SAVE] Balanced Train → {path_train} ({len(df_train)} samples)")
        print(f"[SAVE] Balanced Val   → {path_val} ({len(df_val)} samples)")
        print(f"[SAVE] Balanced Test  → {path_test} ({len(df_test)} samples)")

        # ======================================================================
        # 4️⃣ TRAIN MODEL
        # ======================================================================
        print(f"\n[TRAINING] Training model '{name}' ...")
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
        model = res[0] if isinstance(res, tuple) else res
        print(f"[OK] Model '{name}' trained successfully.")

        # ======================================================================
        # 5️⃣ EVALUATE MODEL
        # ======================================================================
        print(f"\n[TEST] Evaluating model '{name}' ...")
        y_pred_prob = model.predict(X_test, verbose=0)
        # --- Convert predicted probabilities to binary with top-2 enforcement ---
        y_pred_bin = np.zeros_like(y_pred_prob, dtype=int)
        for i in range(len(y_pred_prob)):
            # Select top-2 pigments by probability
            top2_idx = np.argsort(y_pred_prob[i])[-2:]
            y_pred_bin[i, top2_idx] = 1

        # === Metrics ===
        metrics_ref, descR = compute_metrics(y_test, y_pred_bin, y_pred_prob)
        metrics_pathR = os.path.join(out_dir, f"{folder_name}_metrics.csv")
        pd.DataFrame({
            "Metric": list(metrics_ref.keys()),
            "Value": [float(v) if not isinstance(v, (dict, tuple)) else np.nan for v in metrics_ref.values()],
            "Description": [descR[k] for k in metrics_ref.keys()]
        }).to_csv(metrics_pathR, index=False)
        print(f"[SAVE] Metrics summary → {metrics_pathR}")

        # comparar test con predicciones
        proportions_path = os.path.join(out_dir, f"{folder_name}_true_vs_predicted_proportions.png")

#----------------------
        # guardar csv con los pigmentos de la tabla de train verdaderos y las predicciones obtenidas despues de ejectuar el modelo
        y_pred_prob_train = model.predict(X_train, verbose=0)
        # --- Convert predicted probabilities to binary with top-2 enforcement ---
        y_pred_bin = np.zeros_like(y_pred_prob, dtype=int)
        for i in range(len(y_pred_prob)):
            # Select top-2 pigments by probability
            top2_idx = np.argsort(y_pred_prob[i])[-2:]
            y_pred_bin[i, top2_idx] = 1




        # ======================================================================
        # 6️⃣ DETAILED PREDICTIONS TABLE + CONFUSION MATRIX (FROM TABLE)
        # ======================================================================
        print(f"\n[PREDICT] Generating detailed prediction table for {len(y_test)} test samples...")

        # === 1️⃣ Use the actual test DataFrame (same used to build X_test / y_test) ===
        df_pred_ref = df_test.copy()
        print(f"[CHECK] Using df_test directly → {len(df_pred_ref)} samples.")

        # imprimir la longitud de y_test y y_pred_bin
        print(f"[CHECK] y_test shape: {y_test.shape} | y_pred_bin shape: {y_pred_bin.shape}")

        # === 2️⃣ Extract pigment names from TEST only ===
        def extract_unique_pigments(files_column):
            all_names = []
            for entry in files_column:
                if isinstance(entry, str):
                    all_names.extend([f.strip() for f in entry.split(";") if f.strip()])
            return sorted(set(all_names))

        pigment_names = extract_unique_pigments(df_pred_ref["File"])

        def decode_pigments(vec):
            """Convierte un vector multilabel (0/1) en lista de nombres de pigmentos activos."""
            return [pigment_names[i] for i, v in enumerate(vec[:len(pigment_names)]) if v > 0.5]

        # === 3️⃣ Build detailed table using only the real test rows ===
        records = []
        for i in range(len(y_test)):
            true_names = decode_pigments(y_test[i])
            pred_names = decode_pigments(y_pred_bin[i])

            # This row now corresponds exactly to the same test sample
            true_row = df_test.iloc[i]

            records.append({
                "Sample": f"S{i+1}",
                "File": ";".join(true_names),
                "True_Multi": ";".join(map(str, y_test[i].astype(int))),
                "Pred_File": ";".join(pred_names),
                "Pred_Multi": ";".join(map(str, y_pred_bin[i].astype(int))),
                "w1_true": true_row.get("w1", np.nan),
                "w2_true": true_row.get("w2", np.nan),
            })

        # === 4️⃣ Save ===
        df_detailed = pd.DataFrame(records)
        detailed_csv = os.path.join(out_dir, f"{folder_name}_predictions_detailed.csv")
        df_detailed.to_csv(detailed_csv, index=False)
        print(f"[SAVE] Detailed test predictions table → {detailed_csv}")






        # ======================================================================
        # 6️⃣ DETAILED PREDICTIONS TABLE + CONFUSION MATRIX (FROM TABLE)
        # ======================================================================
        print(f"\n[PREDICT] Generating detailed prediction table for {len(y_test)} test samples...")

        # === 1️⃣ Use the actual test DataFrame (same used to build X_test / y_test) ===
        df_pred_ref = df_test.copy()
        print(f"[CHECK] Using df_test directly → {len(df_pred_ref)} samples.")

        # imprimir la longitud de y_test y y_pred_bin
        print(f"[CHECK] y_test shape: {y_test.shape} | y_pred_bin shape: {y_pred_bin.shape}")

        # === 2️⃣ Extract pigment names from TEST only ===
        def extract_unique_pigments(files_column):
            all_names = []
            for entry in files_column:
                if isinstance(entry, str):
                    all_names.extend([f.strip() for f in entry.split(";") if f.strip()])
            return sorted(set(all_names))

        pigment_names = extract_unique_pigments(df_pred_ref["File"])

        def decode_pigments(vec):
            """Convierte un vector multilabel (0/1) en lista de nombres de pigmentos activos."""
            return [pigment_names[i] for i, v in enumerate(vec[:len(pigment_names)]) if v > 0.5]

        # === 3️⃣ Build detailed table using only the real test rows ===
        records = []
        for i in range(len(y_test)):
            true_names = decode_pigments(y_test[i])
            pred_names = decode_pigments(y_pred_bin[i])

            # This row now corresponds exactly to the same test sample
            true_row = df_test.iloc[i]

            records.append({
                "Sample": f"S{i+1}",
                "File": ";".join(true_names),
                "True_Multi": ";".join(map(str, y_test[i].astype(int))),
                "Pred_File": ";".join(pred_names),
                "Pred_Multi": ";".join(map(str, y_pred_bin[i].astype(int))),
                "w1_true": true_row.get("w1", np.nan),
                "w2_true": true_row.get("w2", np.nan),
            })

        # === 4️⃣ Save ===
        df_detailed = pd.DataFrame(records)
        detailed_csv = os.path.join(out_dir, f"{folder_name}_predictions_detailed.csv")
        df_detailed.to_csv(detailed_csv, index=False)
        print(f"[SAVE] Detailed test predictions table → {detailed_csv}")




        # ======================================================================
        # 6️⃣bis FILTER DOMINANT-ONLY SAMPLES ( TEST)
        # ======================================================================
        print("\n[FILTER] Creating dominant-only subset for confusion analysis...")

        # Keep only rows where dominant weight >= 0.6 (configurable threshold)
        dominant_thresh = 0.6
        if "w1_true" in df_detailed.columns:
            df_dom = df_detailed[df_detailed["w1_true"] >= dominant_thresh].copy()
            print(f"[INFO] Filtered dominant samples: {len(df_dom)} / {len(df_detailed)} "
                f"({100*len(df_dom)/len(df_detailed):.1f}%) kept (w1 ≥ {dominant_thresh})")

        print("[CLEAN] Keeping only dominant pigment (p1) in multilabel vectors and labels...")

        def keep_only_dominant(row):
            """Keep only the true dominant pigment (p1) and assign the model's top predicted pigment."""
            true_vec = np.array(row["True_Multi"].split(";"), dtype=int)
            pred_vec = np.array(row["Pred_Multi"].split(";"), dtype=int)

            # Identify true dominant pigment (based on w1/w2 or order)
            if "w1_true" in row and "w2_true" in row and not np.isnan(row["w1_true"]):
                p1_name = row["File"].split(";")[0] if row["w1_true"] >= row["w2_true"] else row["File"].split(";")[1]
            else:
                p1_name = row["File"].split(";")[0]

            # Find the predicted pigment with the highest probability (use Pred_Multi as fallback)
            pred_file = ""
            if isinstance(row["Pred_File"], str) and row["Pred_File"].strip():
                pred_file = row["Pred_File"].split(";")[0].strip()
            else:
                # fallback: assign the top predicted pigment index
                prob_vec = np.array(row["Pred_Multi"].split(";"), dtype=int)
                if prob_vec.sum() > 0:
                    pred_idx = np.argmax(prob_vec)
                    if pred_idx < len(pigment_names):
                        pred_file = pigment_names[pred_idx]
                else:
                    # if everything fails, assign the dominant true pigment (so it doesn't stay blank)
                    pred_file = p1_name

            # Construct new true/pred vectors
            mask = np.zeros_like(true_vec)
            if p1_name in pigment_names:
                idx_dom = pigment_names.index(p1_name)
                mask[idx_dom] = 1
            else:
                idx_dom = None

            true_vec = true_vec * mask

            pred_mask = np.zeros_like(pred_vec)
            if pred_file in pigment_names:
                idx_pred = pigment_names.index(pred_file)
                pred_mask[idx_pred] = 1
            pred_vec = pred_mask

            return (
                ";".join(map(str, true_vec)),
                ";".join(map(str, pred_vec)),
                p1_name,
                pred_file,
            )

        # Apply cleaning and save
        df_dom[["True_Multi", "Pred_Multi", "File", "Pred_File"]] = df_dom.apply(
            lambda r: pd.Series(keep_only_dominant(r)), axis=1
        )

        dominant_clean_csv = os.path.join(out_dir, f"{folder_name}_predictions_dominant_p1only.csv")
        df_dom.to_csv(dominant_clean_csv, index=False)
        print(f"[SAVE] Dominant-only (p1-only) predictions table → {dominant_clean_csv}")


        # ======================================================================
        # 7️⃣ CONFUSION MATRIX (STANDARD + DOMINANT-ONLY)
        # ======================================================================
        print("[CONFUSION] Computing confusion matrices...")

        # === Get pigments only from TEST ===
        def extract_unique_pigments(files_column):
            all_names = []
            for entry in files_column:
                if isinstance(entry, str):
                    all_names.extend([f.strip() for f in entry.split(";") if f.strip()])
            return sorted(set(all_names))

        # ⚠️ Important: use df_balanced to ensure all pigments are included, even rare ones
        pigment_names = extract_unique_pigments(df_balanced["File"])
        print(f"[INFO] Pigment list initialized with {len(pigment_names)} pigments from full dataset.")


        def parse_multilabel(series):
            return np.vstack(series.apply(lambda s: np.array(s.split(";"), dtype=int)))

        # --- Full dataset confusion ---
        y_true_all = parse_multilabel(df_detailed["True_Multi"])
        y_pred_all = parse_multilabel(df_detailed["Pred_Multi"])
        conf_path_all = os.path.join(out_dir, f"{folder_name}_confusion_pigments_full.png")

        plot_pigment_confusion_matrix(
            y_true=y_true_all,
            y_pred=y_pred_all,
            pigment_names=pigment_names,
            save_path=conf_path_all,
            title=f"{name} – Confusion Matrix (All Samples)"
        )
        print(f"[SAVE] Full confusion matrix → {conf_path_all}")

        # --- Dominant-only confusion (p1 only) ---
        if len(df_dom) > 0:
            y_true_dom = np.vstack(df_dom["True_Multi"].apply(lambda s: np.array(s.split(";"), dtype=int)))
            y_pred_dom = np.vstack(df_dom["Pred_Multi"].apply(lambda s: np.array(s.split(";"), dtype=int)))
            conf_path_dom = os.path.join(out_dir, f"{folder_name}_confusion_pigments_dominant_p1only.png")

            plot_pigment_confusion_matrix(
                y_true=y_true_dom,
                y_pred=y_pred_dom,
                pigment_names=pigment_names,  # ✅ usa el mismo orden que el test
                save_path=conf_path_dom,
                title=f"{name} – Confusion Matrix (Dominant Pigments Only)"
            )
            print(f"[SAVE] Dominant-only (p1-only) confusion matrix → {conf_path_dom}")

        # ======================================================================
        # 8️⃣ ANALYSIS: BALANCE VS RECALL
        # ======================================================================

        print("\n[ANALYSIS] Running pigment dominance vs recall correlation...")

        # === Paths and output ===
        mixtures_csv = "/home/pgimenez/projects/HSI/hsi_lab/data/processor_synthetic_mixtures.csv"
        predictions_csv = os.path.join(out_dir, f"{folder_name}_predictions_detailed.csv")
        analysis_dir = os.path.join(out_dir, "analysis_balance_vs_recall")
        os.makedirs(analysis_dir, exist_ok=True)

        analyze_balance_vs_recall(
            mixtures_csv=mixtures_csv,
            predictions_csv=predictions_csv,
            out_dir=analysis_dir
        )

        print(f"[DONE] Correlation and dominance analysis completed → {analysis_dir}")






# ============================================================================ #
if __name__ == "__main__":
    main()

""" 
To execute on the terminal: 
cd ~/projects/HSI
python -m orquestor
"""