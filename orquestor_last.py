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
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
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
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

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
        # 2️⃣ SIMPLE BALANCED SPLIT BY PIGMENT ROLES (w1 & w2)
        # ======================================================================
        print("\n[INFO] Performing simplified balanced split (70/15/15) by pigment roles...")

        # --- Copy and extract pigment roles (w1, w2)
        df_balanced = df_balanced.copy()

        def extract_roles(file_str):
            if isinstance(file_str, str):
                parts = [p.strip() for p in file_str.split(";") if p.strip()]
                if len(parts) == 1:
                    return parts[0], None
                elif len(parts) >= 2:
                    return parts[0], parts[1]
            return None, None

        df_balanced[["Pigment_w1", "Pigment_w2"]] = df_balanced["File"].apply(
            lambda f: pd.Series(extract_roles(f))
        )

        # --- Stratified split by dominant pigment (w1)
        train_idx, temp_idx = train_test_split(
            np.arange(len(df_balanced)),
            test_size=0.30,
            stratify=df_balanced["Pigment_w1"],
            random_state=42
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            stratify=df_balanced.loc[temp_idx, "Pigment_w1"],
            random_state=42
        )

        # --- Build DataFrames
        df_train = df_balanced.iloc[train_idx].reset_index(drop=True)
        df_val   = df_balanced.iloc[val_idx].reset_index(drop=True)
        df_test  = df_balanced.iloc[test_idx].reset_index(drop=True)

        print(f"[OK] Split complete: train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")

        # ======================================================================
        # 3️⃣ CHECK & SAVE PIGMENT ROLE COUNTS
        # ======================================================================
        def pigment_role_counts(df, split_name):
            c_w1 = Counter(df["Pigment_w1"])
            c_w2 = Counter(df["Pigment_w2"].dropna())
            pigments = sorted(set(c_w1.keys()) | set(c_w2.keys()))
            df_counts = pd.DataFrame({
                "Pigment": pigments,
                "Count_w1": [c_w1.get(p, 0) for p in pigments],
                "Count_w2": [c_w2.get(p, 0) for p in pigments],
                "Total": [c_w1.get(p, 0) + c_w2.get(p, 0) for p in pigments]
            })
            csv_path = os.path.join(out_dir, f"{name}_{split_name}_role_counts.csv")
            df_counts.to_csv(csv_path, index=False)
            print(f"[SAVE] {split_name} pigment role counts → {csv_path}")
            return df_counts

        counts_train = pigment_role_counts(df_train, "train")
        counts_val   = pigment_role_counts(df_val, "val")
        counts_test  = pigment_role_counts(df_test, "test")

        # --- Quick print summary
        print("\n[INFO] Example pigment role distribution (train):")
        print(counts_train.head())

        # ======================================================================
        # 4️⃣ BUILD MATRICES
        # ======================================================================
        X_train, y_train, input_len = build_Xy(df_train)
        X_val,   y_val,   _         = build_Xy(df_val)
        X_test,  y_test,  _         = build_Xy(df_test)

        num_classes = y_train.shape[1]
        print(f"[DATA] input_len={input_len} | X_train={X_train.shape} | y_train={y_train.shape}")



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


        """""
        # === DEBUG: SAVE AND INSPECT MODEL PREDICTIONS ===
        pred_debug_dir = os.path.join(out_dir, "predictions_debug")
        os.makedirs(pred_debug_dir, exist_ok=True)

        # --- Shape and basic check ---
        print(f"[DEBUG] y_pred_prob shape: {y_pred_prob.shape} | range = ({y_pred_prob.min():.4f}, {y_pred_prob.max():.4f})")

        # --- Optional: preview of first sample ---
        np.set_printoptions(precision=4, suppress=True)
        print("[DEBUG] First sample prediction (probabilities per label):")
        print(y_pred_prob[0])

        # --- Convert to DataFrame for CSV inspection ---
        n_labels = y_pred_prob.shape[1]
        pred_cols = [f"Pred_{i+1:02d}" for i in range(n_labels)]
        pred_df = pd.DataFrame(y_pred_prob, columns=pred_cols)

        # Add file/sample info if available
        if "File" in df_test.columns:
            pred_df.insert(0, "File", df_test["File"].values)
        if "w1" in df_test.columns:
            pred_df.insert(1, "w1_true", df_test["w1"].values)
        if "w2" in df_test.columns:
            pred_df.insert(2, "w2_true", df_test["w2"].values)

        # --- Save CSV (for full inspection) ---
        pred_csv_path = os.path.join(pred_debug_dir, f"{folder_name}_y_pred_prob.csv")
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"[SAVE] Raw prediction probabilities → {pred_csv_path}")  """

        
        # ====================================================================== #
        # 🧩 PREDICTED PROBABILITIES → TOP-2 BINARY (ORDERED)
        # ====================================================================== #
        y_pred_bin = np.zeros_like(y_pred_prob, dtype=int)

        for i in range(len(y_pred_prob)):
            # Select top-2 indices sorted by descending probability
            sorted_idx = np.argsort(y_pred_prob[i])[::-1][:2]
            y_pred_bin[i, sorted_idx] = 1

        # ====================================================================== #
        # 🧾 GENERATE TEST PREVIEW CSV (TRUE vs PRED, ORDERED by PROBABILITY)
        # ====================================================================== #
        preview_dir = os.path.join(out_dir, "datasets_debug")
        os.makedirs(preview_dir, exist_ok=True)

        def extract_unique_pigments(files_column):
            all_names = []
            for entry in files_column:
                if isinstance(entry, str):
                    all_names.extend([f.strip() for f in entry.split(";") if f.strip()])
            return sorted(set(all_names))

        pigment_names = extract_unique_pigments(df_balanced["File"])

        # ====================================================================== #
        # 🧩 BUILD TRUE vs PRED TABLE (aligned per-row order of mixtures)
        # ====================================================================== #

        # Choose prediction mode: "w1", "w2", or "full"
        pred_mode = "w1"   # ⬅️ cambia esto según quieras filtrar por w1, w2 o full

        records = []
        for i in range(len(y_test)):
            file_entry = df_test.iloc[i]["File"]
            true_names = [f.strip() for f in str(file_entry).split(";") if f.strip()]
            true_vec = y_test[i]
            pred_vec = y_pred_prob[i]  # use raw probabilities for matching

            # --- Identify top-2 predicted pigments ---
            top2_idx = np.argsort(pred_vec)[::-1][:2]
            top2_preds = [(pigment_names[j], pred_vec[j]) for j in top2_idx if j < len(pigment_names)]

            # --- Align predicted pigments with the order of the true pigments ---
            pred_ordered = []
            for true_name in true_names:
                # If true pigment is among the top2, keep it first
                match = [p for p, _ in top2_preds if p == true_name]
                if match:
                    pred_ordered.append(match[0])
                else:
                    # otherwise pick the most probable remaining one
                    for p, _ in top2_preds:
                        if p not in pred_ordered:
                            pred_ordered.append(p)
                            break

            # --- Ensure we always have 2 predictions (pad if missing) ---
            while len(pred_ordered) < 2:
                pred_ordered.append(
                    top2_preds[len(pred_ordered)][0] if len(top2_preds) > len(pred_ordered) else "None"
                )

            # --- Create binary prediction vector depending on the mode ---
            y_pred_bin[i] = np.zeros_like(y_pred_bin[i])

            if pred_mode == "w1" and len(pred_ordered) > 0:
                first_pred = pred_ordered[0]
                if first_pred in pigment_names:
                    y_pred_bin[i, pigment_names.index(first_pred)] = 1

            elif pred_mode == "w2" and len(pred_ordered) > 1:
                second_pred = pred_ordered[1]
                if second_pred in pigment_names:
                    y_pred_bin[i, pigment_names.index(second_pred)] = 1

            elif pred_mode == "full":
                for p in pred_ordered[:2]:
                    if p in pigment_names:
                        y_pred_bin[i, pigment_names.index(p)] = 1

            # --- True vector stays as is (or you could also limit to dominant pigment) ---
            records.append({
                "Sample": f"S{i+1}",
                "File": ";".join(true_names),
                "True_Multi": ";".join(map(str, true_vec.astype(int))),
                "Pred_File": ";".join(pred_ordered),
                "Pred_Multi": ";".join(map(str, y_pred_bin[i].astype(int))),
                "P1_true": true_names[0] if len(true_names) > 0 else "",
                "P2_true": true_names[1] if len(true_names) > 1 else "",
                "P1_pred": pred_ordered[0],
                "P2_pred": pred_ordered[1],
                "w1_true": df_test.iloc[i].get("w1", np.nan),
                "w2_true": df_test.iloc[i].get("w2", np.nan),
            })

        df_true_vs_pred = pd.DataFrame(records)

        # === Save CSV ===
        true_vs_pred = os.path.join(preview_dir, f"{folder_name}_True_vs_Pred_{pred_mode}.csv")
        df_true_vs_pred.to_csv(true_vs_pred, index=False)
        print(f"[SAVE] Test vs Prediction preview ({pred_mode} mode) → {true_vs_pred}")

            
        # ======================================================================
        # 📊 EXTRA SUMMARY: pigment counts as w1/w2 in True & Pred
        # ======================================================================

        # --- Helper to extract roles (same logic as synthetic mixtures)
        def extract_roles(file_str):
            if isinstance(file_str, str):
                parts = [p.strip() for p in file_str.split(";") if p.strip()]
                if len(parts) == 1:
                    return parts[0], None
                elif len(parts) >= 2:
                    return parts[0], parts[1]
            return None, None

        # --- Count occurrences
     

        def count_roles(series):
            """Return separate Counters for w1 and w2 positions."""
            c_w1, c_w2 = Counter(), Counter()
            for entry in series.dropna():
                w1, w2 = extract_roles(entry)
                if w1: c_w1[w1] += 1
                if w2: c_w2[w2] += 1
            return c_w1, c_w2

        # --- Apply for TRUE and PRED parts
        c_true_w1, c_true_w2 = count_roles(df_true_vs_pred["File"])
        c_pred_w1, c_pred_w2 = count_roles(df_true_vs_pred["Pred_File"])

        # --- Combine all pigments
        all_pigs = sorted(set(c_true_w1) | set(c_true_w2) | set(c_pred_w1) | set(c_pred_w2))

        df_counts = pd.DataFrame({
            "Pigment": all_pigs,
            "Count_w1_true": [c_true_w1.get(p, 0) for p in all_pigs],
            "Count_w2_true": [c_true_w2.get(p, 0) for p in all_pigs],
            "Count_w1_pred": [c_pred_w1.get(p, 0) for p in all_pigs],
            "Count_w2_pred": [c_pred_w2.get(p, 0) for p in all_pigs],
        })

        df_counts["Total_true"] = df_counts["Count_w1_true"] + df_counts["Count_w2_true"]
        df_counts["Total_pred"] = df_counts["Count_w1_pred"] + df_counts["Count_w2_pred"]

        # --- Save CSV
        counts_csv = os.path.join(preview_dir, f"{folder_name}_TruePred_role_counts.csv")
        df_counts.to_csv(counts_csv, index=False)
        print(f"[SAVE] Pigment role counts summary → {counts_csv}")

        # --- Optional: quick preview
        print(df_counts.head())


        """""
        # === DEBUG: SAVE AND INSPECT BINARIZED PREDICTIONS ===
        bin_debug_dir = os.path.join(out_dir, "predictions_debug")
        os.makedirs(bin_debug_dir, exist_ok=True)

        print(f"[DEBUG] y_pred_bin shape: {y_pred_bin.shape}")
        print("[DEBUG] First 5 rows of y_pred_bin (1 = predicted pigment):")
        print(y_pred_bin[:5])

        # --- Convert to DataFrame ---
        n_labels = y_pred_bin.shape[1]
        bin_cols = [f"PredBin_{i+1:02d}" for i in range(n_labels)]
        bin_df = pd.DataFrame(y_pred_bin, columns=bin_cols)

        # Add file/sample info
        if "File" in df_test.columns:
            bin_df.insert(0, "File", df_test["File"].values)
        if "w1" in df_test.columns:
            bin_df.insert(1, "w1_true", df_test["w1"].values)
        if "w2" in df_test.columns:
            bin_df.insert(2, "w2_true", df_test["w2"].values)

        # --- Save CSV ---
        bin_csv_path = os.path.join(bin_debug_dir, f"{folder_name}_y_pred_bin.csv")
        bin_df.to_csv(bin_csv_path, index=False)
        print(f"[SAVE] Binarized predictions (top-2 per sample) → {bin_csv_path}")"""

        # === Metrics ===
        metrics_ref, descR = compute_metrics(y_test, y_pred_bin, y_pred_prob)
        metrics_pathR = os.path.join(out_dir, f"{folder_name}_metrics.csv")
        pd.DataFrame({
            "Metric": list(metrics_ref.keys()),
            "Value": [float(v) if not isinstance(v, (dict, tuple)) else np.nan for v in metrics_ref.values()],
            "Description": [descR[k] for k in metrics_ref.keys()]
        }).to_csv(metrics_pathR, index=False)
        print(f"[SAVE] Metrics summary → {metrics_pathR}")


        # === PLOT TRUE vs PREDICTED PROPORTIONS using preview table ===
        proportions_path = os.path.join(out_dir, f"{folder_name}_true_vs_predicted_proportions_from_preview.png")

        try:
            print(f"[PLOT] Generating True vs Predicted proportions chart (from test preview) ...")
            plot_true_vs_predicted_proportions(
                csv_path=true_vs_pred,        
                pigment_names=pigment_names,   
                save_path=proportions_path,
                title=f"{name} – True vs Predicted Pigment Proportions (from preview)"
            )
            print(f"[SAVE] Proportion comparison plot → {proportions_path}")
        except Exception as e:
            print(f"[WARN] Could not generate proportions plot from preview: {e}")


        # ====================================================================== #
        # 7️⃣ CONFUSION MATRIX (from Test vs Pred Preview)
        # ====================================================================== #
        print("[CONFUSION] Computing confusion matrices from test preview...")

        # === Load the preview table (previously saved) ===
        df_true_vs_pred = pd.read_csv(true_vs_pred)
        print(f"[LOAD] Preview table loaded: {len(df_true_vs_pred)} samples")

        # === Extract unique pigment names from the full dataset (to keep order consistent) ===
        def extract_unique_pigments(files_column):
            all_names = []
            for entry in files_column:
                if isinstance(entry, str):
                    all_names.extend([f.strip() for f in entry.split(";") if f.strip()])
            return sorted(set(all_names))

        pigment_names = extract_unique_pigments(df_balanced["File"])
        print(f"[INFO] Pigment list initialized with {len(pigment_names)} pigments from full dataset.")

        # === Parse multilabel columns (True_Multi, Pred_Multi) ===
        def parse_multilabel(series):
            return np.vstack(series.apply(lambda s: np.round(np.array(s.split(";"), dtype=float)).astype(int)))



        # ================================================================= #
        # 🧩 1️⃣ FULL CONFUSION MATRIX (All Samples)
        # ================================================================= #
        print("[CONFUSION] Generating FULL confusion matrix...")

        y_true_all = parse_multilabel(df_true_vs_pred["True_Multi"])
        y_pred_all = parse_multilabel(df_true_vs_pred["Pred_Multi"])

        conf_path_all = os.path.join(out_dir, f"{folder_name}_confusion_pigments_preview_full.png")
        csv_used_full = os.path.join(out_dir, f"{folder_name}_confusion_used_full.csv")
        csv_counts_full = os.path.join(out_dir, f"{folder_name}_confusion_counts_full.csv")

        # 💾 Save dataset used for full matrix
        df_true_vs_pred.to_csv(csv_used_full, index=False)
        print(f"[SAVE] Dataset used for FULL confusion matrix → {csv_used_full}")

        plot_pigment_confusion_matrix(
            y_true=y_true_all,
            y_pred=y_pred_all,
            pigment_names=pigment_names,
            save_path=conf_path_all,
            title=f"{name} – Confusion Matrix (from Test Preview – All Samples)"
        )
        print(f"[SAVE] Full confusion matrix (from preview) → {conf_path_all}")

        # === Count pigment occurrences in True and Pred ===
        true_counts_full = y_true_all.sum(axis=0)
        pred_counts_full = y_pred_all.sum(axis=0)
        df_counts_full = pd.DataFrame({
            "Pigment": pigment_names,
            "Count_in_True_Multi": true_counts_full.astype(int),
            "Count_in_Pred_Multi": pred_counts_full.astype(int)
        })
        df_counts_full.to_csv(csv_counts_full, index=False)
        print(f"[SAVE] Pigment occurrence counts (FULL) → {csv_counts_full}")


        # ================================================================= #
        # 🧩 2️⃣ DOMINANT-ONLY CONFUSION MATRIX (w1 ≥ 0.6)
        # ================================================================= #
        if "w1_true" in df_true_vs_pred.columns:
            df_dom_prev = df_true_vs_pred[df_true_vs_pred["w1_true"] >= 0.6].copy()
            print(f"[INFO] Dominant subset: {len(df_dom_prev)} / {len(df_true_vs_pred)} samples (w1 ≥ 0.6)")

            def keep_only_dominant(row):
                """Keep only the true dominant pigment (p1) and top predicted pigment."""
                true_vec = np.array(row["True_Multi"].split(";"), dtype=int)
                pred_vec = np.array(row["Pred_Multi"].split(";"), dtype=int)

                # True dominant pigment = first in File
                true_name = ""
                if isinstance(row["File"], str) and row["File"].strip():
                    true_name = row["File"].split(";")[0].strip()

                # Pred dominant pigment = aquel con mayor probabilidad predicha
                pred_name = ""
                if isinstance(row["Pred_Multi"], str) and row["Pred_Multi"].strip():
                    pred_vec = np.array(row["Pred_Multi"].split(";"), dtype=float)
                    top_idx = np.argmax(pred_vec)
                    if top_idx < len(pigment_names):
                        pred_name = pigment_names[top_idx]


                new_true = np.zeros_like(true_vec)
                new_pred = np.zeros_like(pred_vec)

                if true_name in pigment_names:
                    new_true[pigment_names.index(true_name)] = 1
                if pred_name in pigment_names:
                    new_pred[pigment_names.index(pred_name)] = 1

                return (
                    ";".join(map(str, new_true)),
                    ";".join(map(str, new_pred)),
                    true_name,
                    pred_name
                )


            # Apply dominant cleaning
            df_dom_prev[["True_Multi", "Pred_Multi", "True_P1", "Pred_P1"]] = df_dom_prev.apply(
                lambda r: pd.Series(keep_only_dominant(r)), axis=1
            )

            if len(df_dom_prev) > 0:
                y_true_dom = parse_multilabel(df_dom_prev["True_Multi"])
                y_pred_dom = parse_multilabel(df_dom_prev["Pred_Multi"])

                conf_path_dom = os.path.join(out_dir, f"{folder_name}_confusion_pigments_preview_dominant.png")
                csv_used_dom = os.path.join(out_dir, f"{folder_name}_confusion_used_dominant.csv")
                csv_counts_dom = os.path.join(out_dir, f"{folder_name}_confusion_counts_dominant.csv")

                # 💾 Save dataset used for dominant-only matrix
                df_dom_prev.to_csv(csv_used_dom, index=False)
                print(f"[SAVE] Dataset used for DOMINANT confusion matrix → {csv_used_dom}")

                plot_pigment_confusion_matrix(
                    y_true=y_true_dom,
                    y_pred=y_pred_dom,
                    pigment_names=pigment_names,
                    save_path=conf_path_dom,
                    title=f"{name} – Confusion Matrix (from Test Preview – Dominant Pigments Only, w1 ≥ 0.6)"
                )
                print(f"[SAVE] Dominant-only confusion matrix (from preview) → {conf_path_dom}")

                # === Count pigment occurrences in True and Pred ===
                true_counts_dom = y_true_dom.sum(axis=0)
                pred_counts_dom = y_pred_dom.sum(axis=0)
                df_counts_dom = pd.DataFrame({
                    "Pigment": pigment_names,
                    "Count_in_True_Multi": true_counts_dom.astype(int),
                    "Count_in_Pred_Multi": pred_counts_dom.astype(int)
                })
                df_counts_dom.to_csv(csv_counts_dom, index=False)
                print(f"[SAVE] Pigment occurrence counts (DOMINANT) → {csv_counts_dom}")

            else:
                print("[INFO] No dominant samples found (w1_true ≥ 0.6).")




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
            predictions_csv=true_vs_pred,  
            out_dir=analysis_dir
        )
        print(f"[DONE] Correlation and dominance analysis completed → {analysis_dir}")


# ============================================================================ #
if __name__ == "__main__":
    main()



""" 
To execute on the terminal: 
cd ~/projects/HSI
python orquestor_last.py --models DNN_bayesian_grid_fixed
"""