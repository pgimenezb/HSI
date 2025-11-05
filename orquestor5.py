# ============================================================================ #
#  PIPELINE: TRAIN & EVALUATE MODEL USING MANUAL CSV SELECTION
# ============================================================================ #
import os
import re
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Local project imports ---
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
    rebalance_by_pigment,
    analyze_balance_vs_recall,
)

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #
CSV_PATH = "/home/pgimenez/projects/HSI/hsi_lab/data/processor_synthetic_mixtures.csv"

# ============================================================================ #
# MAIN PIPELINE
# ============================================================================ #
def main():
    args = parse_args()
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    csv_base = os.path.splitext(os.path.basename(CSV_PATH))[0]

    for name in model_names:
        folder_name = f"{name}_{csv_base.replace('processor_', '')}"
        out_dir = os.path.join("outputs", folder_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n🚀 Starting pipeline for model: {name}")
        print(f"[SETUP] Output directory → {out_dir}")

        # ====================================================================== #
        # 1️⃣ LOAD CSV + APPLY QUOTAS
        # ====================================================================== #
        print(f"[LOAD] Loading dataset from: {CSV_PATH}")
        df_raw = pd.read_csv(CSV_PATH)
        print(f"[OK] Loaded dataset with {len(df_raw)} samples, {df_raw.shape[1]} columns")

        if variables.get("apply_region_quotas", True) and "Region" in df_raw.columns:
            df_balanced = apply_per_file_region_quotas(df_raw, variables)
            print("[INFO] Region quotas applied to base dataset.")
        else:
            df_balanced = df_raw.copy()
            print("[INFO] Region quotas skipped (no 'Region' column).")

        # ====================================================================== #
        # 2️⃣ STRATIFIED TRAIN/VAL/TEST SPLIT
        # ====================================================================== #
        idx_train, idx_val, idx_test = stratified_balanced_split(
            df_balanced, test_size=0.15, val_size=0.15, seed=variables.get("seed", 42)
        )

        df_train = df_balanced.iloc[idx_train].reset_index(drop=True)
        df_val   = df_balanced.iloc[idx_val].reset_index(drop=True)
        df_test  = df_balanced.iloc[idx_test].reset_index(drop=True)
        print(f"[OK] Split complete: train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")

        # ====================================================================== #
        # 3️⃣ BUILD DATA MATRICES
        # ====================================================================== #
        X_train, y_train, input_len = build_Xy(df_train)
        X_val,   y_val,   _         = build_Xy(df_val)
        X_test,  y_test,  _         = build_Xy(df_test)
        num_classes = y_train.shape[1]

        # ====================================================================== #
        # ⚖️ REBALANCE BY PIGMENT (applied to train/val/test)
        # ====================================================================== #
        region_row_quota = variables.get("region_row_quota", {})
        target_per_pigment = region_row_quota.get(1, 300)
        print(f"[INFO] Target rows per pigment: {target_per_pigment}")

        def apply_rebalance(df_part, y_part, split_name):
            idx_bal = rebalance_by_pigment(df_part, y_part, target_per_pigment)
            df_bal = df_part.iloc[idx_bal].reset_index(drop=True)
            y_bal = y_part[idx_bal]
            counts = y_bal.sum(axis=0)
            balance_csv = os.path.join(out_dir, f"{name}_{split_name}_pigment_balance.csv")
            pd.DataFrame({
                "Pigment": [f"P{i+1:02d}" for i in range(len(counts))],
                "Count": counts.astype(int)
            }).to_csv(balance_csv, index=False)
            print(f"[SAVE] Pigment balance ({split_name}) → {balance_csv}")
            return df_bal, y_bal

        df_train, y_train = apply_rebalance(df_train, y_train, "train")
        df_val,   y_val   = apply_rebalance(df_val,   y_val,   "val")
        df_test,  y_test  = apply_rebalance(df_test,  y_test,  "test")

        print(f"[INFO] Final balanced sets: train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")

        # ====================================================================== #
        # 4️⃣ SAVE DATASETS
        # ====================================================================== #
        for d, n in [(df_train,"train"),(df_val,"val"),(df_test,"test")]:
            p = os.path.join(out_dir, f"{folder_name}_{n}.csv")
            d.to_csv(p, index=False)
            print(f"[SAVE] {n.capitalize()} split → {p} ({len(d)} samples)")

        # ====================================================================== #
        # 5️⃣ TRAIN MODEL
        # ====================================================================== #
        print(f"\n[TRAINING] Training model '{name}' ...")
        tune = import_model_trainer(name)
        res = tune(
            X_train, y_train, X_val, y_val,
            input_len=input_len, num_classes=num_classes,
            trials=args.trials or variables.get("trials"),
            epochs=args.epochs or variables.get("epochs"),
            batch_size=args.batch_size or variables.get("batch_size"),
            n_jobs=variables.get("optuna_n_jobs", 1),
            seed=variables.get("seed", 42)
        )
        model = res[0] if isinstance(res, tuple) else res
        print(f"[OK] Model '{name}' trained successfully.")

        # ====================================================================== #
        # 6️⃣ EVALUATE ON TEST
        # ====================================================================== #
        print(f"\n[TEST] Evaluating model '{name}' ...")
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred_bin = np.zeros_like(y_pred_prob, dtype=int)
        for i in range(len(y_pred_prob)):
            top2 = np.argsort(y_pred_prob[i])[-2:]
            y_pred_bin[i, top2] = 1

        metrics_ref, descR = compute_metrics(y_test, y_pred_bin, y_pred_prob)
        metrics_path = os.path.join(out_dir, f"{folder_name}_metrics.csv")
        pd.DataFrame({
            "Metric": list(metrics_ref.keys()),
            "Value": [float(v) if not isinstance(v, (dict, tuple)) else np.nan for v in metrics_ref.values()],
            "Description": [descR[k] for k in metrics_ref.keys()]
        }).to_csv(metrics_path, index=False)
        print(f"[SAVE] Metrics summary → {metrics_path}")

        # ====================================================================== #
        # 7️⃣ DETAILED PREDICTIONS TABLE (TEST)
        # ====================================================================== #
        print(f"\n[PREDICT] Generating detailed prediction table for {len(y_test)} test samples...")
        df_pred_ref = df_test.copy()
        pigment_names = sorted({p.strip() for sub in df_pred_ref["File"].astype(str)
                                for p in sub.split(";") if p.strip()})

        def decode(vec): return [pigment_names[i] for i,v in enumerate(vec[:len(pigment_names)]) if v>0.5]

        records=[]
        for i in range(len(y_test)):
            true_row=df_test.iloc[i]
            records.append({
                "Sample":f"S{i+1}",
                "File":";".join(decode(y_test[i])),
                "True_Multi":";".join(map(str,y_test[i].astype(int))),
                "Pred_File":";".join(decode(y_pred_bin[i])),
                "Pred_Multi":";".join(map(str,y_pred_bin[i].astype(int))),
                "w1_true":true_row.get("w1",np.nan),
                "w2_true":true_row.get("w2",np.nan)
            })
        df_detailed=pd.DataFrame(records)
        detailed_csv=os.path.join(out_dir,f"{folder_name}_predictions_detailed.csv")
        df_detailed.to_csv(detailed_csv,index=False)
        print(f"[SAVE] Detailed predictions table → {detailed_csv}")

        # ====================================================================== #
        # 8️⃣ CONFUSION MATRIX: TRUE PREFIX vs ARGMAX (P21-safe)
        # ====================================================================== #
        print("\n[CONFUSION] Computing prefix-based dominant confusion matrix (P21-safe)...")
        from report import plot_pigment_confusion_matrix

        df_dom=df_detailed.copy()
        def extract_prefix(s):
            if isinstance(s,str) and s.strip():
                f=s.split(";")[0].strip()
                m=re.match(r"^(\d{1,3})[_-]?",f)
                if m: return int(m.group(1))
            return None
        df_dom["True_Prefix"]=df_dom["File"].apply(extract_prefix)

        num_samples=len(df_dom)
        num_pigments=max(21,y_pred_prob.shape[1])
        pigment_names=[f"P{i:02d}" for i in range(1,num_pigments+1)]

        if len(y_pred_prob)!=len(df_dom):
            n=min(len(y_pred_prob),len(df_dom))
            y_pred_prob=y_pred_prob[:n]; df_dom=df_dom.iloc[:n]
        pred_idx=np.argmax(y_pred_prob,axis=1)
        df_dom["Pred_Index"]=pred_idx+1
        df_dom["Pred_Prefix"]=df_dom["Pred_Index"]

        y_true=np.zeros((len(df_dom),num_pigments),int)
        y_pred=np.zeros((len(df_dom),num_pigments),int)
        for i,r in enumerate(df_dom.itertuples()):
            if r.True_Prefix and 1<=r.True_Prefix<=num_pigments:
                y_true[i,r.True_Prefix-1]=1
            if 0<=pred_idx[i]<num_pigments:
                y_pred[i,pred_idx[i]]=1

        prefix_csv=os.path.join(out_dir,f"{folder_name}_dominance_prefix_vs_argmax.csv")
        df_dom[["File","Pred_File","True_Prefix","Pred_Prefix"]].to_csv(prefix_csv,index=False)
        print(f"[SAVE] True prefix vs Predicted argmax table → {prefix_csv}")

        conf_path=os.path.join(out_dir,f"{folder_name}_confusion_prefix_vs_argmax.png")
        plot_pigment_confusion_matrix(
            y_true=y_true,y_pred=y_pred,pigment_names=pigment_names,
            save_path=conf_path,title=f"{name} – Confusion (True Prefix vs Pred Argmax)"
        )
        print(f"[SAVE] Prefix-based confusion matrix → {conf_path}")

        # ====================================================================== #
        # 9️⃣ ANALYSIS: BALANCE vs RECALL
        # ====================================================================== #
        print("\n[ANALYSIS] Running pigment dominance vs recall correlation...")
        analysis_dir=os.path.join(out_dir,"analysis_balance_vs_recall")
        os.makedirs(analysis_dir,exist_ok=True)
        analyze_balance_vs_recall(
            mixtures_csv=CSV_PATH,
            predictions_csv=detailed_csv,
            out_dir=analysis_dir
        )
        print(f"[DONE] Correlation & dominance analysis completed → {analysis_dir}")

# ============================================================================ #
if __name__ == "__main__":
    main()
