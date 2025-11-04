import os
import random
import numpy as np
import pandas as pd
from hsi_lab.data.config import variables
from hsi_lab.data.processor_pure_pigments import HSIDataProcessor

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hsi_lab.data.config import variables
from hsi_lab.data.processor_pure_pigments import HSIDataProcessor


# ======================================================================
# GENERATE SYNTHETIC 2-PIGMENT MIXTURES (BALANCED)
# ======================================================================
def generate_synthetic_2pigment_mixtures(
    df_used: pd.DataFrame,
    n_samples: int = 6300,
    out_dir: str = "/home/pgimenez/projects/HSI/hsi_lab/data"
):
    os.makedirs(out_dir, exist_ok=True)
    print("\n[PHASE A] Generating and evaluating balanced synthetic 2-pigment mixtures...")

    # === Pigments and spectral columns ===
    pigment_names = sorted(df_used["File"].unique().tolist())
    n_pigments = len(pigment_names)
    print(f"[INFO] Found {n_pigments} unique pigments from dataframe.")

    vis_cols = [c for c in df_used.columns if c.lower().startswith("vis_")]
    swir_cols = [c for c in df_used.columns if c.lower().startswith("swir_")]
    all_cols = vis_cols + swir_cols
    input_len = len(all_cols)

    # === Mean reflectance per pigment ===
    R_pigments = []
    for name_pig in pigment_names:
        dfp = df_used[df_used["File"] == name_pig]
        if len(dfp) == 0:
            continue
        mean_spec = dfp[all_cols].mean(axis=0).to_numpy(dtype=np.float32)
        R_pigments.append(mean_spec)
    R_pigments = np.array(R_pigments, dtype=np.float32)

    if R_pigments.shape[0] != n_pigments:
        print("[WARN] Some pigments missing data — check input DataFrame.")
        n_pigments = R_pigments.shape[0]

    # === Generate 2-pigment mixtures (balanced for both p1 and p2) ===
    y_true = np.zeros((n_samples, n_pigments), dtype=int)
    mixture_info = []
    X_mix = np.zeros((n_samples, input_len), dtype=np.float32)

    print(f"[INFO] Target total samples: {n_samples}")
    print(f"[INFO] Expected balanced frequency per pigment ≈ {2 * n_samples // n_pigments} appearances (p1+p2 combined)")

    # --- Step 1: create all possible unique pairs (i < j) ---
    all_pairs = []
    for i in range(n_pigments):
        for j in range(n_pigments):
            if i != j:
                all_pairs.append((i, j))
    random.shuffle(all_pairs)

    total_pairs = len(all_pairs)
    reps_per_pair = max(1, n_samples // total_pairs)
    print(f"[DEBUG] There are {total_pairs} unique pigment pairs, using each {reps_per_pair} times on average.")

    # --- Step 2: expand pairs to reach target sample count ---
    full_pairs = []
    for pair in all_pairs:
        full_pairs.extend([pair] * reps_per_pair)

    # Adjust to exact n_samples
    if len(full_pairs) < n_samples:
        full_pairs.extend(random.choices(all_pairs, k=n_samples - len(full_pairs)))
    elif len(full_pairs) > n_samples:
        full_pairs = random.sample(full_pairs, n_samples)

    random.shuffle(full_pairs)

    # --- Step 3: generate mixtures with random dominant pigment ---
    for i, (p1, p2) in enumerate(full_pairs):
        # Random dominant: 50% chance to swap roles
        if random.random() < 0.5:
            p1, p2 = p2, p1

        # Random dominant weight (always >0.5)
        w1 = np.clip(np.random.normal(0.8, 0.05), 0.7, 0.9)
        w2 = 1.0 - w1

        y_true[i, [p1, p2]] = 1
        mixture_info.append((pigment_names[p1], pigment_names[p2], w1, w2))

        mix = w1 * R_pigments[p1] + w2 * R_pigments[p2]
        mix += np.random.normal(0, 0.002, size=input_len)
        X_mix[i] = np.clip(mix, 0, 1)

    print(f"[OK] Generated {len(X_mix)} balanced mixtures.")

    # --- Step 4: build final DataFrame ---
    records = []
    for i in range(n_samples):
        p1, p2, w1, w2 = mixture_info[i]
        row = {
            "File": f"{p1};{p2}",
            "Multi": y_true[i].tolist(),
            "w1": round(w1, 3),
            "w2": round(w2, 3),
            "Region": 1
        }
        for j, col in enumerate(all_cols):
            row[col] = float(X_mix[i, j])
        records.append(row)

    df_mixtures = pd.DataFrame(records, columns=["File", "Multi", "w1", "w2", "Region"] + all_cols)

    # --- Step 5: save CSV ---
    mix_info_path = os.path.join(out_dir, "processor_synthetic_mixtures.csv")
    df_mixtures.to_csv(mix_info_path, index=False)
    print(f"[SAVE] Synthetic mixtures CSV -> {mix_info_path}")
    print(f"[INFO] DataFrame shape: {df_mixtures.shape}")

    # --- Step 6 (optional): verify pigment balance ---
    from collections import Counter
    p1_counts = Counter([p1 for p1, _, _, _ in mixture_info])
    p2_counts = Counter([p2 for _, p2, _, _ in mixture_info])
    print("\n[STATS] Pigment usage summary:")
    for i, name in enumerate(pigment_names):
        print(f"  {i+1:02d}. {name:25s} p1={p1_counts.get(name,0):4d} | p2={p2_counts.get(name,0):4d}")

    return df_mixtures

    # === Build DataFrame ===
    records = []
    for i in range(n_samples):
        p1, p2, w1, w2 = mixture_info[i]
        row = {
            "File": f"{p1};{p2}",
            "Multi": y_true[i].tolist(),
            "w1": round(w1, 3),
            "w2": round(w2, 3),
            "Region": 1  # fixed value
        }
        for j, col in enumerate(all_cols):
            row[col] = float(X_mix[i, j])
        records.append(row)

    df_mixtures = pd.DataFrame(records,
                               columns=["File", "Multi", "w1", "w2", "Region"] + all_cols)

    # === Save CSV ===
    mix_info_path = os.path.join(out_dir, "processor_synthetic_mixtures.csv")
    df_mixtures.to_csv(mix_info_path, index=False)
    print(f"[SAVE] Synthetic mixtures CSV -> {mix_info_path}")
    print(f"[INFO] DataFrame shape: {df_mixtures.shape}")
    print(f"[INFO] Example columns: {df_mixtures.columns[:10].tolist()}")

    # === Frequency analysis ===
    counts_total = y_true.sum(axis=0)
    freq_path = os.path.join(out_dir, "processor_synthetic_mixtures_balanced.csv")
    pd.DataFrame({
        "Pigment": pigment_names,
        "Count_Total(p1+p2)": counts_total
    }).to_csv(freq_path, index=False)
    print(f"[SAVE] Pigment frequency table -> {freq_path}")

    # === Plot ===
    plt.figure(figsize=(10, 4))
    plt.bar([f"P{i+1:02d}" for i in range(n_pigments)], counts_total)
    plt.title("Pigment frequency in synthetic mixtures (p1 + p2 combined)")
    plt.xlabel("Pigment")
    plt.ylabel("Total count (p1 + p2)")
    plt.tight_layout()
    plt.show()

    return df_mixtures


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    print("[INIT] Loading pure pigments dataframe...")
    processor = HSIDataProcessor(variables)
    processor.load_h5_files()
    df_used = processor.dataframe(mode="filtered")

    print("[OK] DataFrame loaded — generating balanced synthetic mixtures.")
    df_synth = generate_synthetic_2pigment_mixtures(df_used)
    print("[DONE] Synthetic mixtures generated successfully.")


"""
To execute on the terminal:
cd ~/projects/HSI
python -m hsi_lab.data.processor_synthetic_mixtures
"""
