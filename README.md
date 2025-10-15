# HSI

Training CNN/DNN models for **multi-label** classification of VIS+SWIR spectra with Optuna.

## Project Layout

Projects/
└── HSI/
├── train.py # orchestrator (load → split → tune → eval → save)
├── README.md
├── hsi_lab/
│ ├── config.py # editable experiment variables
│ ├── data/
│ │ └── processor.py # HDF5 loading → DataFrame + helpers + caching
│ ├── eval/
│ │ └── report.py # reporting metrics and confusion matrix plotting
│ └── models/
│ ├── metrics.py # Keras metrics (e.g., strict_accuracy)
│ ├── cnn_baseline.py
│ ├── cnn_residual.py
│ ├── cnn_dilated.py
│ ├── dnn_baseline.py
│ ├── dnn_selu.py
│ └── dnn_wide.py
└── outputs/
├── saved_models/
│ └── <RUN_ID>/ # heavy models (.h5 / *.keras)
└── other_outputs/
├── logs/ # per-model logs (.log)
└── figures/ # plots (confusion matrix, etc.)

## Quick Setup

```bash
conda activate hsi
pip install -r requirements.txt
# if anything is missing:
# pip install matplotlib optuna scikit-learn h5py pandas numpy
```

## Data Configuration

Edit `hsi_lab/config.py`:

- `data_folder`, `excel_file`
- `data_type`
- `num_files`, `start_index`
- `selected_regions`, `selected_subregions`
- `outputs_dir`, `models_dir` (they’re nested under `runs/<RUN_ID>` automatically)

## Run (CLI)

**Single model with full reports**
```bash
python train.py --models cnn_baseline --trials 40 --epochs 60 --reports
```

**Multiple models (same split, sequential by default)**
```bash
python train.py --models cnn_baseline,cnn_residual,cnn_dilated --trials 40 --epochs 60 --reports
```

**Parallel across models (Joblib)**
```bash
python train.py --models cnn_baseline,cnn_residual,cnn_dilated   --n-jobs-models 3 --trials 40 --epochs 60 --reports
```

**Fixed RUN_ID to group outputs**
```bash
python train.py --run-id 20250101-120000 --models cnn_baseline --reports
```
If not provided, `train.py` generates `RUN_ID` from a timestamp. Optuna storage: `outputs/runs/<RUN_ID>/optuna.db`.

## Outputs

- **Per-model logs:** `outputs/runs/<RUN_ID>/logs/<model>.log`  
  View live: `tail -f outputs/runs/<RUN_ID>/logs/cnn_baseline.log`
- **Figures:** `outputs/runs/<RUN_ID>/figures/<model>_confusion_matrix.png` (only with `--reports`)
- **Light artifacts:** `*_best_params.csv`, `*_trials_summary.csv`, `*_summary.json` under `outputs/runs/<RUN_ID>/`
- **Models:** `saved_models/<RUN_ID>/<model>.h5` and `<model>.keras`

## Tips

- Ensure packages: `hsi_lab/`, `hsi_lab/data/`, `hsi_lab/models/`, `hsi_lab/eval/` all have `__init__.py`.
- When running from repo root this usually isn’t needed, but if imports fail:
  ```bash
  export PYTHONPATH="$(pwd)"
  ```

## Run in a Screen

```bash
screen -S hsi
conda activate hsi && cd ~/projects/HSI
python train.py --models cnn_baseline,cnn_residual,cnn_dilated --reports
# o:
python train.py --models cnn_baseline --reports --limit-rows 0
# also limiting rows per subregion:
python train.py --models cnn_baseline --reports --group-by Subregion --per-group-limit 500
# or per subregion and pigment (combination of columns):
python train.py --models cnn_baseline --reports --group-by Subregion,Pigment --per-group-limit 200
# detach: Ctrl+A then D
# reattach: screen -r hsi
```

**With log in a Screen / explicit RUN_ID**
```bash
conda activate hsi
cd ~/projects/HSI
export RID=$(date +%Y%m%d-%H%M%S)
python train.py --models cnn_baseline,cnn_residual,cnn_dilated --trials 40 --epochs 60 --reports --run-id "$RID"
```

## Run with JOB SCHEDULER ()
cd ~/projects/HSI
qsub -v MODELS="cnn_baseline,cnn_residual,cnn_dilated",TRIALS=40,EPOCHS=60,N_JOBS_MODELS=3,OPTUNA_N_JOBS=4 job_scheduler.pbs


## Save to Git
cd ~/projects/HSI
git remote -v 
# if NO origin, add (SSH):
git remote add origin git@github.com:pgimenez/HSI.git
git status
git init
git add -A
git commit -m "Initial commit"
git branch -M main
git push -u origin main

## Save to Git from remote SSH
git remote add origin git@github.com:pgimenez/HSI.git
git push -u origin main

## Troubleshooting

- **No module named `hsi_lab`**  
  Add missing `__init__.py` files and/or:
  ```bash
  export PYTHONPATH="$(pwd)"
  ```

- **`matplotlib` not found / plotting fails**  
  ```bash
  pip install matplotlib
  ```
  and run with `--reports`.

- **No figures saved**  
  Use `--reports`. Images go to `outputs/runs/<RUN_ID>/figures/`.

- **Stale DataFrame after changing pipeline**  
  Cache lives in `outputs/cache/`. Delete it or call `dataframe_cached(force=True)`.

---

Use `train.py` to run one or several models, track each run under `runs/<RUN_ID>`, and find models in `saved_models/<RUN_ID>/` with detailed logs and metrics saved alongside.
