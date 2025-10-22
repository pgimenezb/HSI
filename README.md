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
  └── outputs/runs/<RUN_ID>/
    ├─ optuna.db
    ├─ cnn_baseline_confusion_matrix.png
    ├─ cnn_baseline_confusion_matrix_pigments.png
    ├─ cnn_baseline_confusion_matrix_binders.png
    ├─ cnn_baseline_confusion_matrix_mixtures.png
    ├─ cnn_baseline_test_global_metrics.csv
    ├─ cnn_baseline_test_per_group_metrics.csv
    ├─ cnn_baseline_test_classification_report.csv
    ├─ cnn_baseline_test_classification_report_pigments.csv
    ├─ cnn_baseline_test_classification_report_binders.csv
    ├─ cnn_baseline_test_classification_report_mixtures.csv
    ├─ cnn_baseline_best_params.csv
    ├─ cnn_baseline_trials_summary.csv
    ├─ cnn_baseline_summary.json
    ├─ cnn_baseline.h5           
    └─ cnn_baseline.keras         

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


## Tips

- Ensure packages: `hsi_lab/`, `hsi_lab/data/`, `hsi_lab/models/`, `hsi_lab/eval/` all have `__init__.py`.
- When running from repo root this usually isn’t needed, but if imports fail:
  ```bash
  export PYTHONPATH="$(pwd)"
  ```

## Run in a Screen

```bash
screen -S hsi
conda activate hsi
cd /home/pgimenez/projects/HSI
# if only 1 model:
python -m train.py --models cnn_baseline
# if more than one:
python -m train.py --models cnn_baseline,dnn_wide --reports
# or:
python train.py --models cnn_baseline --reports --limit-rows 0
# also limiting rows per subregion:
python train.py --models cnn_baseline --group-by Subregion --per-group-limit-map "1=300,2=100,3=100,4=100"
#or:
python train.py --models cnn_baseline --group-by Subregion --per-group-limit-map "1=300,2=100" 
#  limiting rows per region:
python train.py --models cnn_baseline --group-by Region --per-group-limit-map "1=300,2=100,3=100,4=100"
# or per subregion and pigment (combination of columns):
python train.py --models cnn_baseline --group-by Subregion,Pigment 
python train.py --models cnn_baseline,dnn_wide --reports --group-by Subregion,Pigment --per-group-limit 200
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
#one model withou row limit:
qsub -v MODELS=cnn_baseline,REPORTS_FLAG= hsi_train.pbs
#one model with row limit:
qsub -v MODELS=cnn_baseline,GROUP_BY=Subregion,PER_GROUP_LIMIT=500 hsi_train.pbs
#more than one model with row limit:
qsub -v MODELS=cnn_baseline,cnn_residual,TRIALS=40,EPOCHS=60,OPTUNA_N_JOBS=4,REPORTS_FLAG=--reports hsi_train.pbs

# sceen
screen -r hsi_par
#if:
pgimenez@cyan:~$ screen -r hsi_par 
There is a screen on: 1094269.hsi_par (14/10/25 14:40:50) (Attached)
# then:
screen -dr 1094269

## Save to Git
cd ~/projects/HSI
git remote -v 
# if NO origin, add (SSH):
git remote add origin git@github.com:pgimenez/HSI.git
git add -A
git init
git add -A
git commit -m "Initial commit 1"
git branch -M main
git push -u origin main

## Save to Git from remote SSH
git remote add origin git@github.com:pgimenez/HSI.git
git push -u origin main

# if other processes
cd ~/projects/HSI
ps aux | grep '[g]it'
rm -f .git/index.lock
ssh -T git@github.com
git remote set-url origin git@github.com:pgimenezb/HSI.git
git remote -v
git remote add origin git@github.com:pgimenez/HSI.git
git init
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
