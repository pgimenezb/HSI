#!/bin/bash
# =================== PBS RESOURCES ====================
# Ajusta estos recursos a tu cola/nodo
#PBS -N hsi_train
#PBS -l nodes=1:ppn=4
#PBS -l mem=16gb
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -V
# =====================================================

set -euo pipefail

########################
# USER CONFIG (edit)
########################
# Conda
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-hsi}"            # cambia si tu env se llama distinto

# Modelos y parámetros por defecto (puedes sobreescribir con qsub -v VAR=...)
MODELS="${MODELS:-cnn_baseline}"         # ej: cnn_baseline,cnn_residual,cnn_dilated
TRIALS="${TRIALS:-50}"
EPOCHS="${EPOCHS:-50}"
N_JOBS_MODELS="${N_JOBS_MODELS:-1}"      # paralelismo entre modelos
OPTUNA_N_JOBS="${OPTUNA_N_JOBS:-4}"      # paralelismo interno de Optuna
REPORTS_FLAG="${REPORTS_FLAG:---reports}"# solo útil si MODELS contiene >1

# Sampling opcional por grupo (si no quieres, deja vacío GROUP_BY o PER_GROUP_LIMIT)
GROUP_BY="${GROUP_BY:-}"                  # ej: "Subregion" o "Subregion,Pigment"
PER_GROUP_LIMIT="${PER_GROUP_LIMIT:-}"    # ej: 500

########################
# ENV / MODULES
########################
# Activa conda
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  . "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || true
fi

# Evita problemas GUI de matplotlib
export MPLBACKEND=Agg
# (opcional) controla threads si usas solo CPU
export OMP_NUM_THREADS="${PBS_NUM_PPN:-4}"
export MKL_NUM_THREADS="${PBS_NUM_PPN:-4}"
export TF_NUM_INTRAOP_THREADS="${PBS_NUM_PPN:-4}"
export TF_NUM_INTEROP_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export TF_CPP_MIN_LOG_LEVEL=2   # menos ruido de TF

########################
# PATHS
########################
WORKDIR="${PBS_O_WORKDIR:-$PWD}"                 # repo desde el que haces qsub
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)_${PBS_JOBID:-local}}"

# scratch local del nodo
SCRATCHDIR="/scratch/$USER/${PBS_JOBID:-hsi_local}"
TMPDIR="$SCRATCHDIR/tmp"
mkdir -p "$SCRATCHDIR" "$TMPDIR"

echo "Host:        $(hostname)"
echo "WORKDIR:     $WORKDIR"
echo "SCRATCHDIR:  $SCRATCHDIR"
echo "RUN_ID:      $RUN_ID"
date

########################
# SYNC CODE -> SCRATCH
########################
# Copia el repo (sin .git ni outputs previos) al scratch
mkdir -p "$SCRATCHDIR/HSI"
rsync -a \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude ".ipynb_checkpoints" \
  --exclude "outputs" \
  "$WORKDIR/." "$SCRATCHDIR/HSI/"

cd "$SCRATCHDIR/HSI"

# Asegura que Python vea el paquete hsi_lab desde aquí
export PYTHONPATH="$PWD"

########################
# COMANDO DE EJECUCIÓN
########################
# Construye flags opcionales de sampling
GROUP_FLAGS=""
if [ -n "$GROUP_BY" ]; then
  GROUP_FLAGS+=" --group-by \"$GROUP_BY\""
fi
if [ -n "$PER_GROUP_LIMIT" ]; then
  GROUP_FLAGS+=" --per-group-limit $PER_GROUP_LIMIT"
fi

# train.py guarda TODO en outputs/runs/<RUN_ID>/ según tu código actual
PYTHON_CMD="python -u train.py \
  --models ${MODELS} \
  --trials ${TRIALS} \
  --epochs ${EPOCHS} \
  --n-jobs-models ${N_JOBS_MODELS} \
  --optuna-n-jobs ${OPTUNA_N_JOBS} \
  --run-id ${RUN_ID} \
  ${REPORTS_FLAG} \
  ${GROUP_FLAGS}
"

# Prepara carpeta de logs del job dentro del run
RUN_OUT_DIR="$SCRATCHDIR/HSI/outputs/runs/${RUN_ID}"
mkdir -p "$RUN_OUT_DIR"
JOB_LOG_SCRATCH="$RUN_OUT_DIR/job_${RUN_ID}.log"

echo "====================================================" | tee -a "$JOB_LOG_SCRATCH"
echo "Running: $PYTHON_CMD" | tee -a "$JOB_LOG_SCRATCH"
echo "====================================================" | tee -a "$JOB_LOG_SCRATCH"

# Ejecuta en scratch
# shellcheck disable=SC2086
eval $PYTHON_CMD 2>&1 | tee -a "$JOB_LOG_SCRATCH"

########################
# SYNC OUTPUTS -> WORK
########################
# Sincroniza outputs del run de vuelta al repo
mkdir -p "$WORKDIR/outputs/runs"
rsync -a "$RUN_OUT_DIR/" "$WORKDIR/outputs/runs/${RUN_ID}/"

echo "Archivos sincronizados a:"
echo " - $WORKDIR/outputs/runs/${RUN_ID}/"

########################
# LIMPIEZA
########################
echo "Contenido de scratch:"
ls -al "$SCRATCHDIR" || true

# Limpia scratch
rm -rf "$SCRATCHDIR"

date
echo "Job done."
