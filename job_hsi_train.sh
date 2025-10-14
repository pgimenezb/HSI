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
# Conda en Obelix
CONDA_HOME="$HOME/miniconda3"
CONDA_ENV="vsserver"            # cambia si tu env se llama distinto (p.ej. "hsi")

# Modelos y parámetros por defecto (puedes sobreescribir con qsub -v)
MODELS="${MODELS:-cnn_baseline,cnn_residual,cnn_dilated}"
TRIALS="${TRIALS:-40}"
EPOCHS="${EPOCHS:-60}"
N_JOBS_MODELS="${N_JOBS_MODELS:-1}"      # paralelismo entre modelos
OPTUNA_N_JOBS="${OPTUNA_N_JOBS:-4}"      # paralelismo interno de Optuna
REPORTS_FLAG="--reports"                 # quítalo si no quieres figuras

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
# (opcional) reduce oversubscription si usas sólo CPU
export OMP_NUM_THREADS="${PBS_NUM_PPN:-4}"
export MKL_NUM_THREADS="${PBS_NUM_PPN:-4}"
export TF_NUM_INTRAOP_THREADS="${PBS_NUM_PPN:-4}"
export TF_NUM_INTEROP_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

########################
# PATHS
########################
WORKDIR="${PBS_O_WORKDIR:-$PWD}"            # donde hiciste qsub (tu repo)
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
# Copiamos el código (sin .git, sin outputs previos) a scratch
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
# train.py ya crea outputs/runs/<RUN_ID>/ y saved_models/<RUN_ID> automáticamente
PYTHON_CMD="python -u train.py \
  --models ${MODELS} \
  --trials ${TRIALS} \
  --epochs ${EPOCHS} \
  --n-jobs-models ${N_JOBS_MODELS} \
  --optuna-n-jobs ${OPTUNA_N_JOBS} \
  --run-id ${RUN_ID} \
  ${REPORTS_FLAG}
"

# Logs del job (además de los logs por modelo que escribe train.py)
JOB_LOG_SCRATCH="$SCRATCHDIR/HSI/outputs/runs/${RUN_ID}/logs/job_${RUN_ID}.log"
mkdir -p "$(dirname "$JOB_LOG_SCRATCH")"

echo "====================================================" | tee -a "$JOB_LOG_SCRATCH"
echo "Running: $PYTHON_CMD" | tee -a "$JOB_LOG_SCRATCH"
echo "====================================================" | tee -a "$JOB_LOG_SCRATCH"

# Ejecuta en scratch
eval "$PYTHON_CMD" 2>&1 | tee -a "$JOB_LOG_SCRATCH"

########################
# SYNC OUTPUTS -> WORK
########################
# Sincroniza todo lo generado de vuelta a tu repo
mkdir -p "$WORKDIR/outputs/runs" "$WORKDIR/saved_models"

# outputs de la ejecución
rsync -a "$SCRATCHDIR/HSI/outputs/runs/${RUN_ID}/" "$WORKDIR/outputs/runs/${RUN_ID}/"

# modelos pesados (si existen)
if [ -d "$SCRATCHDIR/HSI/saved_models/${RUN_ID}" ]; then
  rsync -a "$SCRATCHDIR/HSI/saved_models/${RUN_ID}/" "$WORKDIR/saved_models/${RUN_ID}/"
fi

echo "Archivos sincronizados a:"
echo " - $WORKDIR/outputs/runs/${RUN_ID}/"
echo " - $WORKDIR/saved_models/${RUN_ID}/ (si aplica)"

########################
# LIMPIEZA
########################
echo "Contenido de scratch:"
ls -al "$SCRATCHDIR" || true

# Limpia scratch
rm -rf "$SCRATCHDIR"

date
echo "Job done."
