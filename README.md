# HSI

Estructura modular:
- `hsi_lab/config.py`: variables editables del experimento.
- `hsi_lab/data/processor.py`: carga HDF5, dataframe, visualización.
- `hsi_lab/models/metrics.py`: métricas y matrices de confusión.
- `hsi_lab/models/cnn.py`: definición CNN y tuning con Optuna.
- `train.py`: orquestador (carga datos → entrena → evalúa → guarda artefactos).

## Ejecutar
```bash
conda activate hsi
python train.py --epochs 50 --trials 30


## 2.4 `hsi_lab/__init__.py`
```bash
printf "" > hsi_lab/__init__.py
