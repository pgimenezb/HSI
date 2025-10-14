# hsi_lab/models/cnn.py
import os
import numpy as np
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense, Dropout,
    GlobalAveragePooling1D, BatchNormalization, Input
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from .metrics import strict_accuracy


def _set_seeds(seed: int = 42):
    """Fija semillas para reproducibilidad razonable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_cnn(input_len: int, num_classes: int, trial: optuna.trial.Trial) -> tf.keras.Model:
    """
    Construye una CNN 1D para espectros con hiperparámetros sugeridos por Optuna.
    input_len: nº de canales espectrales
    num_classes: longitud del vector multilabel
    """
    m = Sequential(name="HSI_CNN")
    m.add(Input(shape=(input_len, 1)))

    # Bloque 1
    m.add(Conv1D(
        filters=trial.suggest_categorical("filters_1", [128, 256, 512]),
        kernel_size=trial.suggest_int("kernel_1", 3, 7),
        activation="relu",
        padding="valid",
    ))
    m.add(MaxPooling1D(pool_size=2))
    m.add(BatchNormalization())
    m.add(Dropout(trial.suggest_float("dropout_1", 0.3, 0.6)))

    # Bloque 2
    m.add(Conv1D(
        filters=trial.suggest_categorical("filters_2", [256, 512, 1024]),
        kernel_size=trial.suggest_int("kernel_2", 3, 7),
        activation="relu",
        padding="valid",
    ))
    m.add(MaxPooling1D(pool_size=2))
    m.add(BatchNormalization())
    m.add(Dropout(trial.suggest_float("dropout_2", 0.3, 0.6)))

    # Cabeza
    m.add(GlobalAveragePooling1D())
    m.add(Dense(trial.suggest_categorical("dense_units", [512, 1024, 2048]), activation="relu"))
    m.add(Dropout(trial.suggest_float("dropout_3", 0.3, 0.6)))

    # Salida multilabel
    m.add(Dense(num_classes, activation="sigmoid"))

    # Optimizador
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    m.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.5), strict_accuracy],
    )
    return m


def tune_and_train(
    X_train, y_train, X_val, y_val,
    input_len: int, num_classes: int,
    trials: int = 30, epochs: int = 50,
    storage: str = None, study_name: str = None, n_jobs: int = 1,
    **kwargs,
):
    import time, os
    seed = int(kwargs.get("seed", 42))
    _set_seeds(seed)

    batch_choices = kwargs.get("batch_choices", [32, 64, 128])
    patience = int(kwargs.get("patience", 5))

    if storage is None:
        storage = f"sqlite:///{os.path.abspath('outputs/optuna.db')}"
    if study_name is None:
        study_name = f"cnn_{time.strftime('%Y%m%d')}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def objective(trial: optuna.trial.Trial):
        _set_seeds(seed)
        model = build_cnn(input_len, num_classes, trial)
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=trial.suggest_categorical("batch_size", batch_choices),
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)],
        )
        return float(hist.history["val_strict_accuracy"][-1])

    study.optimize(objective, n_trials=trials, n_jobs=n_jobs)

    best = study.best_trial.params
    fixed = optuna.trial.FixedTrial(best)
    model = build_cnn(input_len, num_classes, fixed)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=best.get("batch_size", 64),
        verbose=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)],
    )
    return model, study, best


