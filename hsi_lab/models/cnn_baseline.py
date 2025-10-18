import os, time, numpy as np, optuna, tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dense, Dropout,
                                     GlobalAveragePooling1D, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from hsi_lab.data.config import variables

def _set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_cnn_multihead(input_len, n_pigments, n_mix, trial):
    x_in = Input(shape=(input_len, 1), name="spec")

    x = Conv1D(
        filters=trial.suggest_categorical("filters_1", [128, 256, 512]),
        kernel_size=trial.suggest_int("kernel_1", 3, 7),
        activation="relu"
    )(x_in)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    x = Dropout(trial.suggest_float("dropout_1", 0.3, 0.6))(x)

    x = Conv1D(
        filters=trial.suggest_categorical("filters_2", [256, 512, 1024]),
        kernel_size=trial.suggest_int("kernel_2", 3, 7),
        activation="relu"
    )(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    x = Dropout(trial.suggest_float("dropout_2", 0.3, 0.6))(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(trial.suggest_categorical("dense_units", [512, 1024, 2048]), activation="relu")(x)
    x = Dropout(trial.suggest_float("dropout_3", 0.3, 0.6))(x)

    out_pig = Dense(n_pigments, activation="softmax", name="pig")(x)
    out_mix = Dense(n_mix,      activation="softmax", name="mix")(x)

    # LR robusto para Trial y FixedTrial
    if hasattr(trial, "suggest_float"):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    else:
        lr = float(getattr(trial, "params", {}).get("lr", 1e-3))

    model = Model(inputs=x_in, outputs=[out_pig, out_mix], name="HSI_CNN_Multihead")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={"pig": "categorical_crossentropy", "mix": "categorical_crossentropy"},
        metrics={
            "pig": [tf.keras.metrics.CategoricalAccuracy(name="acc")],
            "mix": [tf.keras.metrics.CategoricalAccuracy(name="acc")]
        }
    )
    return model

def tune_and_train(X_train, y_train, X_val, y_val, input_len, num_classes,
                   trials=None, epochs=None, storage=None, study_name=None, n_jobs=1, **kw):
    """y_* viene concatenado (N, n_pig + 4). Aquí lo separamos en dos cabezas."""
    seed = int(kw.get("seed", variables.get("seed", 42)))
    _set_seeds(seed)

    n_pig = int(variables.get("num_files"))
    n_mix = 4

    trials = int(trials or variables.get("trials", 10))
    epochs = int(epochs or variables.get("epochs", 10))
    n_jobs = int(n_jobs or variables.get("optuna_n_jobs", 1))

    if storage is None:
        storage = f"sqlite:///{os.path.abspath('outputs/optuna.db')}"
    if study_name is None:
        study_name = f"cnn_baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    if str(storage).startswith("sqlite"):
        n_jobs = 1  # SQLite no soporta bien n_jobs>1 concurrente

    def split_y(y):
        y = y.astype("float32")
        return {"pig": y[:, :n_pig], "mix": y[:, n_pig:n_pig + n_mix]}

    ytr, yva = split_y(y_train), split_y(y_val)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=False,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def objective(trial):
        _set_seeds(seed)
        model = build_cnn_multihead(input_len, n_pig, n_mix, trial)
        h = model.fit(
            X_train,
            ytr,
            validation_data=(X_val, yva),
            epochs=epochs,
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            verbose=0,
            callbacks= EarlyStopping("val_loss", patience=5, restore_best_weights=True),
        )       
        # objetivo: media de accuracies de las dos cabezas
        return float(0.5 * (np.max(h.history["val_pig_acc"]) + np.max(h.history["val_mix_acc"])))

    study.optimize(objective, n_trials=trials, n_jobs=n_jobs)

    best = study.best_trial.params
    if "lr" not in best:
        best["lr"] = 1e-3

    model = build_cnn_multihead(input_len, n_pig, n_mix, optuna.trial.FixedTrial(best))
    model.fit(
        X_train,
        ytr,
        validation_data=(X_val, yva),
        epochs=epochs,
        batch_size=best.get("batch_size", 64),
        verbose=1,
        callbacks=[EarlyStopping("val_loss", patience=int(kw.get("patience", 5)), restore_best_weights=True)],
    )
    return model, study, best
