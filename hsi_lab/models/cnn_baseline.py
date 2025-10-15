import os, time, numpy as np, optuna, tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GlobalAveragePooling1D, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from hsi_lab.data.config import variables   # 👈 importa las configuraciones globales
from .metrics import strict_accuracy

def _set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_cnn(input_len, num_classes, trial):
    m = Sequential(name="HSI_CNN_Baseline")
    m.add(Input(shape=(input_len, 1)))
    m.add(Conv1D(filters=trial.suggest_categorical("filters_1",[128,256,512]),
                 kernel_size=trial.suggest_int("kernel_1",3,7), activation="relu"))
    m.add(MaxPooling1D(2)); m.add(BatchNormalization()); 
    m.add(Dropout(trial.suggest_float("dropout_1",0.3,0.6)))

    m.add(Conv1D(filters=trial.suggest_categorical("filters_2",[256,512,1024]),
                 kernel_size=trial.suggest_int("kernel_2",3,7), activation="relu"))
    m.add(MaxPooling1D(2)); m.add(BatchNormalization()); 
    m.add(Dropout(trial.suggest_float("dropout_2",0.3,0.6)))

    m.add(GlobalAveragePooling1D())
    m.add(Dense(trial.suggest_categorical("dense_units",[512,1024,2048]), activation="relu"))
    m.add(Dropout(trial.suggest_float("dropout_3",0.3,0.6)))
    m.add(Dense(num_classes, activation="sigmoid"))

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    m.compile(
        optimizer=Adam(lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.5), strict_accuracy]
    )
    return m

def tune_and_train(X_train, y_train, X_val, y_val, input_len, num_classes,
                   trials=None, epochs=None, storage=None, study_name=None, n_jobs=1, **kw):
    """Entrena el modelo CNN con Optuna, usando config.py si no se especifican trials/epochs."""
    seed = int(kw.get("seed", variables.get("seed", 42)))
    _set_seeds(seed)

    # Si no vienen por parámetro, usa config.py
    trials = trials or int(variables.get("trials", 50))
    epochs = epochs or int(variables.get("epochs", 50))

    batch_choices = kw.get("batch_choices", [32, 64, 128])
    patience = int(kw.get("patience", 5))

    y_train = y_train.astype("float32")
    y_val   = y_val.astype("float32")

    if storage is None:
        storage = f"sqlite:///{os.path.abspath('outputs/optuna.db')}"
    if study_name is None:
        study_name = f"cnn_baseline_{time.strftime('%Y%m%d')}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def objective(trial):
        _set_seeds(seed)
        model = build_cnn(input_len, num_classes, trial)
        h = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=trial.suggest_categorical("batch_size", batch_choices),
            verbose=0,
            callbacks=[EarlyStopping("val_loss", patience=patience, restore_best_weights=True)]
        )
        return float(np.max(h.history["val_bin_acc"]))

    study.optimize(objective, n_trials=trials, n_jobs=n_jobs)

    best = study.best_trial.params
    model = build_cnn(input_len, num_classes, optuna.trial.FixedTrial(best))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=best.get("batch_size", 64),
        verbose=1,
        callbacks=[EarlyStopping("val_loss", patience=patience, restore_best_weights=True)]
    )
    return model, study, best
