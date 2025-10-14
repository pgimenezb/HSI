import os, time, numpy as np, optuna, tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from .metrics import strict_accuracy

def _set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"]=str(seed); np.random.seed(seed); tf.random.set_seed(seed)

def build_dnn_wide(input_len, num_classes, trial):
    m = Sequential(name="HSI_DNN_Wide")
    m.add(Input(shape=(input_len,1))); m.add(Flatten())
    width = trial.suggest_categorical("width",[512,1024,2048])
    depth = trial.suggest_categorical("depth",[2,3,4])
    drop = trial.suggest_float("drop",0.4,0.7)
    for _ in range(depth):
        m.add(Dense(width, activation="relu")); m.add(BatchNormalization()); m.add(Dropout(drop))
    m.add(Dense(num_classes, activation="sigmoid"))
    lr = trial.suggest_float("lr",1e-5,5e-3,log=True)
    m.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=[strict_accuracy])
    return m

def tune_and_train(X_train,y_train,X_val,y_val,input_len,num_classes,trials=20,epochs=60,
                   storage=None,study_name=None,n_jobs=1,**kw):
    seed=int(kw.get("seed",42)); _set_seeds(seed)
    patience=int(kw.get("patience",6)); batch_choices=kw.get("batch_choices",[32,64,128])
    if storage is None: storage=f"sqlite:///{os.path.abspath('outputs/optuna.db')}"
    if study_name is None: study_name=f"dnn_wide_{time.strftime('%Y%m%d')}"
    study=optuna.create_study(direction="maximize", study_name=study_name, storage=storage,
                              load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
                              sampler=optuna.samplers.TPESampler(seed=seed))
    def objective(trial):
        _set_seeds(seed)
        model=build_dnn_wide(input_len,num_classes,trial)
        h=model.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=epochs,
                    batch_size=trial.suggest_categorical("batch_size",batch_choices),
                    verbose=0, callbacks=[EarlyStopping("val_loss",patience=patience,restore_best_weights=True)])
        return float(h.history["val_strict_accuracy"][-1])
    study.optimize(objective, n_trials=trials, n_jobs=n_jobs)
    best=study.best_trial.params; model=build_dnn_wide(input_len,num_classes,optuna.trial.FixedTrial(best))
    model.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=epochs,
              batch_size=best.get("batch_size",64), verbose=1,
              callbacks=[EarlyStopping("val_loss",patience=patience,restore_best_weights=True)])
    return model, study, best
