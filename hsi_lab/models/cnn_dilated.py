import os, time, numpy as np, optuna, tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv1D, SeparableConv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dense, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from .metrics import strict_accuracy

def _set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"]=str(seed); np.random.seed(seed); tf.random.set_seed(seed)

def _dilated_block(x, f, k, d):
    y = Conv1D(f, k, padding="same", dilation_rate=d, activation="relu")(x)
    y = BatchNormalization()(y)
    return y

def build_cnn_dilated(input_len, num_classes, trial):
    inp = Input(shape=(input_len,1))
    # rama ancha (baja resolución espectral)
    a = _dilated_block(inp, trial.suggest_categorical("a_f",[64,128]), trial.suggest_int("a_k",7,15), trial.suggest_categorical("a_d",[2,3,4]))
    a = MaxPooling1D(2)(a)
    # rama fina (alta resolución)
    b = SeparableConv1D(trial.suggest_categorical("b_f",[64,128,256]), trial.suggest_int("b_k",3,5), padding="same", activation="relu")(inp)
    b = BatchNormalization()(b); b = MaxPooling1D(2)(b)
    # fusion
    x = Add()([a,b]); x = ReLU()(x)
    x = SeparableConv1D(trial.suggest_categorical("head_f",[128,256,512]), 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x); x = Dropout(trial.suggest_float("drop_mid",0.2,0.5))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(trial.suggest_categorical("dense_units",[256,512,1024]), activation="relu")(x)
    x = Dropout(trial.suggest_float("drop_dense",0.3,0.6))(x)
    out = Dense(num_classes, activation="sigmoid")(x)
    m = Model(inp, out, name="HSI_CNN_Dilated")
    lr = trial.suggest_float("lr",1e-5,1e-2,log=True)
    m.compile(optimizer=Adam(lr), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.BinaryAccuracy(name="bin_acc",threshold=0.5), strict_accuracy])
    return m

def tune_and_train(X_train,y_train,X_val,y_val,input_len,num_classes,trials=30,epochs=50,
                   storage=None,study_name=None,n_jobs=1,**kw):
    seed=int(kw.get("seed",42)); _set_seeds(seed)
    patience=int(kw.get("patience",6)); batch_choices=kw.get("batch_choices",[32,64,128])
    if storage is None: storage=f"sqlite:///{os.path.abspath('outputs/optuna.db')}"
    if study_name is None: study_name=f"cnn_dilated_{time.strftime('%Y%m%d')}"
    study=optuna.create_study(direction="maximize", study_name=study_name, storage=storage,
                              load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
                              sampler=optuna.samplers.TPESampler(seed=seed))
    def objective(trial):
        _set_seeds(seed)
        model=build_cnn_dilated(input_len,num_classes,trial)
        h=model.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=epochs,
                    batch_size=trial.suggest_categorical("batch_size",batch_choices),
                    verbose=0, callbacks=[EarlyStopping("val_loss",patience=patience,restore_best_weights=True)])
        return float(h.history["val_strict_accuracy"][-1])
    study.optimize(objective, n_trials=trials, n_jobs=n_jobs)
    best=study.best_trial.params; model=build_cnn_dilated(input_len,num_classes,optuna.trial.FixedTrial(best))
    model.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=epochs,
              batch_size=best.get("batch_size",64), verbose=1,
              callbacks=[EarlyStopping("val_loss",patience=patience,restore_best_weights=True)])
    return model, study, best
