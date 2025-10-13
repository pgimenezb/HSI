import numpy as np
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from .metrics import strict_accuracy

def build_cnn(input_len, num_classes, trial):
    model = Sequential()
    model.add(Conv1D(filters=trial.suggest_categorical('filters_1', [128, 256, 512]),
                     kernel_size=trial.suggest_int('kernel_1', 3, 7),
                     activation='relu',
                     input_shape=(input_len, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(trial.suggest_float('dropout_1', 0.3, 0.6)))

    model.add(Conv1D(filters=trial.suggest_categorical('filters_2', [256, 512, 1024]),
                     kernel_size=trial.suggest_int('kernel_2', 3, 7),
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(trial.suggest_float('dropout_2', 0.3, 0.6)))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(trial.suggest_categorical('dense_units', [512, 1024, 2048]), activation='relu'))
    model.add(Dropout(trial.suggest_float('dropout_3', 0.3, 0.6)))

    model.add(Dense(num_classes, activation='sigmoid'))

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=[strict_accuracy])
    return model

def tune_and_train(X_train, y_train, X_val, y_val, input_len, num_classes, trials=30, epochs=50):
    def objective(trial):
        model = build_cnn(input_len, num_classes, trial)
        hist = model.fit(X_train, y_train,
                         validation_data=(X_val, y_val),
                         epochs=epochs,
                         batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                         verbose=0,
                         callbacks=[EarlyStopping(patience=5, monitor='val_loss')])
        return hist.history['val_strict_accuracy'][-1]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    best = study.best_trial.params
    model = build_cnn(input_len, num_classes, optuna.trial.FixedTrial(best))
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=best["batch_size"], verbose=1)
    return model, study, best
