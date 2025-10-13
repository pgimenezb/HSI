import tensorflow.keras.backend as K

def strict_accuracy(y_true, y_pred):
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.cast(K.greater(y_pred, 0.5), K.floatx())
    correct = K.equal(y_true, y_pred)
    sample_ok = K.all(correct, axis=-1)
    return K.mean(K.cast(sample_ok, K.floatx()))
