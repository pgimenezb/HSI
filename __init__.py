# hsi_lab/models/__init__.py
from importlib import import_module

# Cada modelo implementa: tune_and_train(X_train, y_train, X_val, y_val, input_len, num_classes, trials, epochs, **kwargs)
# y opcionalmente: add_cli_args(parser) para argumentos específicos del modelo.
def get_model(name: str):
    name = name.lower()
    mod = import_module(f"hsi_lab.models.{name}")  # p.ej. cnn, dnn, cnn_residual, dnn_small
    return mod

# Descubrimiento opcional (si quieres listar disponibles)
def list_models():
    return ["cnn", "dnn"]  # mantén esta lista a mano o añade autodescubrimiento si te apetece
