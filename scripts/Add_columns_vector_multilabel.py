# insert_zero_column_vector_multilabel.py
import os
from pathlib import Path
import h5py
import numpy as np

# === CONFIGURACIÓN ===
BASE_DIR = r"C:\Users\pgimenezbarrera\Desktop\database"
OUT_DIR = os.path.join(BASE_DIR, "vector_multilablel_changed")  # (nombre exacto que pediste)
TARGET_DATASET = "labels/vector_multilabel"
INSERT_AT = 20  # insertar entre 19 y 20 (0-index) -> nueva columna queda en índice 20

def ensure_parent_group(fout, path):
    parent = os.path.dirname(path)
    if parent and parent != "/":
        fout.require_group(parent)

def copy_group_attrs(src_group, dst_group):
    for k, v in src_group.attrs.items():
        dst_group.attrs[k] = v

def copy_dataset_default(fin, fout, name, dset):
    ensure_parent_group(fout, name)
    d = fout.create_dataset(
        name,
        shape=dset.shape,
        dtype=dset.dtype,
        compression=dset.compression,
        chunks=dset.chunks,
        shuffle=dset.shuffle,
        fletcher32=dset.fletcher32,
        fillvalue=dset.fillvalue,
    )
    # atributos
    for k, v in dset.attrs.items():
        d.attrs[k] = v
    # copia de datos directa (en bloque)
    d[...] = dset[...]

def process_target_dataset(fin, fout, name, dset):
    shape = list(dset.shape)
    if len(shape) == 0:
        raise ValueError(f"{name} no es un dataset con ejes.")
    last = shape[-1]
    if last < INSERT_AT:
        raise ValueError(
            f"{name} tiene tamaño final {last} < {INSERT_AT}; no se puede insertar en esa posición."
        )
    # Crear dataset destino con el último eje +1
    new_shape = shape[:-1] + [last + 1]

    ensure_parent_group(fout, name)
    out = fout.create_dataset(
        name,
        shape=new_shape,
        dtype=dset.dtype,
        compression=dset.compression,
        chunks=dset.chunks,
        shuffle=dset.shuffle,
        fletcher32=dset.fletcher32,
        # fillvalue no garantiza ceros, así que escribimos nosotros la columna 0 explícitamente
    )
    # Copiar atributos
    for k, v in dset.attrs.items():
        out.attrs[k] = v

    # Copiar parte antes de INSERT_AT
    if INSERT_AT > 0:
        out[..., :INSERT_AT] = dset[..., :INSERT_AT]

    # Columna insertada a 0
    out[..., INSERT_AT] = np.zeros(shape[:-1], dtype=dset.dtype)

    # Copiar parte después (desplazada +1)
    out[..., INSERT_AT + 1 :] = dset[..., INSERT_AT :]

def transform_file(src_path: Path, dst_path: Path):
    with h5py.File(src_path, "r") as fin, h5py.File(dst_path, "w") as fout:
        # Copiar atributos del grupo raíz
        copy_group_attrs(fin["/"], fout["/"])

        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                # crear grupo y copiar atributos
                if name != "/":
                    fout.require_group(name)
                    copy_group_attrs(obj, fout[name])
            elif isinstance(obj, h5py.Dataset):
                if name.replace("\\", "/") == TARGET_DATASET:
                    process_target_dataset(fin, fout, name, obj)
                else:
                    copy_dataset_default(fin, fout, name, obj)

        fin.visititems(visitor)

def main():
    base = Path(BASE_DIR)
    outdir = Path(OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(base.rglob("*.h5"))
    if not h5_files:
        print(f"No se encontraron .h5 en {base}")
        return

    print(f"Procesando {len(h5_files)} archivos .h5…\nSalida: {outdir}\n")

    for src in h5_files:
        dst = outdir / src.name
        try:
            print(f"- {src.name}: ", end="", flush=True)
            transform_file(src, dst)
            print("OK")
        except Exception as e:
            print(f"ERROR -> {e}")

if __name__ == "__main__":
    main()
