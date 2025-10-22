# check_h5_schema.py
import os
from pathlib import Path
import h5py
from collections import defaultdict

# === CONFIGURA AQUÍ TU CARPETA ===
BASE_DIR = r"C:\Users\pgimenezbarrera\Desktop\database"

# === OPCIONES ===
CHECK_DTYPE = False  # pon True si también quieres verificar el tipo de dato (float32, int, etc.)

def collect_structure(h5_path: Path):
    """
    Devuelve dos dicts:
      - datasets: {full_path: {"shape": tuple, "dtype": str}}
      - groups:   {full_path: True}
    """
    datasets = {}
    groups = {}

    with h5py.File(h5_path, "r") as f:
        # raíz cuenta como grupo
        groups["/"] = True

        def visitor(name, obj):
            full = "/" + name if not name.startswith("/") else name
            if isinstance(obj, h5py.Group):
                groups[full] = True
            elif isinstance(obj, h5py.Dataset):
                # Guardar shape y dtype como string para comparar
                datasets[full] = {
                    "shape": tuple(int(x) for x in obj.shape),
                    "dtype": str(obj.dtype)
                }
        f.visititems(visitor)

    return datasets, groups

def compare(reference, target, ref_name, tgt_name, check_dtype=False):
    """
    Compara estructuras y devuelve un dict con diferencias.
    """
    ref_ds, ref_gr = reference
    tgt_ds, tgt_gr = target

    diffs = {}

    # Datasets faltantes / extra
    ref_ds_keys = set(ref_ds.keys())
    tgt_ds_keys = set(tgt_ds.keys())
    missing_ds = sorted(ref_ds_keys - tgt_ds_keys)
    extra_ds   = sorted(tgt_ds_keys - ref_ds_keys)

    if missing_ds:
        diffs["datasets_faltantes"] = missing_ds
    if extra_ds:
        diffs["datasets_extras"] = extra_ds

    # Shape (y dtype) distintos
    shape_mismatch = []
    dtype_mismatch = []
    for k in sorted(ref_ds_keys & tgt_ds_keys):
        if ref_ds[k]["shape"] != tgt_ds[k]["shape"]:
            shape_mismatch.append({
                "dataset": k,
                "ref_shape": ref_ds[k]["shape"],
                "tgt_shape": tgt_ds[k]["shape"]
            })
        if check_dtype and ref_ds[k]["dtype"] != tgt_ds[k]["dtype"]:
            dtype_mismatch.append({
                "dataset": k,
                "ref_dtype": ref_ds[k]["dtype"],
                "tgt_dtype": tgt_ds[k]["dtype"]
            })
    if shape_mismatch:
        diffs["datasets_shape_distinta"] = shape_mismatch
    if check_dtype and dtype_mismatch:
        diffs["datasets_dtype_distinto"] = dtype_mismatch

    # Grupos faltantes / extra
    ref_gr_keys = set(ref_gr.keys())
    tgt_gr_keys = set(tgt_gr.keys())
    missing_gr = sorted(ref_gr_keys - tgt_gr_keys)
    extra_gr   = sorted(tgt_gr_keys - ref_gr_keys)
    if missing_gr:
        diffs["grupos_faltantes"] = missing_gr
    if extra_gr:
        diffs["grupos_extras"] = extra_gr

    return diffs

def main():
    base = Path(BASE_DIR)
    if not base.exists():
        print(f"[ERROR] Carpeta no encontrada: {base}")
        return

    h5_files = sorted(p for p in base.rglob("*.h5"))
    if not h5_files:
        print(f"No se encontraron .h5 en: {base}")
        return

    print(f"Analizando {len(h5_files)} archivos .h5 bajo {base}\n")

    # Tomamos el primero como referencia
    ref_file = h5_files[0]
    ref_struct = collect_structure(ref_file)
    print(f"Archivo de referencia: {ref_file}\n")

    any_diff = False
    for fpath in h5_files[1:]:
        tgt_struct = collect_structure(fpath)
        diffs = compare(ref_struct, tgt_struct, ref_file.name, fpath.name, check_dtype=CHECK_DTYPE)
        if diffs:
            any_diff = True
            print(f"== {fpath}")
            # Datasets faltantes
            if "datasets_faltantes" in diffs:
                print("   - Datasets FALTANTES vs referencia:")
                for k in diffs["datasets_faltantes"]:
                    print(f"       {k}")
            # Datasets extra
            if "datasets_extras" in diffs:
                print("   - Datasets EXTRAS vs referencia:")
                for k in diffs["datasets_extras"]:
                    print(f"       {k}")
            # Shapes distintas
            if "datasets_shape_distinta" in diffs:
                print("   - Datasets con SHAPE distinto:")
                for item in diffs["datasets_shape_distinta"]:
                    print(f"       {item['dataset']}: ref {item['ref_shape']}  !=  {item['tgt_shape']}")
            # Dtype distintos (opcional)
            if "datasets_dtype_distinto" in diffs:
                print("   - Datasets con DTYPE distinto:")
                for item in diffs["datasets_dtype_distinto"]:
                    print(f"       {item['dataset']}: ref {item['ref_dtype']}  !=  {item['tgt_dtype']}")
            # Grupos faltantes / extra
            if "grupos_faltantes" in diffs:
                print("   - Grupos FALTANTES vs referencia:")
                for k in diffs["grupos_faltantes"]:
                    print(f"       {k}")
            if "grupos_extras" in diffs:
                print("   - Grupos EXTRAS vs referencia:")
                for k in diffs["grupos_extras"]:
                    print(f"       {k}")
            print()
    if not any_diff:
        print("✅ Todos los archivos tienen exactamente la misma estructura (mismos grupos y datasets con mismas shapes).")

if __name__ == "__main__":
    main()
