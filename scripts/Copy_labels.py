from pathlib import Path
import h5py

# === RUTAS FIJAS ===
CARPETA = Path(r"C:\Users\pgimenezbarrera\Desktop\Pigments")
SRC = CARPETA / "01_Ultramarin_GA_45010.h5"      # de aquí leemos /labels
DST = CARPETA / "19_Carmin_naccarat_GA_42100.h5"        # aquí sobrescribimos /labels

# === UTILIDADES HDF5 ===
def copy_attrs(src_obj, dst_obj):
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v

def copy_dataset(src_dset, dst_group, name):
    # crea dataset preservando dtype/compresión/chunks
    if name in dst_group:
        del dst_group[name]
    dst = dst_group.create_dataset(
        name,
        shape=src_dset.shape,
        dtype=src_dset.dtype,
        compression=src_dset.compression,
        compression_opts=src_dset.compression_opts,
        chunks=src_dset.chunks,
        shuffle=src_dset.shuffle,
        fletcher32=src_dset.fletcher32,
    )
    dst[...] = src_dset[...]
    copy_attrs(src_dset, dst)

def copy_group_recursive(src_group, dst_parent, dst_name):
    # borra el destino si existe
    if dst_name in dst_parent:
        del dst_parent[dst_name]
    dst_group = dst_parent.create_group(dst_name)
    copy_attrs(src_group, dst_group)

    def _recur(sg, dg):
        for key, item in sg.items():
            if isinstance(item, h5py.Dataset):
                copy_dataset(item, dg, key)
            elif isinstance(item, h5py.Group):
                ng = dg.create_group(key)
                copy_attrs(item, ng)
                _recur(item, ng)
            else:
                # otros tipos (links, etc.) no tratados
                pass

    _recur(src_group, dst_group)

def main():
    if not SRC.exists():
        raise SystemExit(f"[ERROR] No existe origen: {SRC}")
    if not DST.exists():
        raise SystemExit(f"[ERROR] No existe destino: {DST}")

    with h5py.File(SRC, "r") as fsrc, h5py.File(DST, "a") as fdst:
        if "labels" not in fsrc or not isinstance(fsrc["labels"], h5py.Group):
            raise SystemExit(f"[ERROR] El origen no contiene grupo '/labels': {SRC}")
        src_labels = fsrc["labels"]

        # escribir en la raíz del destino como grupo 'labels'
        copy_group_recursive(src_labels, fdst["/"], "labels")
        print(f"[OK] Copiado '/labels' de\n  {SRC}\na\n  {DST}")

if __name__ == "__main__":
    main()
