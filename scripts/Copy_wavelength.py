# copy_wavelength_between_h5.py
from pathlib import Path
import h5py

# === CONFIGURA AQUÍ ===
BASE_DIR = r"C:\Users\pgimenezbarrera\Desktop\database"
SRC_FILE = "01_Ultramarin_GA_45010.h5"      # <- pon aquí el nombre del archivo origen
DST_FILE = "19_Carmin_naccarat_GA_42100.h5"     # <- pon aquí el nombre del archivo destino

# Si quieres que copie a TODOS los .h5 de la carpeta (excepto el origen), pon True:
COPY_TO_ALL = False

# === NO TOCAR DESDE AQUÍ ===
CANDIDATE_SRC_PATHS = ["/swir/wavelength", "/swir/wavelenght"]  # admite ambas grafías
TARGET_PATH = "/swir/wavelength"  # siempre creamos/borramos este nombre en destino

def load_source_dataset(src_path: Path):
    with h5py.File(src_path, "r") as f:
        for p in CANDIDATE_SRC_PATHS:
            if p in f:
                ds = f[p]
                # Capturamos datos y metadatos relevantes
                data = ds[()]  # copia a memoria
                meta = {
                    "dtype": ds.dtype,
                    "shape": ds.shape,
                    "chunks": ds.chunks,
                    "compression": ds.compression,
                    "compression_opts": ds.compression_opts,
                    "shuffle": getattr(ds, "shuffle", False),
                    "fletcher32": getattr(ds, "fletcher32", False),
                    "attrs": {k: ds.attrs[k] for k in ds.attrs.keys()},
                }
                return data, meta, p
    raise FileNotFoundError(
        f"No se encontró ningún dataset 'wavelength/wavelenght' en {src_path}"
    )

def ensure_group(f: h5py.File, group_path: str):
    if group_path == "/" or group_path == "":
        return
    parts = [p for p in group_path.split("/") if p]
    cur = ""
    for p in parts:
        cur += f"/{p}"
        if cur not in f:
            f.create_group(cur)

def write_target_dataset(dst_path: Path, data, meta):
    with h5py.File(dst_path, "r+") as f:
        # aseguramos /swir
        ensure_group(f, "/swir")

        # Si existe, lo borramos para sobrescribir
        if TARGET_PATH in f:
            del f[TARGET_PATH]

        # Creamos el dataset intentando mantener props
        dset = f.create_dataset(
            TARGET_PATH,
            data=data,
            dtype=meta["dtype"],
            chunks=meta["chunks"],
            compression=meta["compression"],
            compression_opts=meta["compression_opts"],
            shuffle=meta["shuffle"],
            fletcher32=meta["fletcher32"],
        )
        # Copiamos atributos
        for k, v in meta["attrs"].items():
            dset.attrs[k] = v

def main():
    base = Path(BASE_DIR)
    src_path = base / SRC_FILE

    if not src_path.exists():
        print(f"[ERROR] Origen no encontrado: {src_path}")
        return

    data, meta, found_path = load_source_dataset(src_path)
    print(f"Origen: {src_path}")
    print(f"Dataset encontrado: {found_path}  (shape={meta['shape']}, dtype={meta['dtype']})")

    if COPY_TO_ALL:
        # Copiar a todos los .h5 excepto el origen
        targets = sorted(p for p in base.glob("*.h5") if p.name != src_path.name)
        if not targets:
            print("No hay archivos destino en la carpeta.")
            return
        for dst in targets:
            try:
                write_target_dataset(dst, data, meta)
                print(f"   Copiado a: {dst.name} -> {TARGET_PATH}")
            except Exception as e:
                print(f"   [ERROR] {dst.name}: {e}")
    else:
        dst_path = base / DST_FILE
        if not dst_path.exists():
            print(f"[ERROR] Destino no encontrado: {dst_path}")
            return
        try:
            write_target_dataset(dst_path, data, meta)
            print(f"Copiado a {dst_path.name} -> {TARGET_PATH}")
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
