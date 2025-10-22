from pathlib import Path
import h5py
import numpy as np
import re

# ========= RUTAS (FIJAS, SIN ARGUMENTOS) =========
BASE = Path(r"C:\Users\pgimenezbarrera\Desktop")
DIR_VIS  = BASE / "vis/vis resize"
DIR_SWIR = BASE / "swir/swir resize"

# ========= UTILIDADES =========
def read_swir_from_h5(path_h5: Path):
    """
    Lee los datos SWIR desde un .h5 de SWIR.
    Acepta estructuras:
      - /swir/data (+ /swir/wavelength)
      - /data o /hyperspectral_data en raíz (por compatibilidad)
    Devuelve (array_data, array_wavelength_o_None)
    """
    with h5py.File(path_h5, "r") as f:
        # 1) ruta típica generada por nuestro conversor
        if "swir" in f and isinstance(f["swir"], h5py.Group) and "data" in f["swir"]:
            arr = f["swir/data"][...]
            wl = f["swir/wavelength"][... ] if "swir/wavelength" in f["swir"] else None
            return arr, wl

        # 2) compatibilidad: data/hyperspectral_data en otro layout
        cand_names = ("data", "hyperspectral_data")
        for name in cand_names:
            if name in f and isinstance(f[name], h5py.Dataset):
                arr = f[name][...]
                # intentar encontrar wavelengths en raíz
                wl = None
                for wname in ("wavelength", "wavelengths"):
                    if wname in f:
                        wl = f[wname][...]
                        break
                return arr, wl

        # 3) búsqueda por nombre del dataset ('data' o 'hyperspectral_data') en cualquier nivel
        found = None
        def visitor(name, obj):
            nonlocal found
            if isinstance(obj, h5py.Dataset) and name.split("/")[-1] in {"data", "hyperspectral_data"}:
                found = obj
        f.visititems(visitor)
        if found is None:
            raise KeyError(f"{path_h5.name} no contiene dataset de datos (data/hyperspectral_data).")
        arr = found[...]
        # wavelengths: buscar un dataset llamado wavelength/wavelengths
        wl_found = None
        def visitor_wl(name, obj):
            nonlocal wl_found
            if isinstance(obj, h5py.Dataset) and name.split("/")[-1] in {"wavelength", "wavelengths"}:
                wl_found = obj
        f.visititems(visitor_wl)
        wl = wl_found[...] if wl_found is not None else None
        return arr, wl


def write_swir_into_vis_h5(vis_h5: Path, swir_data, swir_wl):
    """
    Escribe/actualiza el grupo 'swir' dentro del .h5 de VIS.
    Si ya existe, lo reemplaza limpio.
    """
    with h5py.File(vis_h5, "a") as f:
        g = f.require_group("swir")
        if "data" in g:
            del g["data"]
        g.create_dataset("data", data=swir_data)
        if swir_wl is not None:
            if "wavelength" in g:
                del g["wavelength"]
            g.create_dataset("wavelength", data=swir_wl)


def find_matching_vis_h5(stem: str) -> Path | None:
    """
    Busca en VIS un .h5 cuyo stem coincida EXACTO con 'stem'.
    Si no hay exacto en la raíz, busca en subcarpetas.
    Como plan B, intenta emparejar por el primer número presente en el nombre.
    """
    exact = DIR_VIS / f"{stem}.h5"
    if exact.exists():
        return exact

    # buscar en subcarpetas
    for p in DIR_VIS.rglob("*.h5"):
        if p.stem == stem:
            return p

    # plan B: emparejar por número (e.g., "06_*" ↔️ contiene 06)
    m = re.search(r"(\d+)", stem)
    if not m:
        return None
    num = m.group(1)
    for p in DIR_VIS.rglob("*.h5"):
        if re.search(rf"\b{num}\b", p.stem):
            return p
    return None

# ========= PIPELINE PRINCIPAL =========
def main():
    if not DIR_VIS.exists() or not DIR_SWIR.exists():
        raise SystemExit(f"[ERROR] No existen VIS ({DIR_VIS}) o SWIR ({DIR_SWIR}).")

    # Recorre todos los .h5 en SWIR (subcarpetas incluidas)
    swir_h5s = sorted(DIR_SWIR.rglob("*.h5"))
    if not swir_h5s:
        print("[WARN] No hay .h5 en SWIR. (¿Convertiste antes .hdr/.bip a .h5?)")
        return

    print(f"[INFO] SWIR .h5 detectados: {len(swir_h5s)}")
    merged = 0
    missing = 0
    failed = 0

    for swir_h5 in swir_h5s:
        stem = swir_h5.stem
        vis_h5 = find_matching_vis_h5(stem)
        if vis_h5 is None:
            print(f"[WARN] No encontré VIS .h5 para '{stem}'.")
            missing += 1
            continue

        try:
            swir_data, swir_wl = read_swir_from_h5(swir_h5)
            write_swir_into_vis_h5(vis_h5, swir_data, swir_wl)
            print(f"[OK] Añadido SWIR → {vis_h5}")
            merged += 1
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
            failed += 1

    print(f"\n[RESUMEN] Uniones OK: {merged} | Sin VIS: {missing} | Errores: {failed}")

if __name__ == "__main__":
    main()
