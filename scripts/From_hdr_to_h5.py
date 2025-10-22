from pathlib import Path
import numpy as np
import h5py
from spectral.io import envi

# Carpeta SWIR donde están los .hdr/.bip
SWIR_DIR = Path(r"C:\Users\pgimenezbarrera\Desktop\swir\swir resize")

# --- utilidades ---
def find_binary_for_hdr(hdr: Path) -> Path | None:
    """Intenta deducir el binario a partir del .hdr (campo 'data file' o extensiones comunes)."""
    try:
        meta = envi.read_envi_header(str(hdr))
    except Exception:
        meta = None

    # 1) Si el header declara 'data file'
    if meta:
        df = meta.get('data file') or meta.get('datafile')
        if df:
            # quitar comillas si vienen
            df = str(df).strip().strip('"').strip("'")
            cand = hdr.parent / df
            if cand.exists():
                return cand

    # 2) Si no hay 'data file', probar mismo stem con extensiones típicas
    exts = [".bip", ".bil", ".bsq", ".img"]
    for ext in exts + [e.upper() for e in exts]:
        cand = hdr.with_suffix(ext)
        if cand.exists():
            return cand

    return None


def convert_envi_to_h5_from_hdr(hdr: Path, dst_h5: Path) -> bool:
    """Abre ENVI usando hdr + binario explícito y escribe swir/data (+wavelength si hay)."""
    bin_path = find_binary_for_hdr(hdr)
    if bin_path is None:
        print(f"[SKIP] No encuentro binario para: {hdr}")
        return False

    try:
        img = envi.open(str(hdr), str(bin_path))  # pasar explícitamente el binario
        cube = img.load().astype("float32")
        # wavelengths si existen
        try:
            wvl = np.array(img.bands.centers, dtype="float32")
        except Exception:
            wvl = None

        dst_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst_h5, "w") as f:
            g = f.require_group("swir") # CAMBIAR POR EL QUE TOCA
            g.create_dataset("data", data=cube)
            if wvl is not None and wvl.size > 0:
                g.create_dataset("wavelength", data=wvl)
        print(f"[WRITE] {dst_h5}")
        return True
    except Exception as e:
        print(f"[ERROR ENVI→H5] {hdr.name}: {e}")
        return False


def main():
    if not SWIR_DIR.exists():
        raise SystemExit(f"[ERROR] No existe la carpeta SWIR: {SWIR_DIR}")

    hdrs = sorted(SWIR_DIR.rglob("*.hdr"))
    if not hdrs:
        print("[INFO] No hay .hdr en SWIR.")
        return

    ok = 0
    for hdr in hdrs:
        dst = hdr.with_suffix(".h5")
        if convert_envi_to_h5_from_hdr(hdr, dst):
            ok += 1
    print(f"[RESUMEN] Convertidos: {ok} de {len(hdrs)}")

if __name__ == "__main__":
    main()
