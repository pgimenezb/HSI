import os
import numpy as np
from scipy.ndimage import zoom
from spectral.io import envi

# === CONFIGURACIÓN (ajustadas a lo que pediste) ===
SRC_DIR = r"C:\Users\pgimenezbarrera\Desktop\swir"
DST_DIR = r"C:\Users\pgimenezbarrera\Desktop\swir\swir resize"

# Objetivo ENVI: Samples (X) × Lines (Y) × Bands
TARGET_SAMPLES = 107  # X
TARGET_LINES   = 247  # Y
TARGET_BANDS   = 267  # espectral

# Interpolación: espacial (bicúbica), espectral (lineal)
SPATIAL_ORDER   = 3
SPECTRAL_ORDER  = 1

os.makedirs(DST_DIR, exist_ok=True)

# ------------------ utilidades ------------------

def _read_header_text(hdr_path):
    with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def _parse_header_dict(hdr_path):
    # spectral ya parsea el header; usamos envi.open con solo header para metadata
    from spectral.io.envi import read_envi_header
    return read_envi_header(hdr_path)

def _find_data_file(hdr_path, meta):
    # 1) si el header trae 'data file' / 'data filename'
    for key in ('data file', 'data filename', 'datafile', 'file'):
        if key in meta:
            cand = meta[key].strip().strip('"').strip("'")
            # rutas relativas respecto al .hdr
            if not os.path.isabs(cand):
                cand = os.path.join(os.path.dirname(hdr_path), cand)
            if os.path.exists(cand):
                return cand
    # 2) probar mismo base con .bip y .img
    base = os.path.splitext(hdr_path)[0]
    for ext in ('.bip', '.img'):
        cand = base + ext
        if os.path.exists(cand):
            return cand
    # 3) buscar en el mismo directorio por coincidencias de nombre
    name = os.path.splitext(os.path.basename(hdr_path))[0].lower()
    folder = os.path.dirname(hdr_path)
    for fn in os.listdir(folder):
        if fn.lower().startswith(name) and os.path.splitext(fn)[1].lower() in ('.bip', '.img', '.bin'):
            return os.path.join(folder, fn)
    raise FileNotFoundError("No se encontró el binario ENVI asociado (.bip/.img).")

def resize_spatial(arr_lsb):
    """ arr_lsb: (lines, samples, bands) -> (TARGET_LINES, TARGET_SAMPLES, bands) """
    lines, samples, bands = arr_lsb.shape
    zf_y = TARGET_LINES / lines
    zf_x = TARGET_SAMPLES / samples
    out = zoom(arr_lsb, (zf_y, zf_x, 1.0), order=SPATIAL_ORDER, mode="nearest", grid_mode=True)
    # Ajuste exacto
    out = out[:TARGET_LINES, :TARGET_SAMPLES, :]
    if out.shape[0] < TARGET_LINES or out.shape[1] < TARGET_SAMPLES:
        pad_y = max(0, TARGET_LINES - out.shape[0])
        pad_x = max(0, TARGET_SAMPLES - out.shape[1])
        out = np.pad(out, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
    return out

def resize_spectral(arr_lsb):
    """ Redimensiona el eje espectral a TARGET_BANDS """
    lines, samples, bands = arr_lsb.shape
    if bands == TARGET_BANDS:
        return arr_lsb
    zf_b = TARGET_BANDS / bands
    out = zoom(arr_lsb, (1.0, 1.0, zf_b), order=SPECTRAL_ORDER, mode="nearest", grid_mode=True)
    out = out[:, :, :TARGET_BANDS]
    if out.shape[2] < TARGET_BANDS:
        pad_b = TARGET_BANDS - out.shape[2]
        out = np.pad(out, ((0,0),(0,0),(0,pad_b)), mode="edge")
    return out

def resample_list(values, out_len):
    """Reinterpolación uniforme de listas numéricas (wavelength, fwhm...)."""
    if values is None:
        return None
    try:
        arr = np.asarray([float(x) for x in values], dtype=float)
    except Exception:
        return None
    n = arr.shape[0]
    if n == out_len:
        return [str(x) for x in arr.tolist()]
    src_idx = np.linspace(0, n - 1, num=n, dtype=float)
    dst_idx = np.linspace(0, n - 1, num=out_len, dtype=float)
    newv = np.interp(dst_idx, src_idx, arr)
    return [f"{x:.6f}" for x in newv.tolist()]

def save_envi_bip(dst_base, arr_lsb, meta_in):
    """Guarda arr (lines, samples, bands) como ENVI BIP en dst_base(.hdr/.bip)."""
    lines, samples, bands = arr_lsb.shape

    # Data type ENVI según dtype numpy
    np2envi = {
        np.dtype('uint8'): 1,
        np.dtype('int16'): 2,
        np.dtype('int32'): 3,
        np.dtype('float32'): 4,
        np.dtype('float64'): 5,
        np.dtype('uint16'): 12,
        np.dtype('uint32'): 13,
    }
    dt = np2envi.get(arr_lsb.dtype, 4)

    meta_out = dict(meta_in or {})
    meta_out.update({
        'samples': samples,
        'lines': lines,
        'bands': bands,
        'interleave': 'bip',
        'data type': dt,
    })

    # Reinterpolar metadatos espectrales si existen
    for k in ('wavelength', 'fwhm'):
        if k in meta_in:
            new_list = resample_list(meta_in.get(k), bands)
            if new_list is not None:
                meta_out[k] = new_list
    if 'wavelength units' in meta_in:
        meta_out['wavelength units'] = meta_in['wavelength units']

    hdr_path = dst_base + '.hdr'
    envi.save_image(
        hdr_path,
        arr_lsb,
        dtype=arr_lsb.dtype,
        interleave='bip',
        force=True,
        ext='.bip',
        metadata=meta_out
    )

def process_file(hdr_path):
    base = os.path.splitext(os.path.basename(hdr_path))[0]
    dst_base = os.path.join(DST_DIR, base)

    # Leer metadata y encontrar binario
    meta = _parse_header_dict(hdr_path)
    data_path = _find_data_file(hdr_path, meta)

    # Abrir ENVI indicando explícitamente el binario
    img = envi.open(hdr_path, image=data_path)

    # Carga a (lines, samples, bands)
    arr = img.load()
    # convertimos a float32 para evitar artefactos en interpolación continua
    arr = np.asarray(arr, dtype=np.float32)

    # Redimensionar
    arr = resize_spatial(arr)
    arr = resize_spectral(arr)

    # Guardar salida
    save_envi_bip(dst_base, arr, meta)
    print(f"✔ Guardado: {dst_base}.hdr / .bip")

def main():
    hdr_files = [os.path.join(SRC_DIR, f) for f in os.listdir(SRC_DIR) if f.lower().endswith('.hdr')]
    if not hdr_files:
        print("No se encontraron .hdr en la carpeta de origen.")
        return
    print(f"Encontrados {len(hdr_files)} archivo(s) .hdr")
    for hdr in hdr_files:
        try:
            process_file(hdr)
        except Exception as e:
            print(f"✖ Error en {os.path.basename(hdr)}: {e}")

if __name__ == "__main__":
    main()
