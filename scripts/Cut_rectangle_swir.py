# -*- coding: utf-8 -*-
# Viewer + recorte ROI para cubos ENVI (SWIR) con pseudo-RGB fijo:
# R=1500 nm, G=1302 nm, B=1104 nm. Sin 'spectral'.

import os, re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

# ========= CONFIG =========
IMG_PATH = r"C:\Users\pgimenezbarrera\Desktop\Patrícia GB\Codes - mock ups\VIS & SWIR data - mock ups (Gum Arabic)\SWIR-gumarabic-1_6_2025-07-Corrected.bip"
HDR_PATH = os.path.splitext(IMG_PATH)[0] + ".hdr"

# Rectángulo inicial (admite negativos)
INIT_X, INIT_Y = 104.0, 50.0
INIT_W, INIT_H = -107.0, 247.0

# Carpeta de salida
OUT_DIR = r"C:\Users\pgimenezbarrera\Desktop\Patrícia GB\Codes - mock ups\VIS & SWIR data - mock ups (Gum Arabic)\SWIR"

# Paso de movimiento con flechas (px)
STEP = 10

# Pseudo-RGB SWIR (en nanómetros)
SWIR_RGB_NM = (1500.0, 1302.0, 1104.0)   # (R, G, B)
# ==========================

# Desactivar barra y atajos que mueven la imagen
mpl.rcParams['toolbar'] = 'none'
for k in ['keymap.back','keymap.forward','keymap.pan','keymap.zoom','keymap.fullscreen',
          'keymap.grid','keymap.home','keymap.save']:
    if k in mpl.rcParams:
        mpl.rcParams[k] = []

# ---- Lectura ENVI (.hdr/.dat|.img|.bip|.bil) ----
DTYPE_MAP = {
    1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64,
    12: np.uint16, 13: np.uint32, 14: np.int64, 15: np.uint64
}

def parse_envi_hdr(hdr_path):
    if not os.path.exists(hdr_path):
        raise FileNotFoundError("No se encontró el header ENVI: " + hdr_path)
    txt = open(hdr_path, 'r', encoding='latin-1').read()

    def val(key, default=None):
        m = re.search(r'%s\s*=\s*(\{[^}]*\}|[^\r\n]+)' % key, txt, re.IGNORECASE)
        if not m:
            return default
        v = m.group(1).strip()
        if len(v) >= 2 and v[0] == '{' and v[-1] == '}':
            inner = v[1:-1].strip()
            parts = [s.strip() for s in inner.replace('\n', ' ').split(',')]
            return [p for p in parts if p != '']
        return v

    hdr = {}
    hdr['samples']    = int(val('samples'))
    hdr['lines']      = int(val('lines'))
    hdr['bands']      = int(val('bands'))
    hdr['interleave'] = (val('interleave','bsq') or 'bsq').lower()
    hdr['data_type']  = int(val('data type'))
    hdr['byte_order'] = int(val('byte order', 0))
    hdr['header offset'] = int(val('header offset', 0) or 0)
    hdr['wavelength'] = val('wavelength', None)
    return hdr

def memmap_envi(img_path, hdr):
    if not os.path.exists(img_path):
        raise FileNotFoundError("No se encontró el archivo de datos: " + img_path)
    dt = DTYPE_MAP.get(hdr['data_type'])
    if dt is None:
        raise ValueError("data type ENVI no soportado: %s" % str(hdr['data_type']))
    order = '<' if hdr['byte_order'] == 0 else '>'
    dt = np.dtype(dt).newbyteorder(order)

    L, S, B = hdr['lines'], hdr['samples'], hdr['bands']
    inter = hdr['interleave']
    off = int(hdr.get('header offset', 0))

    mm = np.memmap(img_path, mode='r', dtype=dt, offset=off)
    expected = L * S * B
    if mm.size < expected:
        raise ValueError("Tamaño inesperado: esperados >= %d elementos, hay %d" % (expected, mm.size))

    if inter == 'bsq':
        arr = np.ndarray(shape=(B, L, S), dtype=dt, buffer=mm)
        cube = np.transpose(arr, (1,2,0))  # (L,S,B)
        return cube, 'bsq'
    elif inter == 'bil':
        arr = np.ndarray(shape=(L, B, S), dtype=dt, buffer=mm)
        cube = np.transpose(arr, (0,2,1))  # (L,S,B)
        return cube, 'bil'
    elif inter == 'bip':
        cube = np.ndarray(shape=(L, S, B), dtype=dt, buffer=mm)  # (L,S,B)
        return cube, 'bip'
    else:
        raise ValueError("interleave ENVI no soportado: " + inter)

# ---- Visualización ----
def stretch01(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (2, 98))
    if hi <= lo:
        hi = lo + 1e-6
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0, 1)

def pick_rgb_indices(hdr):
    """Elige bandas para RGB:
       - Si hay 'wavelength' las convierte a nm (si venían en µm) y usa R=1500,G=1302,B=1104 nm.
       - Si no hay 'wavelength', reparte por terciles.
    """
    B = hdr['bands']
    wl = hdr.get('wavelength')
    if wl and isinstance(wl, list):
        arr = []
        for w in wl:
            try:
                arr.append(float(w))
            except:
                arr.append(np.nan)
        wlf = np.array(arr, dtype=float)
        # Si la mediana < 10 asumimos µm -> pasar a nm
        if np.nanmedian(wlf) < 10.0:
            wlf = wlf * 1000.0

        def nearest(t_nm):
            return int(np.nanargmin(np.abs(wlf - t_nm)))

        r_nm, g_nm, b_nm = SWIR_RGB_NM
        return nearest(r_nm), nearest(g_nm), nearest(b_nm)

    # Sin wavelengths
    r = min(B-1, int(0.9*B))
    g = min(B-1, int(0.6*B))
    b = min(B-1, int(0.3*B))
    return r, g, b

def compose_rgb(cube, hdr):
    r_i, g_i, b_i = pick_rgb_indices(hdr)
    r = stretch01(cube[:,:,r_i])
    g = stretch01(cube[:,:,g_i])
    b = stretch01(cube[:,:,b_i])
    rgb = np.dstack([r,g,b])
    return rgb, (r_i, g_i, b_i)

# ---- Utilidades ROI/Guardado ----
def normalize_rect(x, y, w, h):
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))

def clamp_rect(x, y, w, h, W, H):
    if w > W: w = W
    if h > H: h = H
    x = max(0, min(x, W - w))
    y = max(0, min(y, H - h))
    return x, y, w, h

def save_subcube_BSQ(cube, hdr, x, y, w, h, out_dir, base="ROI"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sub = np.asarray(cube[y:y+h, x:x+w, :])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    basepath = os.path.join(out_dir, "%s_x%d_y%d_w%d_h%d_%s" % (base, x, y, w, h, ts))
    img_path = basepath + ".img"   # binario de salida
    hdr_path = basepath + ".hdr"

    # Escribir datos en BSQ banda por banda
    Ls, Ss, Bs = sub.shape
    with open(img_path, 'wb') as f:
        for b in range(Bs):
            f.write(sub[:,:,b].tobytes(order='C'))

    # Header ENVI
    lines = []
    lines.append("ENVI")
    lines.append("samples = %d" % Ss)
    lines.append("lines   = %d" % Ls)
    lines.append("bands   = %d" % Bs)
    lines.append("header offset = 0")
    lines.append("file type = ENVI Standard")
    lines.append("interleave = bsq")
    lines.append("data type  = %d" % hdr['data_type'])
    lines.append("byte order = %d" % hdr['byte_order'])
    if hdr.get('wavelength') and isinstance(hdr['wavelength'], list):
        lines.append("wavelength = {%s}" % ', '.join(hdr['wavelength']))

    with open(hdr_path, 'w', encoding='latin-1') as fh:
        fh.write("\n".join(lines) + "\n")

    print("[OK] Subcubo guardado:")
    print("  " + hdr_path)
    print("  " + img_path)
    return hdr_path, img_path

# ---- Main ----
def main():
    hdr = parse_envi_hdr(HDR_PATH)
    cube, inter = memmap_envi(IMG_PATH, hdr)
    H, W, B = cube.shape
    print("Cubo: %dx%d px, %d bandas | interleave=%s" % (W, H, B, inter))

    x, y, w, h = normalize_rect(INIT_X, INIT_Y, INIT_W, INIT_H)
    x, y, w, h = clamp_rect(x, y, w, h, W, H)

    rgb, rgb_idx = compose_rgb(cube, hdr)

    fig, ax = plt.subplots()
    ax.set_navigate(False)
    ax.imshow(rgb, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, W); ax.set_ylim(H, 0)

    rect = Rectangle((x, y), w, h, fill=False, linewidth=1.8)
    ax.add_patch(rect)

    try:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    except Exception:
        pass

    title_top = "RGB (band idx) R=%d G=%d B=%d | Arrastra | Flechas=%d px | S=guardar | Q/Esc=salir" % (rgb_idx[0], rgb_idx[1], rgb_idx[2], STEP)
    def set_title():
        ax.set_title(title_top + "\nRect: x=%d, y=%d, w=%d, h=%d" % (x, y, w, h))

    dragging = False
    offset = (0, 0)

    def update():
        rect.set_xy((x, y))
        set_title()
        fig.canvas.draw_idle()

    def on_press(ev):
        nonlocal dragging, offset
        if ev.inaxes != ax or ev.xdata is None or ev.ydata is None:
            return
        if (x <= ev.xdata <= x + w) and (y <= ev.ydata <= y + h):
            dragging = True
            offset = (ev.xdata - x, ev.ydata - y)

    def on_motion(ev):
        nonlocal x, y
        if not dragging or ev.inaxes != ax or ev.xdata is None or ev.ydata is None:
            return
        nx = int(round(ev.xdata - offset[0]))
        ny = int(round(ev.ydata - offset[1]))
        nx, ny, _, _ = clamp_rect(nx, ny, w, h, W, H)
        if nx != x or ny != y:
            x, y = nx, ny
            update()

    def on_release(ev):
        nonlocal dragging
        dragging = False

    def on_key(e):
        nonlocal x, y
        if e.key == 'left':
            x = max(0, x - STEP)
        elif e.key == 'right':
            x = min(W - w, x + STEP)
        elif e.key == 'up':
            y = max(0, y - STEP)
        elif e.key == 'down':
            y = min(H - h, y + STEP)
        elif e.key and e.key.lower() == 's':
            save_subcube_BSQ(cube, hdr, x, y, w, h, OUT_DIR)
        elif e.key and e.key.lower() in ('q', 'escape'):
            plt.close(fig); return
        update()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)

    update()
    plt.show()

if __name__ == "__main__":
    main()
