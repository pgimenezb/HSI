# -*- coding: utf-8 -*-
# Concatena DOS cubos ENVI (arriba/abajo) SIN modificar tamaños ni interleave.
import os, re
import numpy as np

DTYPE_MAP = {1:np.uint8,2:np.int16,3:np.int32,4:np.float32,5:np.float64,12:np.uint16,13:np.uint32,14:np.int64,15:np.uint64}

def parse_hdr(h):
    txt=open(h,'r',encoding='latin-1').read()
    def val(k,d=None):
        m=re.search(r'%s\s*=\s*(\{[^}]*\}|[^\r\n]+)'%k,txt,re.IGNORECASE)
        if not m: return d
        v=m.group(1).strip()
        if len(v)>=2 and v[0]=='{' and v[-1]=='}':
            inner=v[1:-1].strip()
            return [s.strip() for s in inner.replace('\n',' ').split(',') if s.strip()!='']
        return v
    H={}
    H['samples']=int(val('samples')); H['lines']=int(val('lines')); H['bands']=int(val('bands'))
    H['interleave']=(val('interleave','bsq') or 'bsq').lower()
    H['data_type']=int(val('data type')); H['byte_order']=int(val('byte order',0))
    H['header offset']=int(val('header offset',0) or 0)
    H['wavelength']=val('wavelength',None)
    # copiar extras si existen
    for opt in ['wavelength units','data ignore value','default bands','sensor type',
                'band names','fwhm','bbl','reflectance scale factor','z plot range',
                'map info','pixel size','x start','y start','projection info']:
        v=val(opt,None)
        if v is not None: H[opt]=v
    return H

def write_hdr(path, meta):
    lines=['ENVI']
    for k,v in meta.items():
        if isinstance(v,list): lines.append(f"{k} = {{{', '.join(str(x) for x in v)}}}")
        else: lines.append(f"{k} = {v}")
    open(path,'w',encoding='latin-1').write("\n".join(lines)+"\n")

def concat_vertical_same_interleave(top_hdr, top_img, bot_hdr, bot_img, out_base):
    Ht=parse_hdr(top_hdr); Hb=parse_hdr(bot_hdr)
    # comprobaciones (NO modificamos tamaños)
    if Ht['samples']!=Hb['samples']:
        raise ValueError(f"samples distintos ({Ht['samples']} vs {Hb['samples']})")
    if Ht['bands']!=Hb['bands']:
        raise ValueError(f"bands distintas ({Ht['bands']} vs {Hb['bands']})")
    if Ht['data_type']!=Hb['data_type']:
        raise ValueError(f"data type distinto ({Ht['data_type']} vs {Hb['data_type']})")
    if Ht['byte_order']!=Hb['byte_order']:
        raise ValueError(f"byte order distinto ({Ht['byte_order']} vs {Hb['byte_order']})")
    # interleave de salida = del TOP
    inter = Ht['interleave']
    dt = np.dtype(DTYPE_MAP[Ht['data_type']]).newbyteorder('<' if Ht['byte_order']==0 else '>')
    # mapear crudo en su interleave nativo para poder escribir streaming
    off_t, off_b = int(Ht.get('header offset',0)), int(Hb.get('header offset',0))
    mm_t = np.memmap(top_img, mode='r', dtype=dt, offset=off_t)
    mm_b = np.memmap(bot_img, mode='r', dtype=dt, offset=off_b)
    Lt, Lb = Ht['lines'], Hb['lines']; S = Ht['samples']; B = Ht['bands']

    out_img = out_base + (os.path.splitext(top_img)[1] if os.path.splitext(top_img)[1] else ".img")
    out_hdr = out_base + ".hdr"

    with open(out_img,'wb') as f:
        if inter=='bsq':
            # top en forma (B, Lt, S), luego bottom (B, Lb, S)
            arr_t = np.ndarray((B,Lt,S), dtype=dt, buffer=mm_t)
            arr_b = np.ndarray((B,Lb,S), dtype=dt, buffer=mm_b)
            for b in range(B):
                f.write(np.ascontiguousarray(arr_t[b,:,:]).tobytes(order='C'))
                f.write(np.ascontiguousarray(arr_b[b,:,:]).tobytes(order='C'))
        elif inter=='bil':
            # (Lt, B, S) seguido de (Lb, B, S): escribir línea a línea
            arr_t = np.ndarray((Lt,B,S), dtype=dt, buffer=mm_t)
            arr_b = np.ndarray((Lb,B,S), dtype=dt, buffer=mm_b)
            for y in range(Lt):
                f.write(np.ascontiguousarray(arr_t[y,:,:]).tobytes(order='C'))
            for y in range(Lb):
                f.write(np.ascontiguousarray(arr_b[y,:,:]).tobytes(order='C'))
        elif inter=='bip':
            # (Lt, S, B) seguido de (Lb, S, B): escribir fila a fila
            arr_t = np.ndarray((Lt,S,B), dtype=dt, buffer=mm_t)
            arr_b = np.ndarray((Lb,S,B), dtype=dt, buffer=mm_b)
            for y in range(Lt):
                f.write(np.ascontiguousarray(arr_t[y,:,:]).tobytes(order='C'))
            for y in range(Lb):
                f.write(np.ascontiguousarray(arr_b[y,:,:]).tobytes(order='C'))
        else:
            raise ValueError("interleave no soportado: "+inter)

    # header de salida (mismas características, lines = Lt+Lb)
    meta = {
        'samples': S,
        'lines': Lt+Lb,
        'bands': B,
        'header offset': 0,
        'file type': 'ENVI Standard',
        'interleave': inter,
        'data type': Ht['data_type'],
        'byte order': Ht['byte_order'],
    }
    # heredar metadatos del top (wavelengths, etc.)
    for opt in ['wavelength units','data ignore value','default bands','sensor type',
                'band names','fwhm','bbl','reflectance scale factor','z plot range',
                'map info','pixel size','x start','y start','projection info','wavelength']:
        if opt in Ht: meta[opt]=Ht[opt]
    write_hdr(out_hdr, meta)
    return out_hdr, out_img

# ==== EDITA AQUÍ ====
if __name__ == "__main__":
    top_hdr = r"C:\Users\pgimenezbarrera\Desktop\Hyperspectral data\ROIs2\Ultramarin_GA_45010_1.hdr"
    top_img = r"C:\Users\pgimenezbarrera\Desktop\Hyperspectral data\ROIs2\Ultramarin_GA_45010_1.dat"  
    bot_hdr = r"C:\Users\pgimenezbarrera\Desktop\Hyperspectral data\ROIs2\Ultramarin_GA_45010.hdr"
    bot_img = r"C:\Users\pgimenezbarrera\Desktop\Hyperspectral data\ROIs2\Ultramarin_GA_45010.dat"

    out_base = r"C:\Users\pgimenezbarrera\Desktop\Hyperspectral data\ROIs2\Ultramarin_GA_45010_VSTACK"

    hdr_out, img_out = concat_vertical_same_interleave(top_hdr, top_img, bot_hdr, bot_img, out_base)
    print("Mosaico vertical creado:\n ", hdr_out, "\n ", img_out)
