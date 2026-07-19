"""Descarga LST de VIIRS (VNP21A1D dia / VNP21A1N noche, 1 km) via earthaccess
(requiere cuenta NASA Earthdata) y lo remapea a la grilla objetivo.

Uso (tras earthaccess.login() o variables EARTHDATA_USERNAME/PASSWORD):
  python scripts/download_viirs.py --start 2025-01-01 --end 2025-03-31 --out data/viirs

Genera viirs_YYYYMMDD_day.npz / _night.npz con {lst (K), hour_utc}.
VIIRS reemplaza a MODIS del paper (Terra esta en orbita degradada); misma
resolucion de 1 km y algoritmo TES equivalente (VNP21 ~ MOD21).
"""
import argparse
import datetime as dt
import os
import re
import tempfile

import h5py
import numpy as np

from geo_utils import NNRemap, sin_tile_latlon, LAT_N, LAT_S, LON_W, LON_E

BBOX = (LON_W, LAT_S, LON_E, LAT_N)  # W, S, E, N
_remap_cache = {}


def read_grid(h5, names):
    """Busca un dataset por nombre dentro del arbol HDFEOS."""
    hits = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] in names:
            hits.append(name)
    h5.visititems(visit)
    return hits[0] if hits else None


def scaled(dset):
    a = dset[()].astype(np.float32)
    attrs = dset.attrs
    fill = attrs.get("_FillValue", [None])[0] if "_FillValue" in attrs else None
    if fill is not None:
        a[a == fill] = np.nan
    sf = float(attrs.get("scale_factor", [1.0])[0]) if "scale_factor" in attrs else 1.0
    off = float(attrs.get("add_offset", [0.0])[0]) if "add_offset" in attrs else 0.0
    return a * sf + off


def process_granule(path, kind, outdir):
    m = re.search(r"\.A(\d{4})(\d{3})\..*h(\d{2})v(\d{2})", os.path.basename(path))
    if not m:
        return "skip-name"
    year, doy, h, v = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    day = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)
    out = os.path.join(outdir, f"viirs_{day:%Y%m%d}_{kind}.npz")
    prev = None
    if os.path.exists(out):
        prev = np.load(out)  # otro tile del mismo dia: se fusionan
    with h5py.File(path, "r") as f:
        p_lst = read_grid(f, {"LST_1KM", "LST"})
        p_qc = read_grid(f, {"QC", "QC_Day", "QC_Night"})
        p_vt = read_grid(f, {"View_Time", "view_time"})
        if p_lst is None:
            return "no-lst"
        lst = scaled(f[p_lst])
        if p_qc is not None:
            qc = f[p_qc][()]
            lst[(qc & 0b11) > 1] = np.nan   # bits 0-1: calidad mandatoria
        vt = scaled(f[p_vt]) if p_vt is not None else None
    key = (h, v)
    if key not in _remap_cache:
        lat, lon = sin_tile_latlon(h, v)
        _remap_cache[key] = NNRemap(lat, lon)
    remap = _remap_cache[key]
    lst_t = remap(lst).astype(np.float32)
    if prev is not None:  # fusion con el tile vecino ya procesado
        gap = ~np.isfinite(lst_t)
        lst_t[gap] = prev["lst"][gap]
    if not np.isfinite(lst_t).any():
        return "empty"
    # hora media de observacion (hora local solar -> UTC aprox con lon central)
    hour_utc = float(prev["hour_utc"]) if prev is not None else np.nan
    if vt is not None and not np.isfinite(hour_utc):
        vt_t = remap(vt)
        if np.isfinite(vt_t).any():
            local = float(np.nanmean(vt_t))
            hour_utc = (local - (LON_W + LON_E) / 2.0 / 15.0) % 24.0
    np.savez_compressed(out, lst=lst_t, hour_utc=np.float32(hour_utc))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/viirs")
    ap.add_argument("--products", default="VNP21A1D,VNP21A1N")
    args = ap.parse_args()

    import earthaccess
    earthaccess.login()  # usa EARTHDATA_USERNAME/PASSWORD, .netrc o interactivo

    os.makedirs(args.out, exist_ok=True)
    for prod in args.products.split(","):
        kind = "day" if prod.endswith("D") else "night"
        results = earthaccess.search_data(
            short_name=prod, temporal=(args.start, args.end),
            bounding_box=BBOX)
        print(f"[viirs] {prod}: {len(results)} granulos")
        with tempfile.TemporaryDirectory() as tmp:
            files = earthaccess.download(results, tmp)
            n = 0
            for fp in files:
                try:
                    if process_granule(fp, kind, args.out) == "ok":
                        n += 1
                except Exception as e:
                    print(f"[warn] {os.path.basename(fp)}: {e}")
            print(f"[viirs] {prod}: {n} escenas procesadas")


if __name__ == "__main__":
    main()
