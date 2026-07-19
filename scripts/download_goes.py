"""Descarga LST de GOES-East (producto ABI-L2-LSTF, ~2 km, horario) desde AWS
(acceso anonimo, sin cuenta) y lo remapea a la grilla de 0.01 grados de Arequipa.

Uso:
  python scripts/download_goes.py --start 2025-01-01 --end 2025-03-31 --out data/goes

Genera un npz por hora: goes_YYYYMMDD_HH.npz con {lst (K), dqf}.
GOES-16 fue GOES-East hasta 2025-04-04; despues GOES-19.
"""
import argparse
import datetime as dt
import os

import numpy as np
import s3fs
import xarray as xr

from geo_utils import NNRemap, goes_latlon, LAT_N, LAT_S, LON_W, LON_E

SWITCH = dt.date(2025, 4, 4)  # GOES-16 -> GOES-19 como GOES-East
_remap_cache = {}


def bucket_for(day):
    return "noaa-goes16" if day < SWITCH else "noaa-goes19"


def process_hour(fs, day, hour, outdir):
    out = os.path.join(outdir, f"goes_{day:%Y%m%d}_{hour:02d}.npz")
    if os.path.exists(out):
        return "skip"
    doy = day.timetuple().tm_yday
    prefix = f"{bucket_for(day)}/ABI-L2-LSTF/{day.year}/{doy:03d}/{hour:02d}/"
    try:
        files = fs.ls(prefix)
    except FileNotFoundError:
        return "no-list"
    if not files:
        return "no-file"
    with fs.open(files[0], "rb") as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        # subconjunto grueso alrededor de Arequipa para acelerar
        lat, lon = goes_latlon(ds)
        m = ((lat > LAT_S - 0.5) & (lat < LAT_N + 0.5) &
             (lon > LON_W - 0.5) & (lon < LON_E + 0.5))
        if not m.any():
            ds.close()
            return "outside"
        r0, r1 = np.where(m.any(axis=1))[0][[0, -1]]
        c0, c1 = np.where(m.any(axis=0))[0][[0, -1]]
        lst = ds["LST"].values[r0:r1 + 1, c0:c1 + 1].astype(np.float32)
        dqf = ds["DQF"].values[r0:r1 + 1, c0:c1 + 1].astype(np.float32)
        ds.close()
    key = (r0, r1, c0, c1)
    if key not in _remap_cache:
        _remap_cache[key] = NNRemap(lat[r0:r1 + 1, c0:c1 + 1],
                                    lon[r0:r1 + 1, c0:c1 + 1])
    remap = _remap_cache[key]
    lst[dqf != 0] = np.nan            # solo pixeles de buena calidad
    np.savez_compressed(out, lst=remap(lst).astype(np.float32))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/goes")
    ap.add_argument("--hours", default="all", help="'all' o lista '14,15,18'")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)
    d0 = dt.date.fromisoformat(args.start)
    d1 = dt.date.fromisoformat(args.end)
    hours = range(24) if args.hours == "all" else [int(h) for h in args.hours.split(",")]

    day = d0
    while day <= d1:
        n_ok = 0
        for h in hours:
            try:
                if process_hour(fs, day, h, args.out) == "ok":
                    n_ok += 1
            except Exception as e:  # tolerante: un archivo malo no detiene el lote
                print(f"[warn] {day} {h:02d}Z: {e}")
        print(f"[goes] {day}: {n_ok} horas nuevas")
        day += dt.timedelta(days=1)


if __name__ == "__main__":
    main()
