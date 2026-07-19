"""Construye los pares de entrenamiento (GOES 2km + SZA + DEM) -> (VIIRS 1km).

Para cada escena VIIRS busca el archivo GOES de la hora mas cercana (+-1 h)
y arma un sample .npz con:
  x: (3, H, W)  [lst_goes_norm, cos_sza, dem_norm]
  y: (H, W)     lst_viirs (K)
  mask: (H, W)  pixeles validos en ambos

Uso:
  python scripts/build_dataset.py --goes data/goes --viirs data/viirs --out data/samples
"""
import argparse
import datetime as dt
import glob
import os
import re

import numpy as np

try:
    from geo_utils import target_mesh, solar_zenith
except ImportError:  # cuando se importa como scripts.build_dataset
    from scripts.geo_utils import target_mesh, solar_zenith

LST_MEAN, LST_STD = 290.0, 15.0     # normalizacion (K)
DEM_MEAN, DEM_STD = 2500.0, 1800.0  # normalizacion (m)


def load_dem(path="data/dem.npy"):
    if os.path.exists(path):
        return np.load(path)
    print("[dem] descargando Copernicus DEM 90m desde AWS (anonimo)...")
    import rasterio
    from rasterio.warp import transform as rio_transform
    os.makedirs(os.path.dirname(path), exist_ok=True)
    LON2D, LAT2D = target_mesh()
    dem = np.zeros(LON2D.shape, dtype=np.float32)
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    tiles = [(s, w) for s in (16, 17, 18) for w in (71, 72, 73, 74)]
    filled = np.zeros(LON2D.shape, dtype=bool)
    for s, w in tiles:
        url = (f"s3://copernicus-dem-90m/Copernicus_DSM_COG_30_S{s}_00_"
               f"W0{w}_00_DEM/Copernicus_DSM_COG_30_S{s}_00_W0{w}_00_DEM.tif")
        m = (LAT2D <= -(s - 1)) & (LAT2D > -s) & (LON2D >= -w) & (LON2D < -(w - 1))
        if not m.any():
            continue
        try:
            with rasterio.open(url) as src:
                rows, cols = src.index(LON2D[m], LAT2D[m])
                band = src.read(1)
                rows = np.clip(rows, 0, band.shape[0] - 1)
                cols = np.clip(cols, 0, band.shape[1] - 1)
                dem[m] = band[rows, cols]
                filled |= m
        except Exception as e:
            print(f"[dem][warn] tile S{s} W{w}: {e}")
    if not filled.all():
        print(f"[dem][warn] {(~filled).sum()} pixeles sin DEM (quedan en 0)")
    np.save(path, dem)
    return dem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goes", default="data/goes")
    ap.add_argument("--viirs", default="data/viirs")
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--min-valid", type=float, default=0.10,
                    help="fraccion minima de pixeles validos")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dem = load_dem()
    dem_n = (dem - DEM_MEAN) / DEM_STD
    LON2D, LAT2D = target_mesh()

    n_ok = 0
    for vf in sorted(glob.glob(os.path.join(args.viirs, "viirs_*.npz"))):
        m = re.search(r"viirs_(\d{8})_(day|night)", vf)
        day = dt.datetime.strptime(m[1], "%Y%m%d")
        v = np.load(vf)
        hour = float(v["hour_utc"])
        if not np.isfinite(hour):
            hour = 18.0 if m[2] == "day" else 6.0   # ~13:30/01:30 hora local
        # GOES de la hora mas cercana (la escena VIIRS es de esa manana/noche)
        cands = [int(round(hour)) % 24, (int(round(hour)) + 1) % 24,
                 (int(round(hour)) - 1) % 24]
        gfile = None
        for hh in cands:
            p = os.path.join(args.goes, f"goes_{m[1]}_{hh:02d}.npz")
            if os.path.exists(p):
                gfile, ghour = p, hh
                break
        if gfile is None:
            continue
        g = np.load(gfile)["lst"]
        y = v["lst"]
        mask = np.isfinite(g) & np.isfinite(y)
        if mask.mean() < args.min_valid:
            continue
        t = day.replace(hour=ghour)
        sza = solar_zenith(t, LAT2D, LON2D)
        x = np.stack([
            np.nan_to_num((g - LST_MEAN) / LST_STD, nan=0.0),
            np.cos(np.deg2rad(sza)).astype(np.float32),
            dem_n.astype(np.float32),
        ])
        out = os.path.join(args.out, f"sample_{m[1]}_{m[2]}.npz")
        np.savez_compressed(out, x=x.astype(np.float32),
                            y=np.nan_to_num(y, nan=0.0).astype(np.float32),
                            mask=mask, date=m[1])
        n_ok += 1
    print(f"[dataset] {n_ok} muestras generadas en {args.out}")


if __name__ == "__main__":
    main()
