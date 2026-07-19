"""Utilidades geoespaciales para el pipeline LST Arequipa.

- Grilla objetivo regular de 0.01 grados (~1 km) sobre Arequipa
- Remapeo por vecino mas cercano (KDTree) desde grillas irregulares
- Lat/lon de la proyeccion geoestacionaria de GOES
- Lat/lon de la grilla sinusoidal de VIIRS/MODIS
- Angulo cenital solar
"""
import numpy as np
from scipy.spatial import cKDTree

# ---- Grilla objetivo: Arequipa y alrededores (Misti, Chachani, costa) ----
LAT_N, LAT_S = -15.2, -17.6   # norte, sur
LON_W, LON_E = -73.0, -70.6   # oeste, este
RES = 0.01                    # ~1.1 km


def target_grid():
    lats = np.arange(LAT_N, LAT_S - 1e-9, -RES)   # descendente
    lons = np.arange(LON_W, LON_E + 1e-9, RES)
    return lats, lons


def target_mesh():
    lats, lons = target_grid()
    return np.meshgrid(lons, lats)  # LON2D, LAT2D


class NNRemap:
    """Remapeo vecino-mas-cercano de una grilla irregular a la grilla objetivo."""

    def __init__(self, src_lat, src_lon, max_dist_deg=0.05):
        valid = np.isfinite(src_lat) & np.isfinite(src_lon)
        self.valid = valid.ravel()
        pts = np.column_stack([src_lat.ravel()[self.valid],
                               src_lon.ravel()[self.valid]])
        self.tree = cKDTree(pts)
        LON2D, LAT2D = target_mesh()
        tgt = np.column_stack([LAT2D.ravel(), LON2D.ravel()])
        self.dist, self.idx = self.tree.query(tgt, k=1)
        self.shape_out = LAT2D.shape
        self.too_far = self.dist > max_dist_deg

    def __call__(self, field):
        flat = field.ravel()[self.valid].astype(np.float32)
        out = flat[self.idx]
        out[self.too_far] = np.nan
        return out.reshape(self.shape_out)


def goes_latlon(ds):
    """Calcula lat/lon 2D del subconjunto de un archivo ABI L2 (xarray Dataset)."""
    p = ds["goes_imager_projection"]
    h = float(p.attrs["perspective_point_height"])
    lon0 = np.deg2rad(float(p.attrs["longitude_of_projection_origin"]))
    req = float(p.attrs["semi_major_axis"])
    rpol = float(p.attrs["semi_minor_axis"])
    x = ds["x"].values.astype(np.float64)   # radianes
    y = ds["y"].values.astype(np.float64)
    X, Y = np.meshgrid(x, y)
    sinx, cosx = np.sin(X), np.cos(X)
    siny, cosy = np.sin(Y), np.cos(Y)
    H = h + req
    a = sinx**2 + cosx**2 * (cosy**2 + (req**2 / rpol**2) * siny**2)
    b = -2.0 * H * cosx * cosy
    c = H**2 - req**2
    disc = b**2 - 4 * a * c
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2 * a)
        sx = rs * cosx * cosy
        sy = -rs * sinx
        sz = rs * cosx * siny
        lat = np.arctan((req**2 / rpol**2) * sz / np.sqrt((H - sx)**2 + sy**2))
        lon = lon0 - np.arctan(sy / (H - sx))
    lat, lon = np.rad2deg(lat), np.rad2deg(lon)
    lat[disc < 0] = np.nan
    lon[disc < 0] = np.nan
    return lat, lon


# ---- Grilla sinusoidal VIIRS/MODIS (tiles 10x10 grados, 1200x1200 px de 1 km) ----
R_SIN = 6371007.181
T_SIN = 1111950.5196666666  # tamano de tile en metros
XMIN, YMAX = -20015109.354, 10007554.677
NPIX = 1200


def sin_tile_latlon(h, v):
    px = T_SIN / NPIX
    x = XMIN + h * T_SIN + (np.arange(NPIX) + 0.5) * px
    y = YMAX - v * T_SIN - (np.arange(NPIX) + 0.5) * px
    X, Y = np.meshgrid(x, y)
    lat = Y / R_SIN
    with np.errstate(invalid="ignore", divide="ignore"):
        lon = X / (R_SIN * np.cos(lat))
    lat, lon = np.rad2deg(lat), np.rad2deg(lon)
    bad = (np.abs(lon) > 180) | ~np.isfinite(lon)
    lat[bad] = np.nan
    lon[bad] = np.nan
    return lat, lon


def solar_zenith(dt_utc, lat2d, lon2d):
    """Angulo cenital solar (grados) — formula astronomica simple."""
    doy = dt_utc.timetuple().tm_yday
    frac_h = dt_utc.hour + dt_utc.minute / 60.0
    decl = np.deg2rad(-23.44 * np.cos(np.deg2rad(360.0 / 365.0 * (doy + 10))))
    hang = np.deg2rad((frac_h - 12.0) * 15.0 + lon2d)
    latr = np.deg2rad(lat2d)
    cosz = (np.sin(latr) * np.sin(decl) +
            np.cos(latr) * np.cos(decl) * np.cos(hang))
    return np.rad2deg(np.arccos(np.clip(cosz, -1, 1)))
