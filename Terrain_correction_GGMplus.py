"""
Complete Bouguer Anomaly (CBA) dari data GGMplus + DEMNAS
==========================================================

Input:
  - ggmplus_all.txt : kolom  lon_deg lat_deg dg_mGal ga_mGal xi_arcsec eta_arcsec geoid_m
                      (dipisah spasi, grid 0.002 deg ~ 220 m)
        dg_mGal = gravity disturbance di permukaan topografi (setara free-air anomaly)
        ga_mGal = percepatan gravitasi total di permukaan topografi
  - DEMNAS GeoTIFF (https://tanahair.indonesia.go.id/portal-web/unduh/demnas)
        resolusi 0.27" (~8 m), datum vertikal EGM2008 (tinggi orthometrik).
        PENTING: DEM harus menutupi area data + buffer sebesar R_OUTER (default 20 km
        ~ 0.18 deg) di semua sisi. Gabungkan (merge) beberapa tile DEMNAS di QGIS
        menjadi satu GeoTIFF bila perlu.

Koreksi terrain memakai metode sektor-cincin (Hammer-style) dengan rumus EKSAK
sektor silinder untuk setiap kompartemen:

    TC = G * rho * dphi * [ (r2 - r1) + sqrt(r1^2 + dh^2) - sqrt(r2^2 + dh^2) ]

dh = h_kompartemen - h_stasiun. TC selalu positif (bukit maupun lembah sama-sama
mengurangi gravitasi relatif terhadap slab Bouguer), sehingga:

    SBA = FAA - BC
    CBA = FAA - BC + TC

Paket: numpy, pandas, scipy, rasterio, matplotlib
"""

import time
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# User inputs
# -------------------------------------------------------------------------
FOLDER = "D:/OneDrive - UNIVERSITAS INDONESIA/modul kuliah/Mata Kuliah Analisis Data Geofisika 2/Data_sample_Gravity_GGMplus/"
GRAVITY_TXT = FOLDER + "ggmplus_all.txt"          # data GGMplus
DEM_TIF     = FOLDER + "DEMNAS_mosaic.tif"        # DEMNAS (sudah di-merge)
OUTPUT_CSV  = FOLDER + "CBA_ggmplus.csv"
OUTPUT_TIF  = FOLDER + "CBA_ggmplus.tif"          # grid CBA untuk dibuka di QGIS
OUTPUT_PNG  = FOLDER + "CBA_map.png"

RHO = 2670.0             # densitas Bouguer, kg/m^3 (2.67 g/cc)

# Free-air anomaly yang dipakai:
#   "dg"      -> pakai dg_mGal GGMplus langsung (disarankan; konsisten secara internal)
#   "classic" -> ga_mGal - gamma0(lat) + 0.3086 * H_DEMNAS
FAA_SOURCE = "dg"

# Parameter terrain correction
R_INNER  = 25.0          # m, radius cincin pertama (>= resolusi DEM yang dipakai)
R_OUTER  = 20000.0       # m, radius terluar (zona Hammer M ~ 21.9 km)
N_RINGS  = 60            # jumlah cincin (jarak geometrik / logaritmik)
N_AZ     = 48            # jumlah sektor azimuth per cincin
DEM_RES_TARGET = 25.0    # m, DEMNAS (~8 m) di-resample (average) ke resolusi ini
CHUNK    = 2000          # jumlah stasiun per batch (atur sesuai RAM)
# -------------------------------------------------------------------------

G = 6.67430e-11
SI_TO_MGAL = 1e5


# =========================================================================
# Utilitas
# =========================================================================
def normal_gravity(lat_deg):
    """Gravitasi normal GRS80 di elipsoid (Somigliana), mGal."""
    s2 = np.sin(np.radians(lat_deg)) ** 2
    g_eq = 978032.67715
    k = 0.001931851353
    e2 = 0.00669438002290
    return g_eq * (1 + k * s2) / np.sqrt(1 - e2 * s2)


def m_per_deg(lat_deg):
    """Panjang 1 derajat lintang & bujur (m) pada lintang tertentu (WGS84)."""
    p = np.radians(lat_deg)
    m_lat = 111132.954 - 559.822 * np.cos(2 * p) + 1.175 * np.cos(4 * p)
    m_lon = 111412.84 * np.cos(p) - 93.5 * np.cos(3 * p)
    return m_lat, m_lon


class DemGrid:
    """DEM dalam array numpy + georeferensi sederhana (north-up)."""

    def __init__(self, data, x0, y0, dx, dy, geographic=True):
        self.data = data.astype(np.float32)   # NaN = nodata
        self.x0, self.y0 = x0, y0             # pojok kiri-atas
        self.dx, self.dy = dx, dy             # ukuran piksel (dy positif)
        self.geographic = geographic

    def sample(self, x, y):
        """Interpolasi bilinear; di luar DEM -> NaN."""
        col = (x - self.x0) / self.dx - 0.5
        row = (self.y0 - y) / self.dy - 0.5
        shp = np.shape(x)
        out = map_coordinates(self.data, [np.ravel(row), np.ravel(col)],
                              order=1, mode="constant", cval=np.nan)
        return out.reshape(shp)


def load_dem(path, xmin, xmax, ymin, ymax, buffer_m, res_target):
    """Baca DEM (window area + buffer), resample average ke ~res_target meter."""
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    from rasterio.warp import transform

    src = rasterio.open(path)
    geographic = src.crs is None or src.crs.is_geographic

    if geographic:
        lat_c = 0.5 * (ymin + ymax)
        m_lat, m_lon = m_per_deg(lat_c)
        bx, by = buffer_m / m_lon, buffer_m / m_lat
        px_m = abs(src.res[0]) * m_lon
    else:
        xs, ys = transform("EPSG:4326", src.crs, [xmin, xmax, xmin, xmax],
                           [ymin, ymin, ymax, ymax])
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        bx = by = buffer_m
        px_m = abs(src.res[0])

    want = (xmin - bx, ymin - by, xmax + bx, ymax + by)
    b = src.bounds
    if want[0] < b.left or want[1] < b.bottom or want[2] > b.right or want[3] > b.top:
        print("PERINGATAN: DEM tidak menutupi seluruh area + buffer "
              f"{buffer_m/1000:.0f} km.\n  dibutuhkan: {want}\n  DEM      : {tuple(b)}\n"
              "  Kompartemen di luar DEM dilewati (lihat kolom tc_coverage).")
    win = from_bounds(max(want[0], b.left), max(want[1], b.bottom),
                      min(want[2], b.right), min(want[3], b.top), src.transform)
    win = win.round_offsets().round_lengths()

    factor = max(1, int(round(res_target / px_m)))
    out_h = max(1, int(win.height // factor))
    out_w = max(1, int(win.width // factor))
    data = src.read(1, window=win, out_shape=(out_h, out_w),
                    resampling=Resampling.average, masked=True)
    data = data.astype(np.float32).filled(np.nan)
    if src.nodata is None:                      # beberapa tile pakai -9999/-32767
        data[data < -1000] = np.nan

    wt = src.window_transform(win)
    dx = wt.a * win.width / out_w
    dy = -wt.e * win.height / out_h
    print(f"DEM dibaca: {out_w} x {out_h} piksel, resolusi ~{px_m*factor:.1f} m "
          f"(faktor resample {factor}), CRS {'geografis' if geographic else src.crs}")
    return DemGrid(data, wt.c, wt.f, dx, dy, geographic), src.crs


# =========================================================================
# Terrain correction
# =========================================================================
def build_compartments(r_inner, r_outer, n_rings, n_az):
    """Cincin geometrik (0 .. r_inner .. r_outer) x n_az sektor."""
    edges = np.concatenate([[0.0], np.geomspace(r_inner, r_outer, n_rings)])
    r1, r2 = edges[:-1], edges[1:]
    rc = np.where(r1 == 0, 0.5 * r2, np.sqrt(r1 * r2))   # radius tengah kompartemen
    dphi = 2 * np.pi / n_az
    az = (np.arange(n_az) + 0.5) * dphi                  # azimuth dari utara

    R1 = np.repeat(r1, n_az); R2 = np.repeat(r2, n_az); RC = np.repeat(rc, n_az)
    AZ = np.tile(az, len(r1))
    de = RC * np.sin(AZ)    # offset timur (m)
    dn = RC * np.cos(AZ)    # offset utara (m)
    return R1, R2, de, dn, dphi


def terrain_correction(x, y, dem, rho, r_inner=25., r_outer=20000.,
                       n_rings=40, n_az=32, chunk=2000, verbose=True):
    """
    x, y : koordinat stasiun dalam CRS DEM (lon/lat jika geografis)
    return: tc (mGal), h_station (m), coverage (fraksi kompartemen yg ada DEM)
    """
    R1, R2, de, dn, dphi = build_compartments(r_inner, r_outer, n_rings, n_az)
    n = len(x)
    tc = np.zeros(n); hs_all = np.full(n, np.nan); cov = np.zeros(n)
    t0 = time.time()

    for i0 in range(0, n, chunk):
        sl = slice(i0, min(i0 + chunk, n))
        xs, ys = np.asarray(x[sl], float), np.asarray(y[sl], float)
        hs = dem.sample(xs, ys)

        if dem.geographic:
            m_lat, m_lon = m_per_deg(ys)
            xc = xs[:, None] + de[None, :] / m_lon[:, None]
            yc = ys[:, None] + dn[None, :] / m_lat[:, None]
        else:
            xc = xs[:, None] + de[None, :]
            yc = ys[:, None] + dn[None, :]

        hc = dem.sample(xc, yc)
        dh = hc - hs[:, None]
        valid = np.isfinite(dh)
        dh = np.where(valid, dh, 0.0)

        g = (R2 - R1) + np.sqrt(R1**2 + dh**2) - np.sqrt(R2**2 + dh**2)
        tc[sl] = G * rho * dphi * g.sum(axis=1) * SI_TO_MGAL
        hs_all[sl] = hs
        cov[sl] = valid.mean(axis=1)

        if verbose:
            done = sl.stop
            el = time.time() - t0
            print(f"\r  TC: {done}/{n} stasiun  ({el:.0f} s, "
                  f"sisa ~{el/done*(n-done):.0f} s)", end="")
    if verbose:
        print()
    tc[~np.isfinite(hs_all)] = np.nan
    return tc, hs_all, cov


# =========================================================================
# Output grid
# =========================================================================
def to_grid(df, col):
    """Kembalikan (grid, lons, lats) bila data berbentuk grid teratur."""
    piv = df.pivot_table(index="lat", columns="lon", values=col)
    return piv.values, piv.columns.values, piv.index.values


def save_geotiff(path, grid, lons, lats):
    import rasterio
    from rasterio.transform import from_origin
    dx = np.median(np.diff(lons)); dy = np.median(np.diff(lats))
    grid_n = grid[::-1]                       # baris pertama = utara
    tr = from_origin(lons.min() - dx / 2, lats.max() + dy / 2, dx, dy)
    with rasterio.open(path, "w", driver="GTiff", height=grid_n.shape[0],
                       width=grid_n.shape[1], count=1, dtype="float32",
                       crs="EPSG:4326", transform=tr, nodata=np.nan) as dst:
        dst.write(grid_n.astype(np.float32), 1)


# =========================================================================
# Main
# =========================================================================
def main():
    # ---- 1. Data GGMplus ------------------------------------------------
    df = pd.read_csv(GRAVITY_TXT, sep=r"\s+")
    df = df.rename(columns={"lon_deg": "lon", "lat_deg": "lat",
                            "dg_mGal": "dg", "ga_mGal": "ga"})
    print(f"{len(df)} titik GGMplus: lon {df.lon.min():.3f}..{df.lon.max():.3f}, "
          f"lat {df.lat.min():.3f}..{df.lat.max():.3f}")

    # ---- 2. DEM ----------------------------------------------------------
    dem, crs = load_dem(DEM_TIF, df.lon.min(), df.lon.max(), df.lat.min(),
                        df.lat.max(), R_OUTER, DEM_RES_TARGET)
    if dem.geographic:
        xs, ys = df.lon.values, df.lat.values
    else:
        from rasterio.warp import transform
        xs, ys = map(np.asarray, transform("EPSG:4326", crs,
                                           df.lon.values, df.lat.values))

    # ---- 3. Terrain correction (+ elevasi stasiun) ------------------------
    print("Menghitung terrain correction ...")
    tc, h, cov = terrain_correction(xs, ys, dem, RHO, R_INNER, R_OUTER,
                                    N_RINGS, N_AZ, CHUNK)
    df["elevation_m"] = h
    df["terrain_corr"] = tc
    df["tc_coverage"] = cov

    # ---- 4. Free-air anomaly --------------------------------------------
    df["g_normal"] = normal_gravity(df.lat)
    df["faa_classic"] = df.ga - df.g_normal + 0.3086 * df.elevation_m
    df["free_air_anomaly"] = df.dg if FAA_SOURCE == "dg" else df.faa_classic

    # ---- 5. Bouguer & CBA -------------------------------------------------
    df["bouguer_corr"] = 2 * np.pi * G * RHO * df.elevation_m * SI_TO_MGAL  # 0.04193*rho*h
    df["SBA_mGal"] = df.free_air_anomaly - df.bouguer_corr
    df["CBA_mGal"] = df.SBA_mGal + df.terrain_corr

    bad = (df.tc_coverage < 0.99).sum()
    if bad:
        print(f"PERINGATAN: {bad} titik punya cakupan DEM < 99% (TC kurang).")
    print(df[["elevation_m", "terrain_corr", "bouguer_corr", "free_air_anomaly",
              "SBA_mGal", "CBA_mGal"]].describe().round(2))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV disimpan: {OUTPUT_CSV}")

    # ---- 6. Peta ---------------------------------------------------------
    grid, lons, lats = to_grid(df, "CBA_mGal")
    try:
        save_geotiff(OUTPUT_TIF, grid, lons, lats)
        print(f"GeoTIFF disimpan: {OUTPUT_TIF}")
    except Exception as e:
        print("GeoTIFF gagal disimpan:", e)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]
    for ax, col, cmap, ttl in zip(
            axs, ["elevation_m", "terrain_corr", "CBA_mGal"],
            ["terrain", "viridis", "jet"],
            ["Elevasi DEMNAS (m)", "Terrain correction (mGal)",
             f"CBA, rho={RHO/1000:.2f} g/cc (mGal)"]):
        g, _, _ = to_grid(df, col)
        im = ax.imshow(g, origin="lower", extent=ext, cmap=cmap, aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(ttl); ax.set_xlabel("Longitude")
    axs[0].set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
