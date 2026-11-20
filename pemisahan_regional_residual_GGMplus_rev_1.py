#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pemisahan Anomali Regional - Residual & Analisis Spektrum (Radially Averaged
Amplitude Spectrum) untuk data Complete Bouguer Anomaly (CBA) grid lon/lat.

Alur kerja
----------
1. Baca CSV (kolom lon, lat, CBA_mGal), susun menjadi grid 2D (utara di atas).
   Baris/kolom yang seluruhnya NaN dipotong, NaN di dalam diisi nearest-neighbour.
2. Analisis spektrum CBA (FFT 2D -> rata-rata radial -> ln A vs k):
   - fit 3 segmen garis lurus (regional / residual / noise) secara otomatis
     (atau manual lewat --breaks),
   - kedalaman = -gradien (k dalam rad/km -> kedalaman dalam km),
   - bilangan gelombang cut-off kc = perpotongan garis regional & residual,
   - lebar jendela moving average N = 2*pi / (kc * dx).
3. Pemisahan regional-residual dengan salah satu / semua metode:
   gaussian (low-pass FFT), moving_average, polynomial (trend surface),
   upward (kontinuasi ke atas).  Residual = CBA - Regional.
4. Analisis spektrum anomali residual -> kedalaman rata-rata sumber residual.
5. Output: CSV (titik), GeoTIFF (EPSG:4326) tiap grid, CSV spektrum,
   ringkasan kedalaman (TXT), dan gambar PNG.

Ketergantungan: numpy, pandas, scipy, matplotlib, dan salah satu dari
rasterio ATAU tifffile (untuk GeoTIFF).

Contoh
------
python pemisahan_regional_residual.py                 # pilih file lewat dialog / input
python pemisahan_regional_residual.py CBA_ggmplus.csv -o hasil
python pemisahan_regional_residual.py CBA_ggmplus.csv --method all
python pemisahan_regional_residual.py CBA_ggmplus.csv --breaks 0.6 2.5
python pemisahan_regional_residual.py CBA_ggmplus.csv --method polynomial --poly-order 2
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_EARTH = 6371008.8  # m


# ---------------------------------------------------------------------------
# 0. Pilih file & kolom input
# ---------------------------------------------------------------------------
def choose_input_file():
    """Buka dialog pilih file; jika tidak ada GUI, minta path lewat terminal."""
    path = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title="Pilih file CSV anomali gravitasi",
                                          filetypes=[("CSV", "*.csv"), ("Semua file", "*.*")])
        root.destroy()
    except Exception:
        path = None
    while not path or not os.path.isfile(path):
        if path:
            print(f"File tidak ditemukan: {path}")
        path = input("Masukkan path file CSV: ").strip().strip('"').strip("'")
    print(f"File input: {path}")
    return path


def choose_column(path, zcol):
    """Jika kolom anomali tidak ada, tampilkan daftar kolom dan minta pilih."""
    cols = list(pd.read_csv(path, nrows=0).columns)
    if zcol in cols:
        return zcol
    print("Kolom anomali '%s' tidak ditemukan. Kolom tersedia:" % zcol)
    for i, c in enumerate(cols):
        print(f"  [{i}] {c}")
    while True:
        ans = input("Pilih nomor atau nama kolom anomali: ").strip()
        if ans.isdigit() and int(ans) < len(cols):
            return cols[int(ans)]
        if ans in cols:
            return ans


# ---------------------------------------------------------------------------
# 1. Data -> grid
# ---------------------------------------------------------------------------
def load_grid(path, xcol="lon", ycol="lat", zcol="CBA_mGal"):
    df = pd.read_csv(path)
    for c in (xcol, ycol, zcol):
        if c not in df.columns:
            raise KeyError(f"Kolom '{c}' tidak ada. Kolom tersedia: {list(df.columns)}")
    df[xcol] = df[xcol].round(8)
    df[ycol] = df[ycol].round(8)
    g = df.pivot_table(index=ycol, columns=xcol, values=zcol, aggfunc="mean", dropna=False)
    g = g.sort_index(ascending=False)          # lat menurun -> baris 0 = utara
    g = g.reindex(sorted(g.columns), axis=1)   # lon menaik
    lon = g.columns.values.astype(float)
    lat = g.index.values.astype(float)
    z = g.values.astype(float)

    # potong baris/kolom yang seluruhnya kosong
    rows = ~np.all(np.isnan(z), axis=1)
    cols = ~np.all(np.isnan(z), axis=0)
    if not rows.any():
        raise ValueError("Semua nilai kosong.")
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    n_cut = z.size - (r1 - r0 + 1) * (c1 - c0 + 1)
    z, lat, lon = z[r0:r1 + 1, c0:c1 + 1], lat[r0:r1 + 1], lon[c0:c1 + 1]

    # isi NaN internal (nearest neighbour)
    mask_nan = np.isnan(z)
    if mask_nan.any():
        idx = ndimage.distance_transform_edt(mask_nan, return_distances=False, return_indices=True)
        z = z[tuple(idx)]

    dlon = np.median(np.diff(lon))
    dlat = np.median(np.abs(np.diff(lat)))
    lat0 = np.deg2rad(lat.mean())
    dx = np.deg2rad(dlon) * R_EARTH * np.cos(lat0)   # m
    dy = np.deg2rad(dlat) * R_EARTH                  # m
    info = dict(n_cut=int(n_cut), n_filled=int(mask_nan.sum()), dlon=dlon, dlat=dlat, dx=dx, dy=dy)
    return lon, lat, z, mask_nan, info


# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------
def _next_pow2(n):
    return int(2 ** np.ceil(np.log2(n)))


def pad_grid(z, factor=2):
    """Pad simetris (mirror) sampai pangkat 2 >= factor*ukuran untuk menekan efek tepi."""
    ny, nx = z.shape
    NY, NX = _next_pow2(factor * ny), _next_pow2(factor * nx)
    py, px = NY - ny, NX - nx
    pad = ((py // 2, py - py // 2), (px // 2, px - px // 2))
    zp = np.pad(z, pad, mode="symmetric")
    return zp, pad


def wavenumber_grid(shape, dx, dy):
    """|k| dalam rad/m."""
    ny, nx = shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    return np.sqrt(KX ** 2 + KY ** 2)


def fft_filter(z, dx, dy, transfer):
    """Terapkan fungsi transfer H(|k|) di domain frekuensi."""
    mean = z.mean()
    zp, ((t, b), (l, r)) = pad_grid(z - mean)
    K = wavenumber_grid(zp.shape, dx, dy)
    out = np.real(np.fft.ifft2(np.fft.fft2(zp) * transfer(K)))
    out = out[t:t + z.shape[0], l:l + z.shape[1]]
    return out + mean


# ---------------------------------------------------------------------------
# 2. Analisis spektrum (radially averaged amplitude spectrum)
# ---------------------------------------------------------------------------
def radial_spectrum(z, dx, dy, detrend="plane", taper=0.1):
    ny, nx = z.shape
    zz = z.copy()
    if detrend == "plane":
        yy, xx = np.mgrid[0:ny, 0:nx]
        A = np.c_[np.ones(zz.size), xx.ravel(), yy.ravel()]
        coef, *_ = np.linalg.lstsq(A, zz.ravel(), rcond=None)
        zz = zz - (A @ coef).reshape(zz.shape)
    else:
        zz = zz - zz.mean()
    # Tukey taper pada tepi
    from scipy.signal.windows import tukey
    zz = zz * np.outer(tukey(ny, taper), tukey(nx, taper))
    NY, NX = _next_pow2(ny), _next_pow2(nx)
    F = np.fft.fft2(zz, s=(NY, NX))
    A = np.abs(F)
    K = wavenumber_grid((NY, NX), dx, dy) * 1000.0     # rad/km

    # bin radial; lebar bin = 2*pi / panjang sisi terpanjang data
    L = max(nx * dx, ny * dy) / 1000.0                  # km
    dk = 2 * np.pi / L
    knyq = np.pi / (max(dx, dy) / 1000.0)
    edges = np.arange(dk / 2, knyq + dk, dk)
    kc, lnA, n = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (K >= a) & (K < b)
        if m.sum() == 0:
            continue
        kc.append(K[m].mean())
        lnA.append(np.log(A[m].mean()))
        n.append(int(m.sum()))
    return np.array(kc), np.array(lnA), np.array(n)


def _linfit(x, y):
    p = np.polyfit(x, y, 1)
    sse = np.sum((np.polyval(p, x) - y) ** 2)
    yhat = np.polyval(p, x)
    r2 = 1 - sse / max(np.sum((y - y.mean()) ** 2), 1e-30)
    return p, sse, r2


def fit_segments(k, lnA, breaks=None, min_pts=3, kmax_frac=1.0):
    """Fit 3 segmen (regional, residual, noise). breaks = (k1, k2) dalam rad/km (opsional)."""
    use = k <= kmax_frac * k.max()
    k, lnA = k[use], lnA[use]
    n = len(k)
    if breaks is not None:
        i = int(np.searchsorted(k, breaks[0]))
        j = int(np.searchsorted(k, breaks[1]))
        best = (i, j)
    else:
        best, best_sse = None, np.inf
        for i in range(min_pts, n - 2 * min_pts + 1):
            for j in range(i + min_pts, n - min_pts + 1):
                sse = sum(_linfit(k[s], lnA[s])[1] for s in (slice(0, i), slice(i, j), slice(j, n)))
                if sse < best_sse:
                    best, best_sse = (i, j), sse
    i, j = best
    names = ["regional", "residual", "noise"]
    segs = {}
    for nm, s in zip(names, (slice(0, i), slice(i, j), slice(j, n))):
        p, sse, r2 = _linfit(k[s], lnA[s])
        segs[nm] = dict(slope=p[0], intercept=p[1], depth_km=-p[0], r2=r2,
                        kmin=k[s].min(), kmax=k[s].max(), npts=len(k[s]))
    r, s = segs["regional"], segs["residual"]
    kc = (s["intercept"] - r["intercept"]) / (r["slope"] - s["slope"])
    if not np.isfinite(kc) or kc <= 0:
        kc = 0.5 * (r["kmax"] + s["kmin"])
    return segs, kc


def fit_single(k, lnA, kmin, kmax):
    m = (k >= kmin) & (k <= kmax)
    p, sse, r2 = _linfit(k[m], lnA[m])
    return dict(slope=p[0], intercept=p[1], depth_km=-p[0], r2=r2,
                kmin=k[m].min(), kmax=k[m].max(), npts=int(m.sum()))


# ---------------------------------------------------------------------------
# 3. Metode pemisahan
# ---------------------------------------------------------------------------
def regional_gaussian(z, dx, dy, kc_radkm):
    kc = kc_radkm / 1000.0  # rad/m
    return fft_filter(z, dx, dy, lambda K: np.exp(-(K ** 2) / (2 * kc ** 2)))


def regional_moving_average(z, N):
    return ndimage.uniform_filter(z, size=N, mode="reflect")


def regional_polynomial(z, order):
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x = (xx.ravel() - nx / 2) / nx
    y = (yy.ravel() - ny / 2) / ny
    cols = [x ** i * y ** j for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    return (A @ coef).reshape(z.shape)


def regional_upward(z, dx, dy, h_m):
    return fft_filter(z, dx, dy, lambda K: np.exp(-K * h_m))


# ---------------------------------------------------------------------------
# 5. Output
# ---------------------------------------------------------------------------
def write_geotiff(path, z, lon, lat, nodata=-9999.0, mask=None):
    arr = z.astype(np.float32).copy()
    if mask is not None:
        arr[mask] = nodata
    dlon = np.median(np.diff(lon))
    dlat = np.median(np.abs(np.diff(lat)))
    west, north = lon[0] - dlon / 2, lat[0] + dlat / 2   # sudut kiri-atas piksel
    try:
        import rasterio
        from rasterio.transform import from_origin
        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype="float32", crs="EPSG:4326", nodata=nodata,
                           transform=from_origin(west, north, dlon, dlat)) as dst:
            dst.write(arr, 1)
        return
    except ImportError:
        pass
    import tifffile
    geokeys = [1, 1, 0, 3,            # header: versi, revisi, minor, jumlah key
               1024, 0, 1, 2,         # GTModelType = Geographic
               1025, 0, 1, 1,         # GTRasterType = PixelIsArea
               2048, 0, 1, 4326]      # GeographicType = WGS84
    extratags = [
        (33550, "d", 3, (dlon, dlat, 0.0), False),                 # ModelPixelScale
        (33922, "d", 6, (0.0, 0.0, 0.0, west, north, 0.0), False),  # ModelTiepoint
        (34735, "H", len(geokeys), tuple(geokeys), False),          # GeoKeyDirectory
        (42113, "s", 0, f"{nodata:g}", False),                      # GDAL_NODATA
    ]
    tifffile.imwrite(path, arr, extratags=extratags, photometric="minisblack")


def plot_spectrum(path, k, lnA, segs, kc, title):
    colors = dict(regional="tab:red", residual="tab:blue", noise="tab:gray")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(k, lnA, "ko", ms=4, label="ln A (rata-rata radial)")
    for nm, s in segs.items():
        kk = np.linspace(s["kmin"], s["kmax"], 20)
        ax.plot(kk, s["slope"] * kk + s["intercept"], "-", lw=2, color=colors.get(nm, "tab:green"),
                label=f"{nm}: h = {s['depth_km']:.2f} km (R²={s['r2']:.2f})")
    if kc is not None:
        ax.axvline(kc, ls="--", color="k", lw=1)
        ax.text(kc, ax.get_ylim()[1], f"  kc = {kc:.3f} rad/km", va="top", fontsize=9)
    ax.set_xlabel("Bilangan gelombang k (rad/km)")
    ax.set_ylabel("ln A")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_maps(path, lon, lat, grids, titles):
    ext = [lon[0], lon[-1], lat[-1], lat[0]]
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8))
    axes = np.atleast_1d(axes)
    for ax, g, t in zip(axes, grids, titles):
        im = ax.imshow(g, extent=ext, origin="upper", cmap="jet", aspect="equal")
        ax.contour(lon, lat, g, levels=12, colors="k", linewidths=0.4)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("Bujur (°)")
        ax.set_ylabel("Lintang (°)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="mGal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Pemisahan regional-residual + analisis spektrum")
    ap.add_argument("csv", nargs="?", help="file CSV input (kosongkan untuk memilih saat dijalankan)")
    ap.add_argument("-o", "--outdir", default="hasil_pemisahan")
    ap.add_argument("--x", default="lon")
    ap.add_argument("--y", default="lat")
    ap.add_argument("--z", default="CBA_mGal")
    ap.add_argument("--method", default="all",
                    choices=["gaussian", "moving_average", "polynomial", "upward", "all"])
    ap.add_argument("--breaks", nargs=2, type=float, metavar=("K1", "K2"),
                    help="batas segmen manual (rad/km): regional|residual dan residual|noise")
    ap.add_argument("--kc", type=float, help="paksa cut-off (rad/km), menimpa hasil spektrum")
    ap.add_argument("--window", type=int, help="paksa lebar jendela moving average (titik grid)")
    ap.add_argument("--poly-order", type=int, default=2)
    ap.add_argument("--up-height", type=float, help="tinggi kontinuasi ke atas (m); default = 1/kc")
    ap.add_argument("--no-denoise", action="store_true",
                    help="jangan buang noise frekuensi tinggi (k > k_noise) dari residual")
    ap.add_argument("--primary", default="gaussian",
                    choices=["gaussian", "moving_average", "polynomial", "upward"],
                    help="metode utama untuk analisis spektrum residual (saat --method all)")
    a = ap.parse_args()
    if not a.csv:
        a.csv = choose_input_file()
    a.z = choose_column(a.csv, a.z)
    if a.outdir == "hasil_pemisahan":   # default -> folder di samping file input
        base = os.path.splitext(os.path.basename(a.csv))[0]
        a.outdir = os.path.join(os.path.dirname(os.path.abspath(a.csv)), f"hasil_{base}")
    os.makedirs(a.outdir, exist_ok=True)

    lon, lat, z, filled, info = load_grid(a.csv, a.x, a.y, a.z)
    ny, nx = z.shape
    dx, dy = info["dx"], info["dy"]
    print(f"Grid: {nx} x {ny} titik, dx={dx:.1f} m, dy={dy:.1f} m "
          f"({nx*dx/1000:.1f} x {ny*dy/1000:.1f} km)")
    print(f"Titik dipotong (baris/kolom kosong): {info['n_cut']}, NaN internal diisi: {info['n_filled']}")

    # ---- spektrum CBA
    k, lnA, cnt = radial_spectrum(z, dx, dy)
    segs, kc = fit_segments(k, lnA, breaks=a.breaks)
    if a.kc:
        kc = a.kc
    dxy = 0.5 * (dx + dy) / 1000.0                       # km
    N = a.window or int(round(2 * np.pi / (kc * dxy)))
    N = max(3, N + (1 - N % 2))                          # ganjil
    lam_c = 2 * np.pi / kc
    pd.DataFrame({"k_rad_per_km": k, "k_cycle_per_km": k / (2 * np.pi),
                  "wavelength_km": 2 * np.pi / k, "lnA": lnA, "n_bin": cnt}).to_csv(
        os.path.join(a.outdir, "spektrum_CBA.csv"), index=False)
    plot_spectrum(os.path.join(a.outdir, "spektrum_CBA.png"), k, lnA, segs, kc,
                  "Analisis Spektrum CBA (radially averaged)")

    # ---- filter noise (low-pass Gaussian pada awal segmen noise) -> residual = band-pass
    k_noise = segs["noise"]["kmin"]
    if a.no_denoise:
        zf = z.copy()
    else:
        zf = regional_gaussian(z, dx, dy, k_noise)

    # ---- pemisahan
    methods = ["gaussian", "moving_average", "polynomial", "upward"] if a.method == "all" else [a.method]
    up_h = a.up_height if a.up_height else 1000.0 / kc   # m (k dalam rad/km)
    results = {}
    for m in methods:
        if m == "gaussian":
            reg = regional_gaussian(z, dx, dy, kc)
        elif m == "moving_average":
            reg = regional_moving_average(z, N)
        elif m == "polynomial":
            reg = regional_polynomial(z, a.poly_order)
        elif m == "upward":
            reg = regional_upward(z, dx, dy, up_h)
        results[m] = (reg, zf - reg)

    # ---- output CSV titik
    LON, LAT = np.meshgrid(lon, lat)
    out = pd.DataFrame({"lon": LON.ravel(), "lat": LAT.ravel(), a.z: z.ravel(),
                        f"{a.z}_denoised": zf.ravel(),
                        "interpolated": filled.ravel().astype(int)})
    for m, (reg, res) in results.items():
        out[f"regional_{m}"] = reg.ravel()
        out[f"residual_{m}"] = res.ravel()
    out = out.sort_values(["lat", "lon"]).reset_index(drop=True)
    out.to_csv(os.path.join(a.outdir, "regional_residual.csv"), index=False, float_format="%.6f")

    # ---- output GeoTIFF + peta
    write_geotiff(os.path.join(a.outdir, "CBA.tif"), z, lon, lat, mask=filled)
    write_geotiff(os.path.join(a.outdir, "CBA_denoised.tif"), zf, lon, lat, mask=filled)
    for m, (reg, res) in results.items():
        write_geotiff(os.path.join(a.outdir, f"regional_{m}.tif"), reg, lon, lat, mask=filled)
        write_geotiff(os.path.join(a.outdir, f"residual_{m}.tif"), res, lon, lat, mask=filled)
        plot_maps(os.path.join(a.outdir, f"peta_{m}.png"), lon, lat, [z, reg, res],
                  ["Complete Bouguer Anomaly", f"Regional ({m})", f"Residual ({m})"])

    # ---- spektrum residual (metode utama)
    prim = a.primary if a.method == "all" else a.method
    reg_p = results[prim][0]
    res_raw = z - reg_p                     # residual sebelum filter noise (untuk kontrol spektrum)
    kr, lnAr, cntr = radial_spectrum(res_raw, dx, dy, detrend="mean")
    # Spektrum grid residual hanya untuk pemeriksaan visual (energi < kc harus tertekan).
    # Gradiennya sudah dibentuk oleh fungsi filter -> TIDAK dipakai untuk kedalaman.
    # Kedalaman residual diambil dari segmen residual pada spektrum CBA.
    plot_spectrum(os.path.join(a.outdir, f"spektrum_residual_{prim}.png"), kr, lnAr,
                  {}, kc, f"Spektrum Grid Residual ({prim}) - pemeriksaan visual")
    pd.DataFrame({"k_rad_per_km": kr, "wavelength_km": 2 * np.pi / kr, "lnA": lnAr,
                  "n_bin": cntr}).to_csv(os.path.join(a.outdir, f"spektrum_residual_{prim}.csv"),
                                         index=False)

    # ---- ringkasan
    lines = [
        "RINGKASAN ANALISIS SPEKTRUM & PEMISAHAN REGIONAL-RESIDUAL",
        "=" * 60,
        f"File input      : {os.path.basename(a.csv)}  (kolom {a.z})",
        f"Grid            : {nx} x {ny}, dx = {dx:.1f} m, dy = {dy:.1f} m",
        f"Luas area       : {nx*dx/1000:.2f} km x {ny*dy/1000:.2f} km",
        "",
        "Spektrum CBA (ln A vs k, k dalam rad/km; kedalaman = -gradien):",
    ]
    for nm, s in segs.items():
        lines.append(f"  {nm:9s}: k {s['kmin']:.3f}-{s['kmax']:.3f} rad/km, n={s['npts']:2d}, "
                     f"gradien {s['slope']:8.3f}, kedalaman {s['depth_km']:.3f} km, R² {s['r2']:.3f}")
    lines += [
        f"  Cut-off kc     : {kc:.4f} rad/km  (lambda_c = {lam_c:.2f} km)",
        f"  Lebar jendela MA: N = {N} titik (~{N*dxy:.2f} km)",
        "",
        f"  k_noise        : {k_noise:.4f} rad/km (lambda = {2*np.pi/k_noise:.2f} km)"
        + ("  [noise TIDAK dibuang]" if a.no_denoise else "  [low-pass noise diterapkan]"),
        "",
        "Parameter pemisahan:",
        f"  gaussian       : sigma_k = kc = {kc:.4f} rad/km",
        f"  moving_average : N = {N}",
        f"  polynomial     : orde {a.poly_order}",
        f"  upward         : h = {up_h:.0f} m",
        "",
        "KEDALAMAN (dari spektrum CBA):",
        f"  sumber regional  : {segs['regional']['depth_km']:.3f} km",
        f"  sumber residual  : {segs['residual']['depth_km']:.3f} km",

        "",
        "Statistik grid (mGal): min / maks / rata-rata / std",
        f"  CBA            : {z.min():8.3f} {z.max():8.3f} {z.mean():8.3f} {z.std():7.3f}",
    ]
    for m, (reg, rs) in results.items():
        lines.append(f"  regional_{m:14s}: {reg.min():8.3f} {reg.max():8.3f} {reg.mean():8.3f} {reg.std():7.3f}")
        lines.append(f"  residual_{m:14s}: {rs.min():8.3f} {rs.max():8.3f} {rs.mean():8.3f} {rs.std():7.3f}")
    lines += ["",
              "Catatan: kedalaman dihitung relatif terhadap bidang pengamatan (permukaan topografi).",
              "Kedalaman regional dibatasi ukuran area: kedalaman yang dapat dipercaya kira-kira",
              f"< ~1/3 sisi terpendek area (~{min(nx*dx, ny*dy)/3000:.1f} km)."]
    txt = "\n".join(lines)
    with open(os.path.join(a.outdir, "ringkasan_kedalaman.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
