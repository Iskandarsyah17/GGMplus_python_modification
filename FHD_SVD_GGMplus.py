#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter turunan untuk anomali residual:
  - FHD  : First Horizontal Derivative (dg/dx, dg/dy, dan total sqrt(gx^2+gy^2))
  - SVD  : Second Vertical Derivative, metode:
           elkins     (Elkins, 1951)          - konvolusi 5x5
           henderson  (Henderson & Zietz, 1949) - konvolusi 5x5
           rosenbach  (Rosenbach, 1953)       - konvolusi 5x5
           fft        (domain frekuensi, |k|^2)
           all        (semua metode di atas)

Input  : CSV (lon, lat, kolom residual) hasil pemisahan_regional_residual.py
         ATAU GeoTIFF residual (mis. residual_gaussian.tif).
         Jika file tidak diberikan, script akan meminta Anda memilih file.
Output : fhd_svd.csv, GeoTIFF tiap hasil, peta PNG, ringkasan.txt

Satuan : FHD dalam mGal/km, SVD dalam mGal/km^2.
Konvensi tanda SVD: positif di atas massa berlebih (densitas tinggi);
kontur SVD = 0 menandai batas/kontak struktur (mis. sesar).

Contoh
------
python fhd_svd.py
python fhd_svd.py hasil_CBA_ggmplus/regional_residual.csv --z residual_gaussian
python fhd_svd.py hasil_CBA_ggmplus/residual_gaussian.tif --svd elkins
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

R_EARTH = 6371008.8  # m

# ---------------------------------------------------------------------------
# Kernel SVD (spasi grid seragam). Hasil konvolusi dibagi dx^2.
# ---------------------------------------------------------------------------
KERNELS = {
    "elkins": np.array([
        [0.0000, -0.0833, 0.0000, -0.0833, 0.0000],
        [-0.0833, -0.0667, -0.0334, -0.0667, -0.0833],
        [0.0000, -0.0334, 1.0668, -0.0334, 0.0000],
        [-0.0833, -0.0667, -0.0334, -0.0667, -0.0833],
        [0.0000, -0.0833, 0.0000, -0.0833, 0.0000]]),
    "henderson": np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, 4.0, -1.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]]),
    "rosenbach": np.array([
        [0.0000, 0.0416, 0.0000, 0.0416, 0.0000],
        [0.0416, -0.3332, -0.7500, -0.3332, 0.0416],
        [0.0000, -0.7500, 4.0000, -0.7500, 0.0000],
        [0.0416, -0.3332, -0.7500, -0.3332, 0.0416],
        [0.0000, 0.0416, 0.0000, 0.0416, 0.0000]]),
}


# ---------------------------------------------------------------------------
# Pilih file / kolom
# ---------------------------------------------------------------------------
def choose_input_file():
    path = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Pilih file residual (CSV atau GeoTIFF)",
            filetypes=[("CSV / GeoTIFF", "*.csv *.tif *.tiff"), ("Semua file", "*.*")])
        root.destroy()
    except Exception:
        path = None
    while not path or not os.path.isfile(path):
        if path:
            print(f"File tidak ditemukan: {path}")
        path = input("Masukkan path file residual (CSV/TIF): ").strip().strip('"').strip("'")
    print(f"File input: {path}")
    return path


def choose_column(cols, zcol):
    if zcol and zcol in cols:
        return zcol
    cand = [c for c in cols if c.lower().startswith("residual")]
    if not zcol and len(cand) == 1:
        return cand[0]
    if zcol:
        print(f"Kolom '{zcol}' tidak ditemukan.")
    print("Kolom tersedia:")
    for i, c in enumerate(cols):
        print(f"  [{i}] {c}")
    while True:
        ans = input("Pilih nomor atau nama kolom residual: ").strip()
        if ans.isdigit() and int(ans) < len(cols):
            return cols[int(ans)]
        if ans in cols:
            return ans


# ---------------------------------------------------------------------------
# Baca data -> grid (baris 0 = utara)
# ---------------------------------------------------------------------------
def read_csv_grid(path, xcol, ycol, zcol):
    cols = list(pd.read_csv(path, nrows=0).columns)
    zcol = choose_column(cols, zcol)
    df = pd.read_csv(path, usecols=[xcol, ycol, zcol])
    df[xcol] = df[xcol].round(8)
    df[ycol] = df[ycol].round(8)
    g = df.pivot_table(index=ycol, columns=xcol, values=zcol, aggfunc="mean", dropna=False)
    g = g.sort_index(ascending=False)
    g = g.reindex(sorted(g.columns), axis=1)
    return g.columns.values.astype(float), g.index.values.astype(float), g.values.astype(float), zcol


def read_tif_grid(path):
    try:
        import rasterio
        with rasterio.open(path) as src:
            z = src.read(1).astype(float)
            if src.nodata is not None:
                z[z == src.nodata] = np.nan
            t = src.transform
            lon = t.c + t.a * (np.arange(src.width) + 0.5)
            lat = t.f + t.e * (np.arange(src.height) + 0.5)
    except ImportError:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            p = tf.pages[0]
            z = p.asarray().astype(float)
            sx, sy, _ = p.tags["ModelPixelScaleTag"].value
            tie = p.tags["ModelTiepointTag"].value
            west, north = tie[3], tie[4]
            if 42113 in p.tags:
                try:
                    nd = float(str(p.tags[42113].value).strip("\x00 "))
                    z[z == nd] = np.nan
                except ValueError:
                    pass
        lon = west + sx * (np.arange(z.shape[1]) + 0.5)
        lat = north - sy * (np.arange(z.shape[0]) + 0.5)
    name = os.path.splitext(os.path.basename(path))[0]
    return lon, lat, z, name


def prepare(z):
    """Potong baris/kolom kosong; isi NaN internal (nearest)."""
    rows = ~np.all(np.isnan(z), axis=1)
    cols = ~np.all(np.isnan(z), axis=0)
    r = np.where(rows)[0]
    c = np.where(cols)[0]
    sl = (slice(r[0], r[-1] + 1), slice(c[0], c[-1] + 1))
    z = z[sl]
    nanmask = np.isnan(z)
    if nanmask.any():
        idx = ndimage.distance_transform_edt(nanmask, return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    return z, nanmask, sl


# ---------------------------------------------------------------------------
# Turunan
# ---------------------------------------------------------------------------
def fhd(z, dx_km, dy_km):
    """dg/dx (timur), dg/dy (utara), total. Baris 0 = utara -> tanda dy dibalik."""
    gy_row, gx = np.gradient(z, dy_km, dx_km)
    gy = -gy_row
    return gx, gy, np.hypot(gx, gy)


def svd_kernel(z, method, dx_km, dy_km):
    k = KERNELS[method]
    out = ndimage.convolve(z, k, mode="reflect")
    return out / (dx_km * dy_km)


def svd_fft(z, dx_km, dy_km):
    ny, nx = z.shape
    NY = int(2 ** np.ceil(np.log2(2 * ny)))
    NX = int(2 ** np.ceil(np.log2(2 * nx)))
    py, px = NY - ny, NX - nx
    t, l = py // 2, px // 2
    m = z.mean()
    zp = np.pad(z - m, ((t, py - t), (l, px - l)), mode="symmetric")
    kx = 2 * np.pi * np.fft.fftfreq(NX, d=dx_km)
    ky = 2 * np.pi * np.fft.fftfreq(NY, d=dy_km)
    KX, KY = np.meshgrid(kx, ky)
    out = np.real(np.fft.ifft2(np.fft.fft2(zp) * (KX ** 2 + KY ** 2)))
    return out[t:t + ny, l:l + nx]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_geotiff(path, z, lon, lat, mask=None, nodata=-9999.0):
    arr = z.astype(np.float32).copy()
    if mask is not None:
        arr[mask] = nodata
    dlon = np.median(np.diff(lon))
    dlat = np.median(np.abs(np.diff(lat)))
    west, north = lon[0] - dlon / 2, lat[0] + dlat / 2
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
    geokeys = [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326]
    extratags = [
        (33550, "d", 3, (dlon, dlat, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, west, north, 0.0), False),
        (34735, "H", len(geokeys), tuple(geokeys), False),
        (42113, "s", 0, f"{nodata:g}", False),
    ]
    tifffile.imwrite(path, arr, extratags=extratags, photometric="minisblack")


def _sym_limits(g, pct=98):
    v = np.nanpercentile(np.abs(g), pct)
    return -v, v


def plot_map(ax, fig, lon, lat, g, title, unit, cmap="jet", symmetric=False, zero_contour=None):
    ext = [lon[0], lon[-1], lat[-1], lat[0]]
    if symmetric:
        vmin, vmax = _sym_limits(g)
    else:
        vmin, vmax = np.nanpercentile(g, [1, 99])
    im = ax.imshow(g, extent=ext, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    if zero_contour is not None:
        ax.contour(lon, lat, zero_contour, levels=[0], colors="k", linewidths=0.8)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.set_xlabel("Bujur (°)")
    ax.set_ylabel("Lintang (°)")
    fig.colorbar(im, ax=ax, shrink=0.8, label=unit)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="FHD & SVD untuk anomali residual")
    ap.add_argument("input", nargs="?", help="CSV atau GeoTIFF residual (kosongkan untuk memilih)")
    ap.add_argument("-o", "--outdir", help="folder output (default: fhd_svd_<nama> di samping input)")
    ap.add_argument("--x", default="lon")
    ap.add_argument("--y", default="lat")
    ap.add_argument("--z", default=None, help="kolom residual (CSV), mis. residual_gaussian")
    ap.add_argument("--svd", default="all", choices=["elkins", "henderson", "rosenbach", "fft", "all"])
    ap.add_argument("--smooth", type=int, default=0,
                    help="lebar moving average sebelum turunan (titik grid, 0 = tanpa)")
    a = ap.parse_args()

    if not a.input:
        a.input = choose_input_file()
    ext = os.path.splitext(a.input)[1].lower()
    if ext in (".tif", ".tiff"):
        lon, lat, z, zname = read_tif_grid(a.input)
    else:
        lon, lat, z, zname = read_csv_grid(a.input, a.x, a.y, a.z)

    z, nanmask, sl = prepare(z)
    lat, lon = lat[sl[0]], lon[sl[1]]
    if a.smooth and a.smooth > 1:
        z = ndimage.uniform_filter(z, size=a.smooth, mode="reflect")

    if not a.outdir:
        base = os.path.splitext(os.path.basename(a.input))[0]
        tag = zname if ext not in (".tif", ".tiff") else base
        a.outdir = os.path.join(os.path.dirname(os.path.abspath(a.input)), f"fhd_svd_{tag}")
    os.makedirs(a.outdir, exist_ok=True)

    dlon = np.median(np.diff(lon))
    dlat = np.median(np.abs(np.diff(lat)))
    dx_km = np.deg2rad(dlon) * R_EARTH * np.cos(np.deg2rad(lat.mean())) / 1000.0
    dy_km = np.deg2rad(dlat) * R_EARTH / 1000.0
    ny, nx = z.shape
    print(f"Data: {zname} | grid {nx} x {ny}, dx = {dx_km*1000:.1f} m, dy = {dy_km*1000:.1f} m")

    # ---- FHD
    gx, gy, fhd_t = fhd(z, dx_km, dy_km)

    # ---- SVD
    methods = ["elkins", "henderson", "rosenbach", "fft"] if a.svd == "all" else [a.svd]
    svd = {}
    for m in methods:
        svd[m] = svd_fft(z, dx_km, dy_km) if m == "fft" else svd_kernel(z, m, dx_km, dy_km)

    # ---- CSV
    LON, LAT = np.meshgrid(lon, lat)
    out = pd.DataFrame({"lon": LON.ravel(), "lat": LAT.ravel(), "residual": z.ravel(),
                        "FHD_x": gx.ravel(), "FHD_y": gy.ravel(), "FHD": fhd_t.ravel()})
    for m, g in svd.items():
        out[f"SVD_{m}"] = g.ravel()
    out = out[~nanmask.ravel()].sort_values(["lat", "lon"]).reset_index(drop=True)
    out.to_csv(os.path.join(a.outdir, "fhd_svd.csv"), index=False, float_format="%.6f")

    # ---- GeoTIFF
    write_geotiff(os.path.join(a.outdir, "FHD.tif"), fhd_t, lon, lat, nanmask)
    write_geotiff(os.path.join(a.outdir, "FHD_x.tif"), gx, lon, lat, nanmask)
    write_geotiff(os.path.join(a.outdir, "FHD_y.tif"), gy, lon, lat, nanmask)
    for m, g in svd.items():
        write_geotiff(os.path.join(a.outdir, f"SVD_{m}.tif"), g, lon, lat, nanmask)

    # ---- peta
    zz = lambda g: np.where(nanmask, np.nan, g)
    main_svd = "elkins" if "elkins" in svd else methods[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    plot_map(axes[0], fig, lon, lat, zz(z), f"Anomali Residual ({zname})", "mGal")
    plot_map(axes[1], fig, lon, lat, zz(fhd_t), "FHD (total) + kontur SVD = 0", "mGal/km",
             cmap="viridis", zero_contour=svd[main_svd])
    plot_map(axes[2], fig, lon, lat, zz(svd[main_svd]), f"SVD ({main_svd}) + kontur 0",
             "mGal/km²", cmap="RdBu_r", symmetric=True, zero_contour=svd[main_svd])
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "peta_residual_FHD_SVD.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    plot_map(axes[0], fig, lon, lat, zz(gx), "FHD arah x (timur)", "mGal/km", cmap="RdBu_r", symmetric=True)
    plot_map(axes[1], fig, lon, lat, zz(gy), "FHD arah y (utara)", "mGal/km", cmap="RdBu_r", symmetric=True)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "peta_FHD_xy.png"), dpi=150)
    plt.close(fig)

    if len(svd) > 1:
        fig, axes = plt.subplots(1, len(svd), figsize=(5.2 * len(svd), 4.8))
        for ax, (m, g) in zip(axes, svd.items()):
            plot_map(ax, fig, lon, lat, zz(g), f"SVD {m}", "mGal/km²", cmap="RdBu_r",
                     symmetric=True, zero_contour=g)
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, "peta_SVD_perbandingan.png"), dpi=150)
        plt.close(fig)

    # ---- ringkasan
    def st(g):
        g = g[~nanmask]
        return f"{g.min():10.4f} {g.max():10.4f} {g.mean():10.4f} {g.std():9.4f}"
    lines = ["RINGKASAN FHD & SVD", "=" * 60,
             f"Input   : {os.path.basename(a.input)}  ({zname})",
             f"Grid    : {nx} x {ny}, dx = {dx_km*1000:.1f} m, dy = {dy_km*1000:.1f} m",
             f"Smoothing sebelum turunan: {a.smooth if a.smooth > 1 else 'tidak'}",
             "", "Statistik: min / maks / rata-rata / std",
             f"  residual (mGal)       : {st(z)}",
             f"  FHD total (mGal/km)   : {st(fhd_t)}",
             f"  FHD_x (mGal/km)       : {st(gx)}",
             f"  FHD_y (mGal/km)       : {st(gy)}"]
    for m, g in svd.items():
        lines.append(f"  SVD_{m:9s} (mGal/km²): {st(g)}")
    if len(svd) > 1:
        lines += ["", "Korelasi antar metode SVD:"]
        keys = list(svd)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                r = np.corrcoef(svd[keys[i]][~nanmask], svd[keys[j]][~nanmask])[0, 1]
                lines.append(f"  {keys[i]} vs {keys[j]}: r = {r:.3f}")
    lines += ["", "Interpretasi singkat:",
              "  - Nilai maksimum FHD menandai batas/kontak densitas (mis. sesar).",
              "  - Kontur SVD = 0 yang berimpit dengan puncak FHD memperkuat indikasi batas struktur.",
              "  - Turunan memperkuat noise; gunakan residual yang sudah difilter atau --smooth."]
    txt = "\n".join(lines)
    with open(os.path.join(a.outdir, "ringkasan.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\nHasil disimpan di: {a.outdir}")


if __name__ == "__main__":
    main()
