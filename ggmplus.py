"""
ggmplus.py
==========

Python port of ggmplus2013_v4.m (Christian Hirt, WA Centre for Geodesy,
Curtin University, Perth, 2013-05-30).

Purpose: seamless extraction of GGMplus gravity field functionals from the
GGMplus binary tiles (5 x 5 deg, 2500 x 2500 cells, 7.2 arc-sec, big-endian).

Main function
-------------
    X, Y, Z = ggmplus2013_v4(functional, minlon1, maxlon1, minlat1, maxlat1,
                             facX, facY, path_basis, nodataval, method)

Inputs
    functional  'geoid', 'gravity' (or 'grav_disturbance'),
                'acceleration' (or 'grav_acceleration'), 'xi', 'eta'
    minlon1, maxlon1, minlat1, maxlat1
                geodetic coordinates (deg) of the selection area
    facX, facY  resolution scaling factors applied to the 7.2" base grid
                  = 1 : data returned at 7.2" without interpolation
                  > 1 : coarser output grid (e.g. 2 -> 14.4"), interpolated
                  < 1 : finer output grid (e.g. 0.5 -> 3.6"), interpolated
                (The MATLAB header has these two descriptions swapped; the
                code itself uses output spacing = 7.2" * fac.)
    path_basis  folder containing the GGMplus sub-folders
                (geoid, dg, ga, xi, eta)
    nodataval   value assigned where no data exist (offshore, |lat| > 60,
                missing tiles). NaN is recommended for plotting.
    method      'nearest', 'linear', 'cubic' or 'spline'. Ignored when
                facX = facY = 1.

Outputs
    X, Y        meshgrid-style 2-D arrays of longitude / latitude (deg)
    Z           2-D array of the functional in its basic unit
                (m, mGal or arc-sec). Row 0 is the southernmost row.

Notes
    * With facX = facY = 1 the cell centres are returned as stored, so X/Y do
      not necessarily match the requested box exactly.
    * With interpolation the output is a line-registered grid starting
      exactly at minlon1/minlat1.

Requires numpy; scipy only for method='spline'.
"""

import math
import os

import numpy as np

__all__ = ["ggmplus2013_v4", "interp2_regular", "tile_filename"]

# ---------------------------------------------------------------------------
# 1 - constants valid for all GGMplus grids
# ---------------------------------------------------------------------------
TILESIZE = 5                 # deg
RESOLUTION = 7.2 / 3600.0    # deg
TILEELEMS = 2500             # elements per tile in lat and lon direction

_FAC = 3600 * 100            # integer coordinates in units of 0.01 arc-sec
_R = 720                     # resolution in 0.01 arc-sec (7.2")
_R2 = 360                    # half resolution

# functional: (conversion factor, file suffix, sub-folder, big-endian dtype)
_FUNCTIONALS = {
    "geoid":             (1000, ".ha",  "geoid", ">i4"),   # mm      -> m
    "gravity":           (10,   ".dg",  "dg",    ">i2"),   # 0.1 mGal -> mGal
    "grav_disturbance":  (10,   ".dg",  "dg",    ">i2"),
    "acceleration":      (10,   ".ga",  "ga",    ">i4"),   # 0.1 mGal -> mGal
    "grav_acceleration": (10,   ".ga",  "ga",    ">i4"),
    "xi":                (10,   ".xi",  "xi",    ">i2"),   # 0.1"     -> "
    "eta":               (10,   ".eta", "eta",   ">i2"),
}


def _mround(x):
    """Round half away from zero (MATLAB round / int32 conversion)."""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def _colon(a, step, b):
    """Equivalent of MATLAB's a:step:b for floats."""
    n = int(math.floor((b - a) / step + 1e-10)) + 1
    if n < 1:
        return np.empty(0)
    return a + np.arange(n) * step


def tile_filename(lat, lon):
    """GGMplus tile name (without suffix) for the SW corner lat/lon,
    e.g. (45, 10) -> 'N45E010', (-5, -75) -> 'S05W075'."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(int(lat)):02d}{ew}{abs(int(lon)):03d}"


# ---------------------------------------------------------------------------
# Interpolation on a regular grid (MATLAB interp2 behaviour)
# ---------------------------------------------------------------------------
def _keys_weights(t):
    """Cubic convolution weights (Keys, a = -0.5) for offsets -1, 0, 1, 2.
    This is the kernel MATLAB's interp2 uses for 'cubic' on uniform grids."""
    a = -0.5

    def w_near(s):          # |s| <= 1
        return (a + 2) * s**3 - (a + 3) * s**2 + 1

    def w_far(s):           # 1 < |s| < 2
        return a * s**3 - 5 * a * s**2 + 8 * a * s - 4 * a

    return [w_far(1 + t), w_near(t), w_near(1 - t), w_far(2 - t)]


def _pad_keys(Z):
    """Add one row/column on each side using Keys' boundary condition
    f(-1) = 3 f(0) - 3 f(1) + f(2)."""
    def pad_axis(M, axis):
        M = np.moveaxis(M, axis, 0)
        lo = 3 * M[0] - 3 * M[1] + M[2]
        hi = 3 * M[-1] - 3 * M[-2] + M[-3]
        out = np.concatenate([lo[None], M, hi[None]], axis=0)
        return np.moveaxis(out, 0, axis)

    return pad_axis(pad_axis(Z, 0), 1)


def _accumulate(terms):
    """Sum w*v, ignoring terms whose weight is exactly zero (so that a NaN
    neighbour does not contaminate a query lying exactly on a valid node)."""
    out = 0.0
    for w, v in terms:
        out = out + np.where(w == 0, 0.0, w * v)
    return out


def interp2_regular(xvec, yvec, Z, xq, yq, method="linear",
                    chunk=2_000_000):
    """
    Interpolate Z (shape len(yvec) x len(xvec)) given on the uniformly
    spaced, ascending vectors xvec / yvec at query points xq, yq.

    xvec, yvec may also be passed as 2-D meshgrid arrays (as in MATLAB's
    interp2(X, Y, Z, xq, yq)). NaN in Z propagates only locally, and
    queries outside the grid return NaN, as in MATLAB.
    """
    xvec = np.asarray(xvec, float)
    yvec = np.asarray(yvec, float)
    if xvec.ndim == 2:
        xvec = xvec[0, :]
    if yvec.ndim == 2:
        yvec = yvec[:, 0]
    Z = np.asarray(Z, float)
    xq = np.asarray(xq, float)
    yq = np.asarray(yq, float)
    shape = np.broadcast(xq, yq).shape
    xq = np.broadcast_to(xq, shape).ravel()
    yq = np.broadcast_to(yq, shape).ravel()

    ny, nx = Z.shape
    method = method.lower()

    if method == "spline":
        if np.isnan(Z).any():
            raise ValueError(
                "method='spline' cannot handle no-data (NaN) cells; "
                "use 'cubic' or 'linear' for areas with no-data cells")
        from scipy.interpolate import RectBivariateSpline
        spl = RectBivariateSpline(yvec, xvec, Z, kx=3, ky=3, s=0)
        out = spl.ev(yq, xq)
        outside = ((xq < xvec[0]) | (xq > xvec[-1]) |
                   (yq < yvec[0]) | (yq > yvec[-1]))
        out[outside] = np.nan
        return out.reshape(shape)

    if method not in ("nearest", "linear", "cubic"):
        raise ValueError(f"unknown interpolation method '{method}'")

    dx = (xvec[-1] - xvec[0]) / (nx - 1) if nx > 1 else 1.0
    dy = (yvec[-1] - yvec[0]) / (ny - 1) if ny > 1 else 1.0
    Zp = _pad_keys(Z) if method == "cubic" else None

    out = np.full(xq.shape, np.nan)
    eps = 1e-9
    for s in range(0, xq.size, chunk):
        fx = (xq[s:s + chunk] - xvec[0]) / dx
        fy = (yq[s:s + chunk] - yvec[0]) / dy
        inside = ((fx >= -eps) & (fx <= nx - 1 + eps) &
                  (fy >= -eps) & (fy <= ny - 1 + eps))
        fx = np.clip(fx[inside], 0, nx - 1)
        fy = np.clip(fy[inside], 0, ny - 1)

        if method == "nearest":
            ix = np.clip(np.floor(fx + 0.5).astype(int), 0, nx - 1)
            iy = np.clip(np.floor(fy + 0.5).astype(int), 0, ny - 1)
            val = Z[iy, ix]
        else:
            ix = np.clip(np.floor(fx).astype(int), 0, max(nx - 2, 0))
            iy = np.clip(np.floor(fy).astype(int), 0, max(ny - 2, 0))
            tx = fx - ix
            ty = fy - iy
            if method == "linear":
                wx = [1 - tx, tx]
                wy = [1 - ty, ty]
                terms = [(wy[j] * wx[i],
                          Z[np.minimum(iy + j, ny - 1),
                            np.minimum(ix + i, nx - 1)])
                         for j in range(2) for i in range(2)]
            else:  # cubic convolution, padded array -> offset +1
                wx = _keys_weights(tx)
                wy = _keys_weights(ty)
                terms = [(wy[j] * wx[i], Zp[iy + j, ix + i])
                         for j in range(4) for i in range(4)]
            val = _accumulate(terms)

        block = out[s:s + chunk]
        block[inside] = val
        out[s:s + chunk] = block

    return out.reshape(shape)


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------
def ggmplus2013_v4(functional, minlon1, maxlon1, minlat1, maxlat1,
                   facX=1.0, facY=1.0, path_basis=".", nodataval=np.nan,
                   method="cubic", verbose=True):
    """Extract a GGMplus functional over a lat/lon box. See module docstring."""
    log = print if verbose else (lambda *a, **k: None)

    # constants that differ between functionals -----------------------------
    try:
        conv_factor, suffix, folder, dtype = _FUNCTIONALS[functional]
    except KeyError:
        raise ValueError(f"unknown functional '{functional}'; choose from "
                         f"{sorted(_FUNCTIONALS)}") from None
    path_tiles = os.path.join(path_basis, folder)
    init_nodata = np.iinfo(np.dtype(dtype)).min      # -2^15 or -2^31

    # fac = 1: no interpolation; otherwise extend the area by 0.1 deg -------
    interpflag = not (facX == 1 and facY == 1)
    tolarea = 0.1 if interpflag else 0.0

    # 2 - extended selection area ------------------------------------------
    minlon2 = minlon1 - tolarea
    maxlon2 = maxlon1 + tolarea
    minlat2 = minlat1 - tolarea
    maxlat2 = maxlat1 + tolarea

    # 5 - geometry of the target matrix (integers, 0.01", cell centred) ----
    # NB: in MATLAB, floor(double/int32) is evaluated in int32 arithmetic,
    # i.e. it rounds. _mround reproduces that so results match the original.
    amin_lat = _mround(minlat2 * _FAC / _R) * _R + _R2
    amax_lat = _mround(maxlat2 * _FAC / _R) * _R - _R2
    amin_lon = _mround(minlon2 * _FAC / _R) * _R + _R2
    amax_lon = _mround(maxlon2 * _FAC / _R) * _R - _R2

    avec_lat = np.arange(amin_lat, amax_lat + 1, _R, dtype=np.int64)
    avec_lon = np.arange(amin_lon, amax_lon + 1, _R, dtype=np.int64)
    if avec_lat.size == 0 or avec_lon.size == 0:
        raise ValueError("selection area is smaller than one GGMplus cell")

    A = np.full((avec_lat.size, avec_lon.size), np.nan)

    # 2/3/4 - tiles to be accessed -----------------------------------------
    tlon0 = math.floor(minlon2 / TILESIZE) * TILESIZE
    tlon1 = math.floor(maxlon2 / TILESIZE) * TILESIZE
    tlat0 = math.floor(minlat2 / TILESIZE) * TILESIZE
    tlat1 = math.floor(maxlat2 / TILESIZE) * TILESIZE

    tiles = []
    for tlon in range(tlon0, tlon1 + 1, TILESIZE):
        if tlon < -180 - TILESIZE or tlon > 180:
            continue
        for tlat in range(tlat0, tlat1 + 1, TILESIZE):
            if tlat < -90 or tlat > 90 - TILESIZE:
                continue
            # longitude wrap-around for the file name only
            flon = tlon + 360 if tlon < -180 else tlon
            flon = flon - 360 if flon > 180 - TILESIZE else flon
            tiles.append((tlat, tlon, tile_filename(tlat, flon) + suffix))

    log(" tiles to be accessed")
    for _, _, name in tiles:
        log(name)

    # 6 - fill the matrix with data from the binary files -----------------
    for tlat, tlon, name in tiles:
        s0_lat = tlat * _FAC + _R2                 # first cell centre
        s0_lon = tlon * _FAC + _R2
        s1_lat = s0_lat + (TILEELEMS - 1) * _R     # last cell centre
        s1_lon = s0_lon + (TILEELEMS - 1) * _R

        lat_a, lat_b = max(s0_lat, amin_lat), min(s1_lat, amax_lat)
        lon_a, lon_b = max(s0_lon, amin_lon), min(s1_lon, amax_lon)
        if lat_a > lat_b or lon_a > lon_b:
            log(" this tile does not intersect with target area")
            continue

        # destination / source index ranges (inclusive -> python slices)
        A_lat = slice((lat_a - amin_lat) // _R, (lat_b - amin_lat) // _R + 1)
        A_lon = slice((lon_a - amin_lon) // _R, (lon_b - amin_lon) // _R + 1)
        S_lat = slice((lat_a - s0_lat) // _R, (lat_b - s0_lat) // _R + 1)
        S_lon = slice((lon_a - s0_lon) // _R, (lon_b - s0_lon) // _R + 1)

        fname = os.path.join(path_tiles, name)
        log("read file")
        log(fname)
        if not os.path.isfile(fname):
            log(f"file {fname} does not exist")
            continue

        S = np.fromfile(fname, dtype=dtype, count=TILEELEMS * TILEELEMS)
        if S.size != TILEELEMS * TILEELEMS:
            raise IOError(f"{fname}: expected {TILEELEMS**2} values, "
                          f"got {S.size}")
        # MATLAB fread + reshape fills column-major: rows = latitude
        # (south -> north), columns = longitude (west -> east)
        S = S.reshape((TILEELEMS, TILEELEMS), order="F")
        S = S[S_lat, S_lon].astype(np.float64)
        S[S < init_nodata + 1] = np.nan             # no-data -> NaN
        A[A_lat, A_lon] = S
        log("read complete")

    log(f" {np.isnan(A).sum()} no-data cells in extracted grid")

    # 7 - X and Y in degrees -----------------------------------------------
    avec_lon_deg = avec_lon / _FAC
    avec_lat_deg = avec_lat / _FAC

    # 8 - interpolate only if fac != 1 -------------------------------------
    if interpflag:
        log(f" interpolate {method}")
        bvec_lon = _colon(minlon1, RESOLUTION * facX, maxlon1)
        bvec_lat = _colon(minlat1, RESOLUTION * facY, maxlat1)
        X, Y = np.meshgrid(bvec_lon, bvec_lat)
        Z = interp2_regular(avec_lon_deg, avec_lat_deg, A, X, Y, method)
        log(" interpolation finished")
    else:
        log(" pass extracted data directly.")
        X, Y = np.meshgrid(avec_lon_deg, avec_lat_deg)
        Z = A

    # 9 - scale to basic unit; 10 - assign user no-data value -------------
    Z = Z / conv_factor
    Z[np.isnan(Z)] = nodataval
    return X, Y, Z
