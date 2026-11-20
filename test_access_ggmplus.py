"""
Test driver for GGMplus access and use (Python port of test_access_ggmplus.m)

Requires ggmplus.py (port of ggmplus2013_v4.m) in the same folder.

Original MATLAB version: Christian Hirt and Sten Claessens,
Western Australian Centre for Geodesy, Curtin University, Perth, 2013-05-30
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from ggmplus import ggmplus2013_v4, interp2_regular

# ------------------------------------------------------------------------
#    INPUT SELECTION
# ------------------------------------------------------------------------

# Path where the GGMplus folders (dg, ga, xi, eta, geoid) are located ------
pathGGMplus = r"D:\OneDrive - UNIVERSITAS INDONESIA\modul kuliah\Mata Kuliah Analisis Data Geofisika 2\Data_sample_Gravity_GGMplus"

# No-data value for points where GGMplus is unavailable -------------------
ndv = np.nan   # NaN gives the best results for plotting

# Desired target area ------------------------------------------------------
minlon1 = 110.0   # western edge  (negative for western hemisphere)
maxlon1 = 111.0  # eastern edge  (negative for western hemisphere)
minlat1 = -7.5   # southern edge (negative for southern hemisphere)
maxlat1 = -7.0   # northern edge (negative for southern hemisphere)

# NB: in this example the files N45E010.<suffix> are assumed to exist at
#     <pathGGMplus>/dg/N45E010.dg       (gravity disturbances)
#     <pathGGMplus>/ga/N45E010.ga       (gravity accelerations)
#     <pathGGMplus>/xi/N45E010.xi       (N-S deflections of the vertical)
#     <pathGGMplus>/eta/N45E010.eta     (E-W deflections of the vertical)
#     <pathGGMplus>/geoid/N45E010.ha    (quasigeoid)

# Factor applied to the GGMplus base resolution of 0.002 deg --------------
# facX = facY = 1 extracts the data *without* interpolation.
# facX = facY = 2 downsamples by a factor of 2 using the method below.
facX = 1.0
facY = 1.0
# Interpolation methods: 'nearest', 'linear', 'cubic', 'spline'
method = "cubic"

# Select functionals of interest (comment out those not needed) -----------
results = {}
#results["gravity"] = ggmplus2013_v4("gravity", minlon1, maxlon1, minlat1, maxlat1,
#                                    facX, facY, pathGGMplus, ndv, method)
results["acceleration"] = ggmplus2013_v4("acceleration", minlon1, maxlon1, minlat1, maxlat1,
                                          facX, facY, pathGGMplus, ndv, method)
#results["xi"] = ggmplus2013_v4("xi", minlon1, maxlon1, minlat1, maxlat1,
#                               facX, facY, pathGGMplus, ndv, method)
#results["eta"] = ggmplus2013_v4("eta", minlon1, maxlon1, minlat1, maxlat1,
#                                facX, facY, pathGGMplus, ndv, method)
#results["geoid"] = ggmplus2013_v4("geoid", minlon1, maxlon1, minlat1, maxlat1,
#                                  facX, facY, pathGGMplus, ndv, method)

# Test point (must lie inside the area above): Zugspitze summit, German Alps
test_phi = -7.25   # geodetic latitude
test_lam = 110.5   # longitude
methodtp = "cubic"  # interpolation method for the test point

# ------------------------------------------------------------------------
#    MAP PRODUCTION
# ------------------------------------------------------------------------
titles = {
    "gravity":      ("GGMplus Gravity disturbances (mGal)", 1),
    "acceleration": ("GGMplus Accelerations (mGal)", 2),
    "xi":           ("GGMplus North-South DoV (arc-sec)", 3),
    "eta":          ("GGMplus East-West DoV (arc-sec)", 4),
    "geoid":        ("GGMplus Geoid (m)", 5),
}

for key, (title, fignum) in titles.items():
    if key not in results:
        continue
    x, y, z = results[key]
    lon, lat = x[0, :], y[:, 0]
    dx = lon[1] - lon[0] if lon.size > 1 else 0
    dy = lat[1] - lat[0] if lat.size > 1 else 0
    fig, ax = plt.subplots(num=fignum)
    # equivalent of imagesc(...) with YDir normal; NaN cells are transparent
    im = ax.imshow(z, origin="lower", aspect="equal", cmap="RdBu_r",
                   extent=[lon[0] - dx / 2, lon[-1] + dx / 2,
                           lat[0] - dy / 2, lat[-1] + dy / 2])
    ax.set_xlim(minlon1, maxlon1)
    ax.set_ylim(minlat1, maxlat1)
    ax.set_title(title)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.grid(True)
    fig.colorbar(im, ax=ax)

# ------------------------------------------------------------------------
#    CALCULATION AT A SINGLE POINT OF INTEREST
# ------------------------------------------------------------------------
print("------------------------------------------------")
print(" Interpolated GGMplus functionals at test point")
print("------------------------------------------------")

fmt = {
    "gravity":      "Gravity disturbance:  {:10.4f} mGal",
    "acceleration": "Acceleration:         {:10.2f} mGal",
    "xi":           "North-South DoV:      {:10.4f} arcsec",
    "eta":          "East-West DoV:        {:10.4f} arcsec",
    "geoid":        "Geoid:                {:10.4f} m",
}
for key, f in fmt.items():
    if key in results:
        x, y, z = results[key]
        val = float(interp2_regular(x, y, z, test_lam, test_phi, methodtp))
        print(f.format(val))

# ------------------------------------------------------------------------
#    ASCII OUTPUT
# ------------------------------------------------------------------------
# Writes longitude, latitude and value for one functional to a text file.
# Disabled by default, as large areas give very large files.
# This example writes gravity disturbances to 'gravity.txt'.

outputfile = os.path.join(pathGGMplus, "gravity.txt")
x, y, z = results["acceleration"]
np.savetxt(outputfile, np.column_stack([x.ravel(), y.ravel(), z.ravel()]),
           fmt="%8.3f %7.3f %6.1f")

plt.show()
