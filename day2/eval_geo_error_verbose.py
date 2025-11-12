#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, pandas as pd
from math import radians, sin, cos, atan2, sqrt

R_E = 6378137.0  # mean Earth radius in meters

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance (meters)"""
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return 2 * R_E * atan2(sqrt(a), sqrt(1 - a))

def ll_diff_m(lat1, lon1, lat2, lon2):
    """Approximate local deltaE, deltaN in meters"""
    lat_m = (lat2 - lat1) * (np.pi / 180.0) * R_E
    lon_m = (lon2 - lon1) * (np.pi / 180.0) * R_E * cos(radians((lat1 + lat2) / 2.0))
    return lon_m, lat_m  # (ΔE, ΔN)

# ====== CONFIG ======
PRED_CSV = "predicted_positions.csv"
GT_CSV   = "P2_DATA_TRAIN/P2_DATA_TRAIN_combined.csv"
OUT_ERR_CSV = "per_sample_geo_error.csv"
# =====================

# ---------- Load ----------
pred = pd.read_csv(PRED_CSV)
gt   = pd.read_csv(GT_CSV).rename(columns={"Latitude":"lat","Longitude":"lon","Altitude":"alt"})
df = pred.merge(gt, on="image_file", how="inner")
if df.empty:
    raise RuntimeError("No overlap rows; check image_file names.")

# ---------- Compute error ----------
ΔE_list, ΔN_list, horiz_list = [], [], []
for a, b, c, d in zip(df.pred_lat, df.pred_lon, df.lat, df.lon):
    dE, dN = ll_diff_m(a, b, c, d)
    ΔE_list.append(dE)
    ΔN_list.append(dN)
    horiz_list.append(haversine_m(a, b, c, d))

df["err_E_m"] = np.array(ΔE_list)     # Easting difference (m)
df["err_N_m"] = np.array(ΔN_list)     # Northing difference (m)
df["err_horiz_m"] = np.abs(horiz_list)
df["err_alt_m"]   = (df.pred_alt - df.alt).abs()
df["err_3d_m"]    = np.sqrt(df.err_horiz_m**2 + df.err_alt_m**2)

# ---------- Print per-sample table ----------
print("=== PER-SAMPLE GEO ERROR ===")
for i, r in df.iterrows():
    print(f"{r.image_file:20s} | ΔE={r.err_E_m:8.2f} m | ΔN={r.err_N_m:8.2f} m | horiz={r.err_horiz_m:7.2f} m | alt={r.err_alt_m:6.2f} m | 3D={r.err_3d_m:7.2f} m")

# ---------- Summary ----------
summ = {
    "N": len(df),
    "east_mae_m":  float(np.mean(np.abs(df.err_E_m))),
    "north_mae_m": float(np.mean(np.abs(df.err_N_m))),
    "horiz_mae_m": float(df.err_horiz_m.mean()),
    "horiz_med_m": float(df.err_horiz_m.median()),
    "alt_mae_m":   float(df.err_alt_m.mean()),
    "alt_med_m":   float(df.err_alt_m.median()),
    "3d_mae_m":    float(df.err_3d_m.mean()),
    "3d_med_m":    float(df.err_3d_m.median()),
    "horiz_p90_m": float(df.err_horiz_m.quantile(0.90)),
    "3d_p90_m":    float(df.err_3d_m.quantile(0.90)),
}
print("\n=== GEO ERROR SUMMARY (meters) ===")
for k,v in summ.items():
    print(f"{k:>12}: {v:.3f}" if isinstance(v,float) else f"{k:>12}: {v}")

# ---------- Save detailed results ----------
cols = [
    "image_file",
    "pred_lat", "pred_lon", "pred_alt",
    "lat", "lon", "alt",
    "err_E_m", "err_N_m", "err_horiz_m", "err_alt_m", "err_3d_m"
]
df[cols].to_csv(OUT_ERR_CSV, index=False)
print(f"\n✅ Saved detailed per-sample errors -> {OUT_ERR_CSV}")
