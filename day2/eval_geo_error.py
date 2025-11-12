#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, pandas as pd
from math import radians, sin, cos, atan2, sqrt

R_E = 6378137.0
def haversine_m(lat1, lon1, lat2, lon2):
    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R_E*atan2(sqrt(a), sqrt(1-a))

PRED_CSV = "predicted_positions.csv"
GT_CSV   = "P2_DATA_TRAIN/P2_DATA_TRAIN_combined.csv"

pred = pd.read_csv(PRED_CSV)
gt   = pd.read_csv(GT_CSV).rename(columns={"Latitude":"lat","Longitude":"lon","Altitude":"alt"})
df = pred.merge(gt, on="image_file", how="inner")
if df.empty: raise RuntimeError("No overlap rows; check image_file names.")

df["err_horiz_m"] = [haversine_m(a,b,c,d) for a,b,c,d in zip(df.pred_lat, df.pred_lon, df.lat, df.lon)]
df["err_alt_m"]   = (df.pred_alt - df.alt).abs()
df["err_3d_m"]    = np.sqrt(df.err_horiz_m**2 + df.err_alt_m**2)

summ = {
    "N": len(df),
    "horiz_mae_m": float(df.err_horiz_m.mean()),
    "horiz_med_m": float(df.err_horiz_m.median()),
    "alt_mae_m":   float(df.err_alt_m.mean()),
    "alt_med_m":   float(df.err_alt_m.median()),
    "3d_mae_m":    float(df.err_3d_m.mean()),
    "3d_med_m":    float(df.err_3d_m.median()),
    "horiz_p90_m": float(df.err_horiz_m.quantile(0.90)),
    "3d_p90_m":    float(df.err_3d_m.quantile(0.90)),
}
print("=== GEO ERROR (meters) ===")
for k,v in summ.items():
    print(f"{k:>12}: {v:.3f}" if isinstance(v,float) else f"{k:>12}: {v}")
