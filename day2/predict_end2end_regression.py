#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, joblib
from math import radians, cos

MODEL_PKL  = "drone_end2end_reg.pkl"
DETECT_CSV = "P2_DATA_TRAIN/detect_drone.csv"   # image_file,center_x,center_y,width,height
OUT_CSV    = "predicted_positions.csv"

R_E = 6378137.0

def enu_to_llh(E, N, h, lat0, lon0, h0):
    lat = lat0 + (N / R_E) * 180.0/np.pi
    lon = lon0 + (E / (R_E * cos(radians(lat0)))) * 180.0/np.pi
    alt = h0 + h
    return float(lat), float(lon), float(alt)

def make_features(u, v, w, h, W, H):
    xn = u / W; yn = v / H
    wn = w / W; hn = h / H
    area = wn * hn
    sqrt_area = np.sqrt(max(area, 1e-12))
    r = np.sqrt((xn - 0.5)**2 + (yn - 0.5)**2)
    aspect = wn / (hn + 1e-12)
    return np.array([xn, yn, wn, hn, sqrt_area, r, aspect], dtype=float)

m = joblib.load(MODEL_PKL)
reg_E, reg_N, reg_U = m["reg_E"], m["reg_N"], m["reg_U"]
CAM_LAT, CAM_LON, CAM_ALT = m["cam_lat"], m["cam_lon"], m["cam_alt"]
W, H = m["img_W"], m["img_H"]

det = pd.read_csv(DETECT_CSV)
rows=[]
for _, r in det.iterrows():
    u, v, w, h = float(r.center_x), float(r.center_y), float(r.width), float(r.height)
    feat = make_features(u, v, w, h, W, H).reshape(1, -1)
    E = float(reg_E.predict(feat)[0])
    N = float(reg_N.predict(feat)[0])
    U = float(reg_U.predict(feat)[0])
    lat, lon, alt = enu_to_llh(E, N, U, CAM_LAT, CAM_LON, CAM_ALT)
    rows.append((r.image_file, lat, lon, alt))

pd.DataFrame(rows, columns=["ImageName","Latitude","Longitude","Altitude"]).to_csv(OUT_CSV, index=False)
print(f"✅ saved {OUT_CSV}")
