#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, pandas as pd, joblib
from math import radians, cos

MODEL_DIR = "models_fold"
META_PKL  = os.path.join(MODEL_DIR, "meta.pkl")
DETECT_CSV = "P2_DATA_TEST/detect_drone.csv"
OUT_CSV    = "predicted_positions.csv"

R_E = 6378137.0

def enu_to_llh(E, N, U, lat0, lon0, h0):
    lat = lat0 + (N / R_E) * 180.0/np.pi
    lon = lon0 + (E / (R_E * cos(radians(lat0)))) * 180.0/np.pi
    alt = h0 + U
    return float(lat), float(lon), float(alt)

def make_features(u, v, w, h, W, H):
    xn=u/W; yn=v/H; wn=w/W; hn=h/H
    area=wn*hn; sqrt_area=np.sqrt(max(area,1e-12))
    r=np.sqrt((xn-0.5)**2 + (yn-0.5)**2)
    aspect=wn/(hn+1e-12)
    return np.array([xn,yn,wn,hn,sqrt_area,r,aspect], dtype=float)

meta = joblib.load(META_PKL)
W, H = meta["img_W"], meta["img_H"]
CAM_LAT, CAM_LON, CAM_ALT = meta["cam_lat"], meta["cam_lon"], meta["cam_alt"]
N_SPLITS = meta["n_splits"]

det = pd.read_csv(DETECT_CSV)
rows=[]
for _, r in det.iterrows():
    u,v,w,h = map(float, (r.center_x, r.center_y, r.width, r.height))
    feat = make_features(u,v,w,h,W,H).reshape(1,-1)

    # average predictions across folds
    E_preds=[]; N_preds=[]; U_preds=[]
    for k in range(1, N_SPLITS+1):
        reg_E = joblib.load(os.path.join(MODEL_DIR, f"f{k}_reg_E.pkl"))
        reg_N = joblib.load(os.path.join(MODEL_DIR, f"f{k}_reg_N.pkl"))
        reg_U = joblib.load(os.path.join(MODEL_DIR, f"f{k}_reg_U.pkl"))
        E_preds.append(float(reg_E.predict(feat)[0]))
        N_preds.append(float(reg_N.predict(feat)[0]))
        U_preds.append(float(reg_U.predict(feat)[0]))
    E = float(np.mean(E_preds))
    N = float(np.mean(N_preds))
    U = float(np.mean(U_preds))

    lat, lon, alt = enu_to_llh(E, N, U, CAM_LAT, CAM_LON, CAM_ALT)
    rows.append((r.image_file, lat, lon, alt))

pd.DataFrame(rows, columns=["ImageName","Latitude","Longitude","Altitude"]).to_csv(OUT_CSV, index=False)
print(f"✅ saved {OUT_CSV}")
