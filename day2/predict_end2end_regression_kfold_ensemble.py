#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, pandas as pd, joblib
from math import radians, cos

MODEL_DIR = "models_fold_e2e"
META_PKL  = os.path.join(MODEL_DIR, "meta.pkl")
DETECT_CSV = "P2_DATA_TRAIN/detect_drone.csv"   # image_file,center_x,center_y,width,height
OUT_CSV    = "predicted_positions.csv"

R_E = 6378137.0

def enu_to_llh(E, N, U, lat0, lon0, h0):
    lat = lat0 + (N / R_E) * 180.0/np.pi
    lon = lon0 + (E / (R_E * cos(radians(lat0)))) * 180.0/np.pi
    alt = h0 + U
    return float(lat), float(lon), float(alt)

def base_feats(u, v, w, h, W, H):
    xn=u/W; yn=v/H; wn=w/W; hn=h/H
    area=wn*hn
    cx = xn - 0.5; cy = yn - 0.5
    r  = np.hypot(cx, cy)
    aspect = wn/(hn+1e-12)
    ang = np.arctan2(cy, cx)
    sa, ca = np.sin(ang), np.cos(ang)
    loga = np.log(max(area, 1e-12))
    x2, y2 = xn*xn, yn*yn
    xy = xn*yn
    w2, h2 = wn*wn, hn*hn
    wh = wn*hn
    return np.array([
        xn, yn, wn, hn,
        cx, cy, r, aspect, area, loga, sa, ca,
        x2, y2, xy, w2, h2, wh
    ], dtype=float)

def make_X_row(u, v, w, h, W, H):
    return base_feats(u, v, w, h, W, H).reshape(1, -1)

meta = joblib.load(META_PKL)
W, H = meta["img_W"], meta["img_H"]
CAM_LAT, CAM_LON, CAM_ALT = meta["cam_lat"], meta["cam_lon"], meta["cam_alt"]
N_SPLITS = meta["n_splits"]
AREA_NEAR_THRESH = meta["area_near_thresh"]
scaler = meta["scaler"]

det = pd.read_csv(DETECT_CSV)

def predict_one_row(u,v,w,h):
    # TTA: jitter ±0.5 px (ปรับเป็น 1.0 ได้)
    TTA = [(0,0,0,0), (0.5,0,0,0), (-0.5,0,0,0), (0,0.5,0,0), (0,-0.5,0,0)]
    E_preds_folds=[]; N_preds_folds=[]; U_preds_folds=[]
    area = (w/W)*(h/H)
    near = (area >= AREA_NEAR_THRESH)

    for k in range(1, N_SPLITS+1):
        # เลือกโมเดลตาม near/far
        if near:
            regE = joblib.load(os.path.join(MODEL_DIR, f"f{k}_near_reg_E.pkl"))
            regN = joblib.load(os.path.join(MODEL_DIR, f"f{k}_near_reg_N.pkl"))
            regU = joblib.load(os.path.join(MODEL_DIR, f"f{k}_near_reg_U.pkl"))
        else:
            regE = joblib.load(os.path.join(MODEL_DIR, f"f{k}_far_reg_E.pkl"))
            regN = joblib.load(os.path.join(MODEL_DIR, f"f{k}_far_reg_N.pkl"))
            regU = joblib.load(os.path.join(MODEL_DIR, f"f{k}_far_reg_U.pkl"))

        Es=[]; Ns=[]; Us=[]
        for du,dv,dw,dh in TTA:
            X = make_X_row(u+du, v+dv, w+dw, h+dh, W, H)
            Xs = scaler.transform(X)
            Es.append(float(regE.predict(Xs)[0]))
            Ns.append(float(regN.predict(Xs)[0]))
            Us.append(float(regU.predict(Xs)[0]))
        E_preds_folds.append(np.mean(Es))
        N_preds_folds.append(np.mean(Ns))
        U_preds_folds.append(np.mean(Us))

    return float(np.mean(E_preds_folds)), float(np.mean(N_preds_folds)), float(np.mean(U_preds_folds))

rows=[]
for _, r in det.iterrows():
    u,v,w,h = map(float, (r.center_x, r.center_y, r.width, r.height))
    E,N,U = predict_one_row(u,v,w,h)
    lat, lon, alt = enu_to_llh(E, N, U, CAM_LAT, CAM_LON, CAM_ALT)
    rows.append((r.image_file, lat, lon, alt))

pd.DataFrame(rows, columns=["ImageName","Latitude","Longitude","Altitude"]).to_csv(OUT_CSV, index=False)
print(f"✅ saved {OUT_CSV}")
