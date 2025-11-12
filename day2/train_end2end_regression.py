#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, joblib, os
from math import radians, cos, sin, atan2, sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ====== CONFIG ======
# จุดกล้อง (อิงจากที่คุณให้)
CAM_LAT = 14.305029
CAM_LON = 101.173010
CAM_ALT = 37.2 + 1.0     # ถ้ารู้ว่าจริงสูงกว่า ~1 m ก็ใส่ offset ได้

# ขนาดภาพที่ตัวเลข bbox อ้างอิง
IMG_W, IMG_H = 1920, 1080

# ไฟล์อินพุต
DETECT_CSV = "P2_DATA_TRAIN/detect_drone.csv"           # image_file,center_x,center_y,width,height
GT_CSV     = "P2_DATA_TRAIN/P2_DATA_TRAIN_combined.csv" # image_file,Latitude,Longitude,Altitude

# ไฟล์โมเดลที่เซฟ
MODEL_PKL  = "drone_end2end_reg.pkl"
# =====================

R_E = 6378137.0

# ---------- Geo helpers ----------
def llh_to_enu(lat, lon, h, lat0, lon0, h0):
    """แปลง Lat/Lon/Alt → ENU (เมตร) เทียบกับกล้อง (lat0,lon0,h0)"""
    dN = (lat - lat0) * np.pi/180.0 * R_E
    dE = (lon - lon0) * np.pi/180.0 * R_E * cos(radians(lat0))
    dU = h - h0
    return np.array([dE, dN, dU], dtype=float)

# ---------- Feature engineering ----------
def make_features(u, v, w, h, W, H):
    # normalize
    xn = u / W
    yn = v / H
    wn = w / W
    hn = h / H
    # simple derived
    area = wn * hn
    sqrt_area = np.sqrt(max(area, 1e-12))
    r = np.sqrt((xn - 0.5)**2 + (yn - 0.5)**2)  # ระยะจากจุดกึ่งกลางภาพ
    aspect = wn / (hn + 1e-12)
    return np.array([xn, yn, wn, hn, sqrt_area, r, aspect], dtype=float)

# ---------- Load & merge ----------
det = pd.read_csv(DETECT_CSV)
gt  = pd.read_csv(GT_CSV).rename(columns={"Latitude":"lat","Longitude":"lon","Altitude":"alt"})
df  = det.merge(gt, on="image_file", how="inner")
if df.empty:
    raise RuntimeError("No rows after merge. ตรวจสอบ image_file ให้ตรงกันระหว่าง detect_drone.csv และ GT")

# ---------- Build X, Y (targets in ENU) ----------
X_list, E_list, N_list, U_list = [], [], [], []
for _, r in df.iterrows():
    u, v, w, h = float(r.center_x), float(r.center_y), float(r.width), float(r.height)
    X_list.append(make_features(u, v, w, h, IMG_W, IMG_H))
    E, N, U = llh_to_enu(float(r.lat), float(r.lon), float(r.alt), CAM_LAT, CAM_LON, CAM_ALT)
    E_list.append(E); N_list.append(N); U_list.append(U)

X = np.vstack(X_list)
E = np.asarray(E_list, float)
N = np.asarray(N_list, float)
U = np.asarray(U_list, float)

# ---------- Train/Val split ----------
Xtr, Xval, Etr, Eval = train_test_split(X, E, test_size=0.2, random_state=42)
_,   _,   Ntr, Nval = train_test_split(X, N, test_size=0.2, random_state=42)
_,   _,   Utr, Uval = train_test_split(X, U, test_size=0.2, random_state=42)

# ---------- Regressors (สามตัวแยก) ----------
def make_reg():
    return XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        learning_rate=0.035,
        reg_lambda=1.2,
        n_jobs=4
    )

reg_E = make_reg(); reg_N = make_reg(); reg_U = make_reg()
reg_E.fit(Xtr, Etr)
reg_N.fit(Xtr, Ntr)
reg_U.fit(Xtr, Utr)

# ---------- Validation ----------
pred_E = reg_E.predict(Xval)
pred_N = reg_N.predict(Xval)
pred_U = reg_U.predict(Xval)

mae_E = mean_absolute_error(Eval, pred_E)
mae_N = mean_absolute_error(Nval, pred_N)
mae_U = mean_absolute_error(Uval, pred_U)

horiz_mae = float(np.mean(np.sqrt((pred_E - Eval)**2 + (pred_N - Nval)**2)))
mae_3d    = float(np.mean(np.sqrt((pred_E - Eval)**2 + (pred_N - Nval)**2 + (pred_U - Uval)**2)))

print("=== VAL MAE (meters) ===")
print(f" E_mae: {mae_E:.2f}")
print(f" N_mae: {mae_N:.2f}")
print(f" U_mae: {mae_U:.2f}")
print(f"horiz: {horiz_mae:.2f}")
print(f"  3d : {mae_3d:.2f}")

# ---------- Save model ----------
joblib.dump(dict(
    reg_E=reg_E, reg_N=reg_N, reg_U=reg_U,
    cam_lat=CAM_LAT, cam_lon=CAM_LON, cam_alt=CAM_ALT,
    img_W=IMG_W, img_H=IMG_H
), MODEL_PKL)

print(f"✅ Saved model -> {MODEL_PKL}")
