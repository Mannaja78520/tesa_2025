#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, pandas as pd, joblib
from math import radians, cos
from sklearn.model_selection import KFold  # เปลี่ยนเป็น GroupKFold ได้
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ====== CONFIG ======
CAM_LAT = 14.305029
CAM_LON = 101.173010
CAM_ALT = 37.2 + 1.0
IMG_W, IMG_H = 1920, 1080

DETECT_CSV = "P2_DATA_TRAIN/detect_drone.csv"           # image_file,center_x,center_y,width,height
GT_CSV     = "P2_DATA_TRAIN/P2_DATA_TRAIN_combined.csv" # image_file,Latitude,Longitude,Altitude

N_SPLITS = 5
MODEL_DIR = "models_fold"
OOF_CSV   = "oof_predictions.csv"
FOLD_INDEX_CSV = "fold_indices.csv"
# =====================

R_E = 6378137.0

def llh_to_enu(lat, lon, h, lat0, lon0, h0):
    dN = (lat - lat0) * np.pi/180.0 * R_E
    dE = (lon - lon0) * np.pi/180.0 * R_E * cos(radians(lat0))
    dU = h - h0
    return np.array([dE, dN, dU], dtype=float)

def make_features(u, v, w, h, W, H):
    xn=u/W; yn=v/H; wn=w/W; hn=h/H
    area=wn*hn; sqrt_area=np.sqrt(max(area,1e-12))
    r=np.sqrt((xn-0.5)**2+(yn-0.5)**2)
    aspect=wn/(hn+1e-12)
    return np.array([xn,yn,wn,hn,sqrt_area,r,aspect], dtype=float)

def make_reg():
    return XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        learning_rate=0.035,
        reg_lambda=1.2,
        n_jobs=4,
        random_state=42
    )

# ---------- Load ----------
det = pd.read_csv(DETECT_CSV)
gt  = pd.read_csv(GT_CSV).rename(columns={"Latitude":"lat","Longitude":"lon","Altitude":"alt"})
df  = det.merge(gt, on="image_file", how="inner")
if df.empty:
    raise RuntimeError("No rows after merge. Check filenames.")

# ---------- Build X, Y ----------
X_list=[]; E_list=[]; N_list=[]; U_list=[]
for _, r in df.iterrows():
    u,v,w,h = map(float, (r.center_x, r.center_y, r.width, r.height))
    X_list.append(make_features(u,v,w,h,IMG_W,IMG_H))
    E,N,U = llh_to_enu(float(r.lat), float(r.lon), float(r.alt), CAM_LAT, CAM_LON, CAM_ALT)
    E_list.append(E); N_list.append(N); U_list.append(U)

X = np.vstack(X_list)
E = np.asarray(E_list, float)
N = np.asarray(N_list, float)
U = np.asarray(U_list, float)
n = len(X)

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- K-Fold ----------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
# ถ้ามีคอลัมน์ group เช่น df['sequence'] ให้ใช้ GroupKFold แล้วเปลี่ยนเป็น:
# from sklearn.model_selection import GroupKFold
# kf = GroupKFold(n_splits=N_SPLITS); groups = df['sequence'].values

oof_E = np.zeros(n); oof_N = np.zeros(n); oof_U = np.zeros(n)
fold_records = []

for k, (tr_idx, val_idx) in enumerate(kf.split(X), 1):  # หรือ .split(X, groups=groups)
    Xtr, Xval = X[tr_idx], X[val_idx]
    Etr, Eval = E[tr_idx], E[val_idx]
    Ntr, Nval = N[tr_idx], N[val_idx]
    Utr, Uval = U[tr_idx], U[val_idx]

    reg_E = make_reg(); reg_N = make_reg(); reg_U = make_reg()
    reg_E.fit(Xtr, Etr)
    reg_N.fit(Xtr, Ntr)
    reg_U.fit(Xtr, Utr)

    pred_E = reg_E.predict(Xval)
    pred_N = reg_N.predict(Xval)
    pred_U = reg_U.predict(Xval)

    oof_E[val_idx] = pred_E
    oof_N[val_idx] = pred_N
    oof_U[val_idx] = pred_U

    # per-fold metrics
    mae_E = mean_absolute_error(Eval, pred_E)
    mae_N = mean_absolute_error(Nval, pred_N)
    mae_U = mean_absolute_error(Uval, pred_U)
    horiz = float(np.mean(np.sqrt((pred_E - Eval)**2 + (pred_N - Nval)**2)))
    mae3d = float(np.mean(np.sqrt((pred_E - Eval)**2 + (pred_N - Nval)**2 + (pred_U - Uval)**2)))

    print(f"[Fold {k}/{N_SPLITS}] E={mae_E:.2f}  N={mae_N:.2f}  U={mae_U:.2f} | horiz={horiz:.2f}  3d={mae3d:.2f}")

    # save fold models
    joblib.dump(reg_E, os.path.join(MODEL_DIR, f"f{k}_reg_E.pkl"))
    joblib.dump(reg_N, os.path.join(MODEL_DIR, f"f{k}_reg_N.pkl"))
    joblib.dump(reg_U, os.path.join(MODEL_DIR, f"f{k}_reg_U.pkl"))

    fold_records.append({"fold": k, "mae_E": mae_E, "mae_N": mae_N, "mae_U": mae_U,
                         "horiz": horiz, "mae_3d": mae3d,
                         "val_count": len(val_idx)})

# ---------- OOF summary ----------
oof_mae_E = mean_absolute_error(E, oof_E)
oof_mae_N = mean_absolute_error(N, oof_N)
oof_mae_U = mean_absolute_error(U, oof_U)
oof_horiz = float(np.mean(np.sqrt((oof_E - E)**2 + (oof_N - N)**2)))
oof_3d    = float(np.mean(np.sqrt((oof_E - E)**2 + (oof_N - N)**2 + (oof_U - U)**2)))

print("\n=== OOF MAE (meters, unbiased) ===")
print(f" E_mae: {oof_mae_E:.2f}")
print(f" N_mae: {oof_mae_N:.2f}")
print(f" U_mae: {oof_mae_U:.2f}")
print(f"horiz : {oof_horiz:.2f}")
print(f"  3d  : {oof_3d:.2f}")

# save OOF predictions
oof_df = df[["image_file"]].copy()
oof_df["E_pred_oof"] = oof_E
oof_df["N_pred_oof"] = oof_N
oof_df["U_pred_oof"] = oof_U
oof_df.to_csv(OOF_CSV, index=False)

pd.DataFrame(fold_records).to_csv(FOLD_INDEX_CSV, index=False)

# meta info for predict-time
joblib.dump(dict(
    cam_lat=CAM_LAT, cam_lon=CAM_LON, cam_alt=CAM_ALT,
    img_W=IMG_W, img_H=IMG_H,
    n_splits=N_SPLITS, model_dir=MODEL_DIR
), os.path.join(MODEL_DIR, "meta.pkl"))

print(f"\n✅ Saved fold models to: {MODEL_DIR}/f*_reg_[E|N|U].pkl")
print(f"✅ Saved OOF to: {OOF_CSV}")
print(f"✅ Saved fold metrics to: {FOLD_INDEX_CSV}")
