#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, pandas as pd, joblib
from math import radians, cos
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ====== CONFIG ======
CAM_LAT = 14.305029
CAM_LON = 101.173010
CAM_ALT = 37.2 + 1.0
IMG_W, IMG_H = 1920, 1080

DETECT_CSV = "P2_DATA_TRAIN/detect_drone.csv"           # image_file,center_x,center_y,width,height
GT_CSV     = "P2_DATA_TRAIN/P2_DATA_TRAIN_combined.csv" # image_file,Latitude,Longitude,Altitude

N_SPLITS = 5
MODEL_DIR = "models_fold_e2e"
OOF_CSV   = "oof_predictions_e2e.csv"
FOLD_SUMMARY_CSV = "fold_metrics_e2e.csv"

# เกณฑ์แยก Near/Far (จากพื้นที่กล่องปกติ)
AREA_NEAR_THRESH = 0.0025   # = 0.25% ของภาพ (ลองปรับ 0.002 ~ 0.004)
# =====================

R_E = 6378137.0

def llh_to_enu(lat, lon, h, lat0, lon0, h0):
    dN = (lat - lat0) * np.pi/180.0 * R_E
    dE = (lon - lon0) * np.pi/180.0 * R_E * cos(radians(lat0))
    dU = h - h0
    return np.array([dE, dN, dU], dtype=float)

def base_feats(u, v, w, h, W, H):
    xn=u/W; yn=v/H; wn=w/W; hn=h/H
    area=wn*hn
    cx = xn - 0.5; cy = yn - 0.5   # offset from center
    r  = np.hypot(cx, cy)
    aspect = wn/(hn+1e-12)
    # มุมเชิงขั้ว (หลีกเลี่ยง atan2 เพื่อลดไม่ต่อเนื่อง แทนด้วย sin/cos ของมุม)
    ang = np.arctan2(cy, cx)
    sa, ca = np.sin(ang), np.cos(ang)
    loga = np.log(max(area, 1e-12))
    # โพลินอม/interaction
    x2, y2 = xn*xn, yn*yn
    xy = xn*yn
    w2, h2 = wn*wn, hn*hn
    wh = wn*hn

    return np.array([
        xn, yn, wn, hn,
        cx, cy, r, aspect, area, loga, sa, ca,
        x2, y2, xy, w2, h2, wh
    ], dtype=float)

def make_X(df, W, H):
    X = []
    areas = []
    for _, r in df.iterrows():
        u,v,w,h = map(float, (r.center_x, r.center_y, r.width, r.height))
        feat = base_feats(u,v,w,h,W,H)
        X.append(feat)
        areas.append((w/W)*(h/H))
    return np.vstack(X), np.asarray(areas, float)

def make_reg():
    return XGBRegressor(
        n_estimators=1600,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        learning_rate=0.03,
        reg_lambda=2.5,      # แรงขึ้น
        reg_alpha=0.1,
        n_jobs=4,
        random_state=42
    )

# ---------- Load ----------
det = pd.read_csv(DETECT_CSV)
gt  = pd.read_csv(GT_CSV).rename(columns={"Latitude":"lat","Longitude":"lon","Altitude":"alt"})
df  = det.merge(gt, on="image_name", how="inner")
if df.empty:
    raise RuntimeError("No rows after merge. Check filenames.")

# ---------- Build X, Y ----------
X_all, area_all = make_X(df, IMG_W, IMG_H)

E_list=[]; N_list=[]; U_list=[]
for _, r in df.iterrows():
    E,N,U = llh_to_enu(float(r.lat), float(r.lon), float(r.alt), CAM_LAT, CAM_LON, CAM_ALT)
    E_list.append(E); N_list.append(N); U_list.append(U)
E_all = np.asarray(E_list, float)
N_all = np.asarray(N_list, float)
U_all = np.asarray(U_list, float)

os.makedirs(MODEL_DIR, exist_ok=True)

# Normalizer สำหรับฟีเจอร์ (ช่วย regularize นิดหน่อย)
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# ---------- K-Fold ----------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# เก็บ OOF สำหรับ ensemble (Near/Far blend)
oof_E = np.zeros(len(X_all)); oof_N = np.zeros(len(X_all)); oof_U = np.zeros(len(X_all))
fold_rows = []

for k, (tr_idx, val_idx) in enumerate(kf.split(X_all_scaled), 1):
    Xtr, Xval = X_all_scaled[tr_idx], X_all_scaled[val_idx]
    Etr, Eval = E_all[tr_idx], E_all[val_idx]
    Ntr, Nval = N_all[tr_idx], N_all[val_idx]
    Utr, Uval = U_all[tr_idx], U_all[val_idx]

    area_tr, area_val = area_all[tr_idx], area_all[val_idx]
    # mask near/far
    near_tr = area_tr >= AREA_NEAR_THRESH
    far_tr  = ~near_tr
    near_val= area_val >= AREA_NEAR_THRESH
    far_val = ~near_val

    # --- โมเดล Near ---
    regE_near, regN_near, regU_near = make_reg(), make_reg(), make_reg()
    if near_tr.any():
        regE_near.fit(Xtr[near_tr], Etr[near_tr])
        regN_near.fit(Xtr[near_tr], Ntr[near_tr])
        regU_near.fit(Xtr[near_tr], Utr[near_tr])
    # --- โมเดล Far ---
    regE_far, regN_far, regU_far = make_reg(), make_reg(), make_reg()
    if far_tr.any():
        regE_far.fit(Xtr[far_tr], Etr[far_tr])
        regN_far.fit(Xtr[far_tr], Ntr[far_tr])
        regU_far.fit(Xtr[far_tr], Utr[far_tr])

    # predict (เลือกโมเดลตาม near/far ของ sample)
    pred_E = np.zeros(len(val_idx)); pred_N = np.zeros(len(val_idx)); pred_U = np.zeros(len(val_idx))
    if near_val.any():
        pred_E[near_val] = regE_near.predict(Xval[near_val])
        pred_N[near_val] = regN_near.predict(Xval[near_val])
        pred_U[near_val] = regU_near.predict(Xval[near_val])
    if far_val.any():
        pred_E[far_val] = regE_far.predict(Xval[far_val])
        pred_N[far_val] = regN_far.predict(Xval[far_val])
        pred_U[far_val] = regU_far.predict(Xval[far_val])

    oof_E[val_idx] = pred_E
    oof_N[val_idx] = pred_N
    oof_U[val_idx] = pred_U

    # metrics (รวม และแยก near/far)
    def summarize(yE,yN,yU,pE,pN,pU, name):
        mae_E = mean_absolute_error(yE, pE)
        mae_N = mean_absolute_error(yN, pN)
        mae_U = mean_absolute_error(yU, pU)
        horiz = float(np.mean(np.hypot(pE - yE, pN - yN)))
        mae3d = float(np.mean(np.sqrt((pE - yE)**2 + (pN - yN)**2 + (pU - yU)**2)))
        print(f"[Fold {k}] {name:>6} | E={mae_E:.2f} N={mae_N:.2f} U={mae_U:.2f} | horiz={horiz:.2f} 3d={mae3d:.2f}")
        fold_rows.append({"fold":k,"subset":name,"E":mae_E,"N":mae_N,"U":mae_U,"horiz":horiz,"mae3d":mae3d,"count":len(yE)})

    summarize(Eval, Nval, Uval, pred_E, pred_N, pred_U, "ALL")
    if near_val.any(): summarize(Eval[near_val], Nval[near_val], Uval[near_val],
                                 pred_E[near_val], pred_N[near_val], pred_U[near_val], "NEAR")
    if far_val.any():  summarize(Eval[far_val],  Nval[far_val],  Uval[far_val],
                                 pred_E[far_val],  pred_N[far_val],  pred_U[far_val],  "FAR")

    # save fold models
    joblib.dump(regE_near, os.path.join(MODEL_DIR, f"f{k}_near_reg_E.pkl"))
    joblib.dump(regN_near, os.path.join(MODEL_DIR, f"f{k}_near_reg_N.pkl"))
    joblib.dump(regU_near, os.path.join(MODEL_DIR, f"f{k}_near_reg_U.pkl"))
    joblib.dump(regE_far,  os.path.join(MODEL_DIR, f"f{k}_far_reg_E.pkl"))
    joblib.dump(regN_far,  os.path.join(MODEL_DIR, f"f{k}_far_reg_N.pkl"))
    joblib.dump(regU_far,  os.path.join(MODEL_DIR, f"f{k}_far_reg_U.pkl"))

# ---------- OOF summary ----------
oof_mae_E = mean_absolute_error(E_all, oof_E)
oof_mae_N = mean_absolute_error(N_all, oof_N)
oof_mae_U = mean_absolute_error(U_all, oof_U)
oof_horiz = float(np.mean(np.hypot(oof_E - E_all, oof_N - N_all)))
oof_3d    = float(np.mean(np.sqrt((oof_E - E_all)**2 + (oof_N - N_all)**2 + (oof_U - U_all)**2)))

print("\n=== OOF MAE (meters) ===")
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

# save fold metrics
pd.DataFrame(fold_rows).to_csv(FOLD_SUMMARY_CSV, index=False)

# save meta for predict-time
joblib.dump(dict(
    cam_lat=CAM_LAT, cam_lon=CAM_LON, cam_alt=CAM_ALT,
    img_W=IMG_W, img_H=IMG_H,
    n_splits=N_SPLITS, model_dir=MODEL_DIR,
    area_near_thresh=AREA_NEAR_THRESH,
    scaler=scaler
), os.path.join(MODEL_DIR, "meta.pkl"))

print(f"\n✅ Saved fold models to: {MODEL_DIR}")
print(f"✅ Saved OOF to: {OOF_CSV}")
print(f"✅ Saved fold metrics to: {FOLD_SUMMARY_CSV}")
