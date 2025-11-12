#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, joblib
from math import sin, cos, radians, atan2
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.optimize import least_squares

# ====== CONFIG ======
CAM_LAT = 14.305029
CAM_LON = 101.173010
CAM_ALT = 37.2 + 1.0          # ถ้ารู้ว่าสูงกว่าจริง +1 m

IMG_W, IMG_H = 1920, 1080

# landmark → bearing ≈ 230° → yaw_ENU ≈ -140°
LANDMARK_LAT, LANDMARK_LON = 14.292944, 101.157656
FIXED_YAW_ENU = -140.03       # 0°=East, บวกทวนเข็ม

DETECT_CSV = "P2_DATA_TRAIN/detect_drone.csv"           # image_file,center_x,center_y,width,height
GT_CSV     = "P2_DATA_TRAIN/P2_DATA_TRAIN_combined.csv" # image_file,Latitude,Longitude,Altitude

# ให้จูน intrinsics ระยะที่สอง
REFINE_INTRINSICS_STAGE2 = True

# ====== CONST ======
R_E = 6378137.0

# ---------- helpers ----------
def wrap180(a): return (a + 180.0) % 360.0 - 180.0

def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon)
    return (np.degrees(atan2(x, y)) + 360) % 360  # 0°=North, CW+

def llh_to_enu(lat, lon, h, lat0, lon0, h0):
    lat0r, lon0r = map(radians, (lat0, lon0))
    dN = (lat - lat0) * np.pi/180 * R_E
    dE = (lon - lon0) * np.pi/180 * R_E * cos(lat0r)
    dU = h - h0
    return np.array([dE, dN, dU], float)

def R_ypr(yaw,pitch,roll):
    y,p,r = map(radians,(yaw,pitch,roll))
    Ry = np.array([[np.cos(y),-np.sin(y),0],
                   [np.sin(y), np.cos(y),0],
                   [0,0,1]], float)
    Rp = np.array([[ np.cos(p),0, np.sin(p)],
                   [0,1,0],
                   [-np.sin(p),0, np.cos(p)]], float)
    Rr = np.array([[1,0,0],
                   [0,np.cos(r),-np.sin(r)],
                   [0,np.sin(r), np.cos(r)]], float)
    return Ry @ Rp @ Rr

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / (n if n>0 else 1.0)

def make_features(u,v,w,h,W,H,cx,cy):
    xn=(u-cx)/W; yn=(v-cy)/H; wn=w/W; hn=h/H
    an=np.sqrt(max(wn*hn,1e-12))
    r =np.sqrt(xn*xn+yn*yn)
    ar=wn/(hn+1e-12)
    return np.array([xn,yn,wn,hn,an,r,ar], float)

def bbox_weight(w,h,W,H):
    # ให้น้ำหนักมากกับ bbox ใหญ่ (ใกล้) แต่ clamp ช่วงให้สมดุล
    an = max((w/W)*(h/H), 1e-12)
    wgt = np.sqrt(an)                # สเกลแบบ sqrt ลดความโตเกิน
    return float(min(max(wgt, 0.02), 0.20))  # 0.02..0.20

# ---------- 1) Load & merge ----------
det = pd.read_csv(DETECT_CSV).rename(columns={"center_x":"u","center_y":"v","width":"w","height":"h"})
gt  = pd.read_csv(GT_CSV).rename(columns={"Latitude":"lat","Longitude":"lon","Altitude":"alt"})
df = det.merge(gt, on="image_file", how="inner")
if df.empty: raise RuntimeError("No rows after merge. ตรวจสอบ image_file ให้ตรงกัน")

df["W"]=IMG_W; df["H"]=IMG_H; W,H=IMG_W,IMG_H
weights = np.array([bbox_weight(df.w.iat[i], df.h.iat[i], W, H) for i in range(len(df))], float)

# ---------- 2) Initial yaw (ตรวจซ้ำ) ----------
bearing = bearing_deg(CAM_LAT,CAM_LON, LANDMARK_LAT, LANDMARK_LON)
yaw_chk = wrap180(90.0 - bearing)
print(f"Landmark check → bearing≈{bearing:.2f}°, yaw_ENU≈{yaw_chk:.2f}°; using FIXED_YAW={FIXED_YAW_ENU:.2f}°")

# ---------- 3) Build 2D-3D ----------
P_enu = np.vstack([llh_to_enu(df.lat.iat[i], df.lon.iat[i], df.alt.iat[i],
                              CAM_LAT, CAM_LON, CAM_ALT) for i in range(len(df))])
uv_list = list(zip(df.u, df.v))

# ---------- 4) Stage 1: fit yaw/pitch/roll (lock intrinsics) ----------
def residuals_ypr(theta, uv, P, fx, fy, cx, cy, W, H, weights):
    yaw, pitch, roll = theta
    Kinv = np.linalg.inv(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float))
    R = R_ypr(yaw,pitch,roll)
    res=[]
    for (u,v), Pi, wi in zip(uv, P, weights):
        v_flipped = (H - 1) - v
        b = unit(Kinv @ np.array([u, v_flipped, 1.0]))
        d = unit(Pi)
        r = unit(R @ b)
        res.extend(wi*(r - d))       # weight per-sample

    # Priors: ดึง pitch → ~8°, roll → 0°
    lam_p = 30.0
    pitch_mu, pitch_sig = 8.0, 2.0
    res += [lam_p*((pitch - pitch_mu)/pitch_sig),
            lam_p*(roll/6.0)]
    return np.array(res)

# lock K เบื้องต้น
fx0=fy0=W; cx0=W/2; cy0=H/2
theta0=[FIXED_YAW_ENU, +6.0, 0.0]       # เงยนิดๆ
b_lo =[FIXED_YAW_ENU-2,  0, -8]         # pitch ไม่ก้ม
b_hi =[FIXED_YAW_ENU+2, 12,  8]         # pitch ไม่เกิน 12°

sol1 = least_squares(
    residuals_ypr, theta0,
    bounds=(b_lo,b_hi),
    args=(uv_list, P_enu, fx0, fy0, cx0, cy0, W, H, weights),
    loss="soft_l1", f_scale=1.5, max_nfev=6000
)
yaw,pitch,roll = map(float, sol1.x)
fx,fy,cx,cy = float(fx0),float(fy0),float(cx0),float(cy0)
print(f"[Stage1] yaw={yaw:.3f}°, pitch={pitch:.3f}°, roll={roll:.3f}°  |  K locked → fx={fx:.0f},fy={fy:.0f},cx={cx:.0f},cy={cy:.0f}")

# ---------- 5) Stage 2: refine intrinsics + (dyaw, dpitch, droll) ----------
def residuals_full_delta(theta, uv, P, W, H, yaw0,pitch0,roll0, weights, yaw_ref):
    # theta = [dyaw, dpitch, droll, sfx, sfy, dcx, dcy]
    dyaw, dpitch, droll, sfx, sfy, dcx, dcy = theta
    yaw = yaw0 + dyaw; pitch = pitch0 + dpitch; roll = roll0 + droll
    fx = (1.0 + sfx)*W; fy = (1.0 + sfy)*W
    cx = (W/2) + dcx;   cy = (H/2) + dcy

    Kinv = np.linalg.inv(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float))
    R = R_ypr(yaw,pitch,roll)

    res=[]
    for (u,v), Pi, wi in zip(uv, P, weights):
        v_flipped = (H - 1) - v
        b = unit(Kinv @ np.array([u, v_flipped, 1.0]))
        d = unit(Pi)
        r = unit(R @ b)
        res.extend(wi*(r - d))

    # Priors: ให้ K ไม่หลุด, pitch ~8°, roll ~0°, และ yaw ~ landmark
    lamK = 10.0
    sigma = 0.12            # focal ±12%
    tau_w = 0.03*W          # cx ±3%W
    tau_h = 0.40*H          # cy ±40%H
    prior = [lamK*(sfx/sigma), lamK*(sfy/sigma),
             lamK*(dcx/tau_w), lamK*(dcy/tau_h)]

    lam_or = 30.0
    pitch_mu, pitch_sig = 8.0, 2.0
    prior += [lam_or*((pitch - pitch_mu)/pitch_sig),
              lam_or*(roll/6.0),
              lam_or*((yaw - yaw_ref)/2.0)]
    return np.hstack([np.array(res), np.array(prior)])

if REFINE_INTRINSICS_STAGE2:
    theta2_0 = [0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0]
    b2_lo = [-1.0, -1.0, -3.0,  -0.20, -0.20,  -0.04*W, -0.40*H]
    b2_hi = [ +1.0, +1.0, +3.0,  +0.20,  +0.20,  +0.04*W,  +0.40*H]
    sol2 = least_squares(
        residuals_full_delta, theta2_0,
        bounds=(b2_lo,b2_hi),
        args=(uv_list, P_enu, W, H, yaw, pitch, roll, weights, FIXED_YAW_ENU),
        loss="soft_l1", f_scale=1.5, max_nfev=7000
    )
    dyaw,dpitch,droll,sfx,sfy,dcx,dcy = map(float, sol2.x)
    yaw += dyaw; pitch += dpitch; roll += droll
    fx = (1.0 + sfx)*W; fy = (1.0 + sfy)*W
    cx = (W/2) + dcx;   cy = (H/2) + dcy
    print(f"[Stage2] yaw={yaw:.3f}°, pitch={pitch:.3f}°, roll={roll:.3f}°  |  fx={fx:.0f},fy={fy:.0f},cx={cx:.0f},cy={cy:.0f}")

# ---------- 6) คำนวณ λ* (ระยะตามแนวรังสี) + สร้างฟีเจอร์ ----------
Rcam = R_ypr(yaw,pitch,roll)
Kinv = np.linalg.inv(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float))

lam_true=[]; X_feat=[]
for i, ((u,v), Pi) in enumerate(zip(uv_list, P_enu)):
    v_flipped = (H - 1) - v
    b = unit(Kinv @ np.array([u, v_flipped, 1.0]))
    rdir = unit(Rcam @ b)
    lam_true.append(float(np.dot(rdir, Pi)))
    X_feat.append(make_features(df.u.iat[i], df.v.iat[i], df.w.iat[i], df.h.iat[i], W, H, cx, cy))

lam_true = np.array(lam_true, float)
mask = lam_true > 0.0
X = np.vstack(X_feat)[mask]; Y = lam_true[mask]

# ---------- 7) Train regressor for λ ----------
Xtr, Xval, Ytr, Yval = train_test_split(X, Y, test_size=0.2, random_state=42)
reg = XGBRegressor(
    n_estimators=900, max_depth=5,
    subsample=0.9, colsample_bytree=0.9,
    learning_rate=0.04, reg_lambda=1.2
)
reg.fit(Xtr, Ytr)
mae = float(np.mean(np.abs(reg.predict(Xval) - Yval)))
print(f"Validation MAE of λ (m): {mae:.2f}")

# ---------- 8) Save ----------
joblib.dump(dict(
    reg=reg,
    yaw=float(yaw), pitch=float(pitch), roll=float(roll),
    fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
    cam_lat=CAM_LAT, cam_lon=CAM_LON, cam_alt=CAM_ALT,
    img_W=W, img_H=H
), "drone_geo_model.pkl")
print("✅ Saved model -> drone_geo_model.pkl")
