"""
Full pipeline for Drone Geo-Localization from a fixed camera
Author: Phuthiphong (Mannaja)
"""

import numpy as np
import pandas as pd
from math import sin, cos, radians, atan2, sqrt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.optimize import least_squares

# ==========================================================
# --- CAMERA SETUP -----------------------------------------
# ==========================================================
cam_lat, cam_lon, cam_alt = 14.305029, 101.173010, 37.2
land_lat, land_lon = 14.292512, 101.157602   # landmark

# --- Compute yaw (bearing from camera to landmark) ---------
def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon)
    brng = atan2(x, y)
    return (np.degrees(brng) + 360) % 360

yaw_approx = bearing_deg(cam_lat, cam_lon, land_lat, land_lon)
print(f"Approx camera yaw (from North): {yaw_approx:.2f}°")

# ==========================================================
# --- GEODESY ----------------------------------------------
# ==========================================================
R_E = 6378137.0  # earth radius

def llh_to_enu(lat, lon, h, lat0, lon0, h0):
    lat0r, lon0r = map(radians, (lat0, lon0))
    dN = (lat - lat0) * np.pi/180 * R_E
    dE = (lon - lon0) * np.pi/180 * R_E * cos(lat0r)
    dU = h - h0
    return np.array([dE, dN, dU])

def enu_to_llh(E, N, lat0, lon0, h):
    lat = lat0 + (N / R_E) * 180/np.pi
    lon = lon0 + (E / (R_E * cos(radians(lat0)))) * 180/np.pi
    return lat, lon, h

# ==========================================================
# --- CAMERA MODEL -----------------------------------------
# ==========================================================
def R_ypr(yaw, pitch, roll):
    y,p,r = map(radians, (yaw,pitch,roll))
    Ry = np.array([[cos(y), -sin(y), 0],
                   [sin(y),  cos(y), 0],
                   [0, 0, 1]])
    Rp = np.array([[cos(p), 0, sin(p)],
                   [0, 1, 0],
                   [-sin(p), 0, cos(p)]])
    Rr = np.array([[1, 0, 0],
                   [0, cos(r), -sin(r)],
                   [0, sin(r),  cos(r)]])
    return Ry @ Rp @ Rr

def unit(v): return v / np.linalg.norm(v)

# ==========================================================
# --- LOAD CSV ---------------------------------------------
# ==========================================================
# Expected columns: u,v,w,h,W,H,lat,lon,alt
df = pd.read_csv("drone_dataset.csv")

# Ground truth in ENU
P_enu = np.vstack([
    llh_to_enu(df.lat[i], df.lon[i], df.alt[i],
               cam_lat, cam_lon, cam_alt)
    for i in range(len(df))
])

uv_list = list(zip(df.u, df.v))
W, H = df.W.iloc[0], df.H.iloc[0]

# ==========================================================
# --- FIT CAMERA ORIENTATION (yaw, pitch, roll) -------------
# ==========================================================
def residuals(theta, uv, P_enu, W, H):
    yaw, pitch, roll, fx, fy, cx, cy = theta
    Kinv = np.linalg.inv(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]))
    R = R_ypr(yaw, pitch, roll)
    res = []
    for (u,v), P in zip(uv, P_enu):
        b = unit(Kinv @ np.array([u,v,1.0]))
        d = unit(P)
        r = unit(R @ b)
        res.extend(r - d)
    return np.array(res)

# initial guess: yaw≈bearing, pitch≈-5°, roll≈0
theta0 = [yaw_approx, -5.0, 0.0, 0.9*W, 0.9*W, W/2, H/2]

res = least_squares(residuals, theta0, args=(uv_list, P_enu, W, H), loss='huber', f_scale=1.0)
yaw, pitch, roll, fx, fy, cx, cy = res.x
print(f"Refined angles (deg): yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")
print(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

# ==========================================================
# --- TRAIN RANGE MODEL (supervised regression) -------------
# ==========================================================
def make_features(u,v,w,h,W,H,cx,cy):
    xn=(u-cx)/W; yn=(v-cy)/H; wn=w/W; hn=h/H; an=np.sqrt(max(wn*hn,1e-12))
    r=np.sqrt(xn*xn+yn*yn)
    return np.array([xn,yn,wn,hn,an,r])

X = np.vstack([make_features(df.u[i],df.v[i],df.w[i],df.h[i],df.W[i],df.H[i],cx,cy)
               for i in range(len(df))])
Y = np.linalg.norm(P_enu, axis=1)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

reg = XGBRegressor(n_estimators=800, max_depth=4, subsample=0.9, colsample_bytree=0.9)
reg.fit(X_train, Y_train)
val_pred = reg.predict(X_val)
print("Validation MAE (range, m):", np.mean(np.abs(val_pred - Y_val)))

# ==========================================================
# --- PREDICT NEW IMAGE ------------------------------------
# ==========================================================
def predict_drone(u,v,w,h,W,H):
    Rcam = R_ypr(yaw, pitch, roll)
    feat = make_features(u,v,w,h,W,H,cx,cy)
    Rpred = reg.predict([feat])[0]
    Kinv = np.linalg.inv(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]))
    b = unit(Kinv @ np.array([u,v,1.0]))
    r = unit(Rcam @ b)
    E,N,U = (Rpred*r).tolist()
    lat, lon, alt = enu_to_llh(E,N,cam_lat,cam_lon,cam_alt+U)
    return lat, lon, alt

# Example test point:
lat_pred, lon_pred, alt_pred = predict_drone(350, 220, 48, 42, W, H)
print("Predicted lat/lon/alt:", lat_pred, lon_pred, alt_pred)

# ==========================================================
# --- SAVE MODEL --------------------------------------------
# ==========================================================
import joblib
joblib.dump(dict(
    reg=reg,
    yaw=yaw, pitch=pitch, roll=roll,
    fx=fx, fy=fy, cx=cx, cy=cy,
    cam_lat=cam_lat, cam_lon=cam_lon, cam_alt=cam_alt
), "drone_geo_model.pkl")

print("✅ Model saved to drone_geo_model.pkl")
