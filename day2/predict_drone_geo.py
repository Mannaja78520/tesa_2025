#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, joblib
from math import radians, cos

R_E = 6378137.0
ALT_OFFSET = 0.0     # อยากชดเชยสูงกล้องเพิ่มเติมก็ใส่ที่นี่

def enu_to_llh(E,N,lat0,lon0,h):
    lat = lat0 + (N/R_E)*180/np.pi
    lon = lon0 + (E/(R_E*cos(radians(lat0))))*180/np.pi
    return lat, lon, h

def R_ypr(yaw,pitch,roll):
    y,p,r = map(radians,(yaw,pitch,roll))
    Ry=[[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]]
    Rp=[[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]]
    Rr=[[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]]
    return np.array(Ry)@np.array(Rp)@np.array(Rr)

def unit(v): 
    v=np.asarray(v,float); n=np.linalg.norm(v); 
    return v/(n if n>0 else 1.0)

def make_features(u,v,w,h,W,H,cx,cy):
    xn=(u-cx)/W; yn=(v-cy)/H; wn=w/W; hn=h/H
    an=np.sqrt(max(wn*hn,1e-12)); r=np.sqrt(xn*xn+yn*yn); ar=wn/(hn+1e-12)
    return np.array([xn,yn,wn,hn,an,r,ar], float)

m = joblib.load("drone_geo_model.pkl")
reg=m["reg"]; yaw=m["yaw"]; pitch=m["pitch"]; roll=m["roll"]
fx=m["fx"]; fy=m["fy"]; cx=m["cx"]; cy=m["cy"]
cam_lat=m["cam_lat"]; cam_lon=m["cam_lon"]; cam_alt=m["cam_alt"]
W=m["img_W"]; H=m["img_H"]

Rcam=R_ypr(yaw,pitch,roll)
Kinv=np.linalg.inv(np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float))

det = pd.read_csv("P2_DATA_TRAIN/detect_drone.csv").rename(columns={"center_x":"u","center_y":"v","width":"w","height":"h"})
rows=[]
for _,r in det.iterrows():
    u,v,w,h = float(r.u),float(r.v),float(r.w),float(r.h)
    feat = make_features(u,v,w,h,W,H,cx,cy)
    lam  = float(reg.predict([feat])[0])
    v_flipped = (H - 1) - v
    b = unit(Kinv @ np.array([u, v_flipped, 1.0]))
    rdir = unit(Rcam @ b)
    E,N,U = (lam*rdir).tolist()
    lat,lon,alt = enu_to_llh(E,N,cam_lat,cam_lon,cam_alt+U+ALT_OFFSET)
    rows.append((r.image_name,lat,lon,alt))

pd.DataFrame(rows, columns=["ImageName","Latitude","Longitude","Altitude"]).to_csv("predicted_positions.csv", index=False)
print("✅ saved predicted_positions.csv")
