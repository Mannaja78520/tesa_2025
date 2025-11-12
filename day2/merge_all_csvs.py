#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_all_csvs.py
-----------------
รวม CSV 2 แบบให้เป็นไฟล์ train_data.csv สำหรับเทรนโมเดล:
  1) detect_drone.csv  -> มี image_file, center_x, center_y, width, height (หลายภาพรวมกัน)
  2) per-image CSV     -> แต่ละภาพ 1 ไฟล์ เช่น img_0001.csv มี Latitude,Longitude,Altitude

ผลลัพธ์:
  - train_data.csv      : image_file,u,v,w,h,W,H,lat,lon,alt,gt_csv
  - merge_report.csv    : รายงาน matched/missing เพื่อตรวจความถูกต้อง

การรัน (กรณีรู้ขนาดภาพตายตัว W,H):
  python3 merge_all_csvs.py \
    --per_image_dir /path/to/per_image_csvs \
    --detect_csv /path/to/detect_drone.csv \
    --default_width 1920 --default_height 1080 \
    --output_csv train_data.csv \
    --report_csv merge_report.csv

การรัน (กรณีมีโฟลเดอร์รูป ให้สคริปต์อ่าน W,H จากไฟล์ภาพจริง):
  python3 merge_all_csvs.py \
    --per_image_dir /path/to/per_image_csvs \
    --detect_csv /path/to/detect_drone.csv \
    --images_dir /path/to/images \
    --output_csv train_data.csv \
    --report_csv merge_report.csv
"""

import os
import sys
import glob
import argparse
import warnings
from typing import Optional, Dict, Tuple

import pandas as pd

# ถ้ามี Pillow จะอ่านขนาดภาพจริงได้ (ไม่มีก็ใช้ --default_width/--default_height)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# คีย์ที่ยอมรับได้ใน per-image CSV (ยืดหยุ่นกับชื่อคอลัมน์)
LAT_KEYS = ["lat","latitude","Latitude","LAT","Latitude "]
LON_KEYS = ["lon","lng","longitude","Longitude","LON","Longitude "]
ALT_KEYS = ["alt","altitude","Altitude","amsl","ALT","AGL","alt_m","Altitude "]
IMG_KEYS = ["image_file","filename","file","image","Image","image"]

def find_first_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # รองรับ case-insensitive
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def extract_lat_lon_alt(df: pd.DataFrame):
    """คืนค่า (lat,lon,alt,image_file_in_csv_or_None) จาก per-image CSV 1 ไฟล์"""
    lat_c = find_first_col(df, LAT_KEYS)
    lon_c = find_first_col(df, LON_KEYS)
    alt_c = find_first_col(df, ALT_KEYS)
    img_c = find_first_col(df, IMG_KEYS)

    lat = df[lat_c].dropna().iloc[0] if lat_c and df[lat_c].notna().any() else None
    lon = df[lon_c].dropna().iloc[0] if lon_c and df[lon_c].notna().any() else None
    alt = df[alt_c].dropna().iloc[0] if alt_c and df[alt_c].notna().any() else None
    img = df[img_c].dropna().iloc[0] if img_c and df[img_c].notna().any() else None
    return lat, lon, alt, img

def get_image_size(images_dir: Optional[str], image_file: str, default_W: Optional[int], default_H: Optional[int]):
    if images_dir:
        if not PIL_AVAILABLE:
            raise RuntimeError("ต้องติดตั้ง pillow หรือกำหนด --default_width/--default_height แทน")
        path = os.path.join(images_dir, image_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"หาไฟล์รูปไม่เจอ: {path}")
        with Image.open(path) as im:
            W, H = im.size  # (width, height)
        return int(W), int(H)
    if default_W is not None and default_H is not None:
        return int(default_W), int(default_H)
    raise ValueError("ต้องระบุ --images_dir หรือทั้ง --default_width และ --default_height")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_image_dir", required=True, help="โฟลเดอร์ CSV รายภาพ เช่น img_0001.csv, img_0002.csv ...")
    ap.add_argument("--detect_csv", required=True, help="ไฟล์รวม detection: image_file,center_x,center_y,width,height")
    ap.add_argument("--output_csv", default="train_data.csv", help="ไฟล์ผลลัพธ์หลัก")
    ap.add_argument("--report_csv", default="merge_report.csv", help="ไฟล์รายงาน matched/missing")
    ap.add_argument("--images_dir", default=None, help="โฟลเดอร์ภาพ (ถ้าต้องการอ่าน W,H จริง)")
    ap.add_argument("--default_width", type=int, default=None, help="ใช้เมื่อไม่มี images_dir")
    ap.add_argument("--default_height", type=int, default=None, help="ใช้เมื่อไม่มี images_dir")
    # ตัวเลือกแก้ basename กรณีชื่อ detect กับ CSV ไม่ตรง pattern กัน (เช่น test_0003.jpg vs img_0003.csv)
    ap.add_argument("--detect_prefix_trim", default="", help="ตัด prefix หน้าชื่อไฟล์ detect เช่น 'test_'")
    ap.add_argument("--gt_prefix_trim", default="", help="ตัด prefix หน้าชื่อไฟล์ GT เช่น 'img_'")
    ap.add_argument("--force_ext", default="", help="บังคับนามสกุลเมื่อเทียบ basename (เช่น .jpg/.png) ไม่จำเป็นส่วนใหญ่")
    args = ap.parse_args()

    # 1) โหลด detection CSV
    det = pd.read_csv(args.detect_csv)
    required_det = ["image_file","center_x","center_y","width","height"]
    for c in required_det:
        if c not in det.columns:
            raise ValueError(f"detect_csv ขาดคอลัมน์จำเป็น: {c}")
    det = det.rename(columns={
        "center_x":"u",
        "center_y":"v",
        "width":"w",
        "height":"h"
    })

    # ฟังก์ชันช่วยทำ basename ที่จะใช้เทียบ
    def det_basename(name: str) -> str:
        b = os.path.splitext(os.path.basename(name))[0]
        if args.detect_prefix_trim and b.startswith(args.detect_prefix_trim):
            b = b[len(args.detect_prefix_trim):]
        return b

    def gt_basename_from_csvpath(csv_path: str, img_in_csv: Optional[str]) -> str:
        if img_in_csv and isinstance(img_in_csv, str) and len(img_in_csv) > 0:
            b = os.path.splitext(os.path.basename(img_in_csv))[0]
        else:
            b = os.path.splitext(os.path.basename(csv_path))[0]
        if args.gt_prefix_trim and b.startswith(args.gt_prefix_trim):
            b = b[len(args.gt_prefix_trim):]
        return b

    # 2) สร้าง GT map: basename → (lat,lon,alt,src_csv,img_in_csv)
    gt_map: Dict[str, Tuple[float,float,float,str,str]] = {}
    gt_files = sorted(glob.glob(os.path.join(args.per_image_dir, "*.csv")))
    if not gt_files:
        raise FileNotFoundError(f"ไม่พบไฟล์ .csv ใน {args.per_image_dir}")

    for path in gt_files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            warnings.warn(f"ข้าม {path}: อ่าน CSV ไม่ได้ ({e})")
            continue
        lat, lon, alt, img_in_csv = extract_lat_lon_alt(df)
        if lat is None or lon is None or alt is None:
            warnings.warn(f"ข้าม {path}: ไม่มี lat/lon/alt ครบ")
            continue
        base = gt_basename_from_csvpath(path, img_in_csv)
        gt_map[base] = (float(lat), float(lon), float(alt), os.path.basename(path), img_in_csv if isinstance(img_in_csv,str) else "")

    if not gt_map:
        raise RuntimeError("ไม่พบ GT entries ที่ใช้ได้เลย (เช็คชื่อคอลัมน์ lat/lon/alt)")

    # 3) รวม
    merged_rows = []
    report_rows = []
    det_seen = set()

    for _, r in det.iterrows():
        img = r["image_file"]
        base_det = det_basename(img)
        if args.force_ext:
            # เผื่อบางงานอยาก normalize นามสกุลก่อน (ปกติไม่ต้อง)
            img = os.path.splitext(img)[0] + args.force_ext
        det_seen.add(base_det)

        matched = base_det in gt_map
        lat = lon = alt = None
        gt_csv = ""
        gt_img = ""
        note = ""

        if matched:
            lat, lon, alt, gt_csv, gt_img = gt_map[base_det]
            try:
                W, H = get_image_size(args.images_dir, img, args.default_width, args.default_height)
            except Exception as e:
                note = f"image_size_error: {e}"
                W = H = None

            if W is not None and H is not None:
                merged_rows.append(dict(
                    image_file=img,
                    u=float(r["u"]), v=float(r["v"]),
                    w=float(r["w"]), h=float(r["h"]),
                    W=int(W), H=int(H),
                    lat=lat, lon=lon, alt=alt,
                    gt_csv=gt_csv
                ))
        else:
            note = "no_matching_gt_csv"

        report_rows.append(dict(
            image_file=img,
            basename_det=base_det,
            matched=matched,
            gt_csv=gt_csv,
            gt_image_in_csv=gt_img,
            lat=lat, lon=lon, alt=alt,
            note=note
        ))

    # GT ที่ไม่ได้ถูกใช้ (ไม่มีใน detect)
    for base, (lat,lon,alt,gt_csv,gt_img) in gt_map.items():
        if base not in det_seen:
            report_rows.append(dict(
                image_file="",
                basename_det=base,
                matched=False,
                gt_csv=gt_csv,
                gt_image_in_csv=gt_img,
                lat=lat, lon=lon, alt=alt,
                note="gt_csv_unmatched_in_detect"
            ))

    # 4) เขียนผลลัพธ์
    if merged_rows:
        out = pd.DataFrame(merged_rows, columns=["image_file","u","v","w","h","W","H","lat","lon","alt","gt_csv"])
        out.to_csv(args.output_csv, index=False, encoding="utf-8")
        print(f"✅ เขียน {len(out)} แถว → {args.output_csv}")
    else:
        print("⚠️ รวมแล้วได้ 0 แถว — กรุณาเช็คว่าชื่อไฟล์ (basename) ของ detect กับ GT ตรงกันไหม")

    rep = pd.DataFrame(report_rows, columns=["image_file","basename_det","matched","gt_csv","gt_image_in_csv","lat","lon","alt","note"])
    rep.to_csv(args.report_csv, index=False, encoding="utf-8")
    print(f"📝 รายงาน {len(rep)} แถว → {args.report_csv}")

if __name__ == "__main__":
    main()
