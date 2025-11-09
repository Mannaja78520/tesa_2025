import cv2, numpy as np

def nothing(x): pass

# ---- เลือก source: 0 = webcam, หรือใส่ "video.mp4" ----
source = 0
cap = cv2.VideoCapture(source)

cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuner", 1280, 720)

# สร้าง trackbars บนหน้าต่าง "Tuner"
for name, maxv in [("H Low",179),("H High",179),("S Low",255),("S High",255),("V Low",255),("V High",255)]:
    cv2.createTrackbar(name, "Tuner", 0 if "Low" in name else maxv, maxv, nothing)

# ค่าตั้งต้นสำหรับสีส้ม
cv2.setTrackbarPos("H Low","Tuner",5)
cv2.setTrackbarPos("H High","Tuner",25)
cv2.setTrackbarPos("S Low","Tuner",200)
cv2.setTrackbarPos("S High","Tuner",255)
cv2.setTrackbarPos("V Low","Tuner",120)
cv2.setTrackbarPos("V High","Tuner",255)

# --- สร้างแคนวาสครั้งแรก เพื่อให้ trackbars โผล่ทันที ---
blank = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(blank, "Tuner ready - showing combined view soon...",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
cv2.imshow("Tuner", blank)
cv2.waitKey(1)  # กระตุ้นให้ HighGUI วาดหน้าต่างพร้อม trackbars

while True:
    ok, frame = cap.read()
    if not ok: break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hL = cv2.getTrackbarPos("H Low","Tuner")
    hH = cv2.getTrackbarPos("H High","Tuner")
    sL = cv2.getTrackbarPos("S Low","Tuner")
    sH = cv2.getTrackbarPos("S High","Tuner")
    vL = cv2.getTrackbarPos("V Low","Tuner")
    vH = cv2.getTrackbarPos("V High","Tuner")

    lower = (min(hL,hH), min(sL,sH), min(vL,vH))
    upper = (max(hL,hH), max(sL,sH), max(vL,vH))

    mask = cv2.inRange(hsv, lower, upper)
    res  = cv2.bitwise_and(frame, frame, mask=mask)

    # รวมภาพ 3 ช่อง แล้วโชว์ใน "Tuner" (หน้าต่างเดียวกับ trackbars)
    f1 = cv2.resize(frame, (426, 240))
    f2 = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (426, 240))
    f3 = cv2.resize(res, (426, 240))
    combined = np.hstack([f1, f2, f3])
    cv2.imshow("Tuner", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
