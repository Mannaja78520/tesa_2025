# วาดเส้นทางการเคลื่อนที่ (trail)
import cv2, numpy as np
from collections import deque

LOWER_ORANGE = (5, 200, 120)
UPPER_ORANGE = (25, 255, 255)
kernel = np.ones((5,5), np.uint8)
HISTORY = 64
pts = deque(maxlen=HISTORY)

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break
    blurred = cv2.GaussianBlur(frame, (7,7), 0)
    hsv  = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame.copy()
    center = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        (x,y,w,h) = cv2.boundingRect(c)
        center = (x+w//2, y+h//2)
        if w*h > 200:  # ป้องกันจุดเล็ก
            cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(output, center, 5, (0,0,255), -1)

    pts.appendleft(center)
    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None: continue
        thickness = int(np.sqrt(HISTORY/float(i+1))*2)
        cv2.line(output, pts[i-1], pts[i], (0,0,255), thickness)

    cv2.imshow("Mask", mask)
    cv2.imshow("Trail", output)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
