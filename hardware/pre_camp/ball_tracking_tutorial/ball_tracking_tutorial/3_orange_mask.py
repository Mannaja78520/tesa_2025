import cv2, numpy as np

# ใส่ค่าจากไฟล์ที่ 2 (ตัวอย่างด้านล่าง) 
LOWER_ORANGE = (10, 174, 120) # ค่า H,S,V ต่ำสุด
UPPER_ORANGE = (28, 255, 255) # ค่า H,S,V สูงสุด

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    res  = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", res)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
