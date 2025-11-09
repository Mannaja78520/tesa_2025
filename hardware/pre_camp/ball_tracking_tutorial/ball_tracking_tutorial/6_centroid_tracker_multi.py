import cv2
import numpy as np
import time

# === Centroid Tracker with full path tracking (unchanged) ===
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.nextObjectID = 0
        self.objects = {}         # objectID: centroid
        self.disappeared = {}     # objectID: frame count
        self.trajectories = {}    # objectID: list of centroids
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.trajectories[self.nextObjectID] = [centroid]
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.trajectories[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects, self.trajectories

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            input_centroids[i] = (x + w // 2, y + h // 2)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            objectIDs = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = input_centroids[col]
                self.disappeared[objectID] = 0
                self.trajectories[objectID].append(input_centroids[col])  # keep full path
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(D.shape[0])) - used_rows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)

            for col in set(range(D.shape[1])) - used_cols:
                self.register(input_centroids[col])

        return self.objects, self.trajectories

# ===== Load video or webcam =====
# Set to 0 for webcam, or a path like "source_dir/vdo1.mov"
video_path = 0
cap = cv2.VideoCapture(video_path)

# ===== Create window + trackbars (for ORANGE in HSV) =====
cv2.namedWindow("Adjust", cv2.WINDOW_NORMAL)

def nothing(x): pass

# HSV lower/upper trackbars
cv2.createTrackbar("H Low",  "Adjust", 5,   179, nothing)
cv2.createTrackbar("H High", "Adjust", 25,  179, nothing)
cv2.createTrackbar("S Low",  "Adjust", 120, 255, nothing)
cv2.createTrackbar("S High", "Adjust", 255, 255, nothing)
cv2.createTrackbar("V Low",  "Adjust", 120, 255, nothing)
cv2.createTrackbar("V High", "Adjust", 255, 255, nothing)

# Morphology + size
cv2.createTrackbar("Dilate",   "Adjust", 2,  20, nothing)
cv2.createTrackbar("Erode",    "Adjust", 0,  10, nothing)
cv2.createTrackbar("Min Area", "Adjust", 300, 5000, nothing)

# ===== Init tracker & FPS =====
tracker = CentroidTracker()
paused = False
frame = None
prev_time = time.time()

while cap.isOpened():
    if not paused:
        ret, img = cap.read()
        if not ret:
            print("✅ Video ended or can't read frame.")
            break
        frame = img.copy()
    else:
        if frame is None:
            continue
        img = frame.copy()

    # Resize for consistent processing & speed
    img = cv2.resize(img, (1280, 720))

    # --- Preprocess ---
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # --- Read parameters ---
    hL = cv2.getTrackbarPos("H Low",  "Adjust")
    sL = cv2.getTrackbarPos("S Low",  "Adjust")
    vL = cv2.getTrackbarPos("V Low",  "Adjust")
    hH = cv2.getTrackbarPos("H High", "Adjust")
    sH = cv2.getTrackbarPos("S High", "Adjust")
    vH = cv2.getTrackbarPos("V High", "Adjust")
    erode_val  = cv2.getTrackbarPos("Erode",  "Adjust")
    dilate_val = cv2.getTrackbarPos("Dilate", "Adjust")
    min_area   = cv2.getTrackbarPos("Min Area","Adjust")

    lower_orange = (min(hL, hH), min(sL, sH), min(vL, vH))
    upper_orange = (max(hL, hH), max(sL, sH), max(vL, vH))

    # --- HSV masking for orange ---
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # --- Morphology to clean noise ---
    kernel = np.ones((5, 5), np.uint8)
    if erode_val > 0:
        mask = cv2.erode(mask, kernel, iterations=erode_val)
    if dilate_val > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate_val)

    # --- Contour Detection (blobs for orange) ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    rects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        rects.append((x, y, w, h))

    # --- Update tracker ---
    objects, trajectories = tracker.update(rects)

    # --- Draw bounding boxes, ID, and full path ---
    for objectID, centroid in objects.items():
        # Draw full path
        path = trajectories[objectID]
        for i in range(1, len(path)):
            cv2.line(output, path[i - 1], path[i], (255, 0, 255), 2)

        # Draw rectangle and ID (match rect nearest to centroid)
        best = None
        best_dist = 1e9
        for (x, y, w, h) in rects:
            cX = x + w // 2
            cY = y + h // 2
            d = (cX - centroid[0])**2 + (cY - centroid[1])**2
            if d < best_dist:
                best_dist = d
                best = (x, y, w, h)
        if best is not None:
            x, y, w, h = best
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, f"ID {objectID}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.circle(output, tuple(centroid), 5, (0, 0, 255), -1)

    # --- FPS Display ---
    current_time = time.time()
    fps = 1.0 / max(1e-6, (current_time - prev_time))
    prev_time = current_time
    cv2.putText(output, f"FPS: {fps:.2f}", (output.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- Show Combined Output (Frame | Mask | Output) ---
    vis_frame = cv2.resize(img, (426, 240))
    vis_mask  = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (426, 240))
    vis_out   = cv2.resize(output, (426, 240))
    combined  = np.hstack((vis_frame, vis_mask, vis_out))
    cv2.imshow("Adjust", combined)

    key = cv2.waitKey(1 if not paused else 100) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACEBAR to pause
        paused = not paused

print("Finished.")
cap.release()
cv2.destroyAllWindows()
