# import library
import os
import cv2

# list files
folder = "VIDEOS"
files = os.listdir(folder)
for f in files:
    full_path = os.path.join(folder, f)
    print(full_path)
    
def video_to_img(video_path, interval=2, output_dir="frames_output"):
    """
    Extract frames from a video file at a fixed interval (in seconds).

    Args:
        video_path (str): Path to the input video file.
        interval (float): Interval in seconds between extracted frames.
        output_dir (str): Directory to save the extracted frames.

    Returns:
        list: List of saved frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}, Save every {frame_interval} frames")

    count = 0
    saved = 0
    saved_files = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_files.append(filename)
            print(f"Saved: {filename}")
            saved += 1

        count += 1

    cap.release()
    print(f"Done. Extracted {saved} frames to '{output_dir}'")
    return saved_files

# extract video to image
video_to_img("VIDEOS/P1_VIDEO_4.mp4", interval=10, output_dir="frames_output")