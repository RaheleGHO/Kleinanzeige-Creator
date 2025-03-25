import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=60):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    extracted_frames = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frame_filename = f"{output_dir}/frame_{frame_count}_at_{int(timestamp)}s.jpg"
            cv2.imwrite(frame_filename, frame)
            extracted_frames.append((frame_filename, timestamp))

        frame_count += 1

    cap.release()
    return extracted_frames
