import cv2
import numpy as np
import torch

def load_video_frames(path, target_size=(224, 224), max_frames=16):
    cap = cv2.VideoCapture(path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = frame[:, :, ::-1] / 255.0
        frames.append(frame)
        count += 1
        if count >= max_frames:
            break

    cap.release()

    frames = np.array(frames)
    frames = frames.astype(np.float32)
    frames = np.transpose(frames, (0, 3, 1, 2))
    return torch.tensor(frames)
