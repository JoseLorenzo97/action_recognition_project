import cv2
import torch
import numpy as np
from models.cnn_lstm import CNNLSTM
from utils.preprocess import load_video_frames

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH = "samples/sample.mp4"    # tu video
OUTPUT_PATH = "output_demo.mp4"
NUM_CLASSES = 5
CLASS_NAMES = ["running", "walking", "jumping", "falling", "punching"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = CNNLSTM(num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# Si tienes pesos entrenados, descomenta esto:
# model.load_state_dict(torch.load("saved_models/cnn_lstm_best.pth"))

model.eval()

# -----------------------------
# PROCESS VIDEO
# -----------------------------
frames_tensor = load_video_frames(VIDEO_PATH)      # (T, C, H, W)
frames_tensor = frames_tensor.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(frames_tensor)
    pred_index = torch.argmax(logits, dim=1).item()
    pred_class = CLASS_NAMES[pred_index]

print(f"Predicted action: {pred_class}")

# -----------------------------
# GENERATE OUTPUT VIDEO
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(
        frame,
        f"Action: {pred_class}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    out.write(frame)

cap.release()
out.release()

print("Demo video saved as:", OUTPUT_PATH)
