import os
import glob
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from models.cnn_lstm import CNNLSTM


# -----------------------------
# Dataset 
# -----------------------------
class VideoFolderDataset(Dataset):


    def __init__(self, root_dir: str, classes: List[str], clip_len: int = 16, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.clip_len = clip_len
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        for idx, cls in enumerate(classes):
            class_dir = os.path.join(root_dir, cls)
            for ext in ("*.mp4", "*.avi", "*.mov"):
                for path in glob.glob(os.path.join(class_dir, ext)):
                    self.samples.append((path, idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No videos found in {root_dir}. Check paths and class names.")

    def __len__(self):
        return len(self.samples)

    def _load_video_frames(self, path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.clip_len

       
        idxs = np.linspace(0, len(frames) - 1, self.clip_len).astype(int)
        clip = [frames[i] for i in idxs]

        if self.transform is not None:
            clip = [self.transform(img) for img in clip]  # list of (C,H,W)

        clip = torch.stack(clip, dim=0)  # (T, C, H, W)
        return clip

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        clip = self._load_video_frames(path)
        return clip, label


# -----------------------------
# Training
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for clips, labels in loader:
        clips = clips.to(device)           # (B, T, C, H, W)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clips.size(0)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        outputs = model(clips)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * clips.size(0)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_f1, cm


def main():
    # -----------------------------
    # CONFIG 
    # -----------------------------
    root_dir = "data/ucf_subset"   
    classes = ["running", "walking", "jumping", "falling", "punching"]
    num_classes = len(classes)

    batch_size = 4
    num_epochs = 5      
    lr = 1e-4
    clip_len = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------------
    # Transforms y dataset
    # -----------------------------
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = VideoFolderDataset(
        root_dir=root_dir,
        classes=classes,
        clip_len=clip_len,
        transform=transform,
    )

    # 70/30 split train/val
    train_len = int(0.7 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # -----------------------------
    # Model CNN+LSTM
    # -----------------------------
    model = CNNLSTM(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, cm = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, f1={train_f1:.4f}")
        print(f"  Val  : loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}")
        print("  Confusion matrix:")
        print(cm)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(best_state, "saved_models/cnn_lstm_best.pth")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print("Model saved to saved_models/cnn_lstm_best.pth")


if __name__ == "__main__":
    main()
