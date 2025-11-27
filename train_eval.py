import os
import glob
import argparse
import csv
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from models.cnn_lstm import CNNLSTM


class VideoFolderDataset(Dataset):
    """
    Expects a folder structure like:
        data_root/
            running/
                vid1.mp4
                vid2.mp4
            walking/
                ...
    """

    def __init__(self, root_dir: str, classes: List[str], clip_len: int = 16, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.clip_len = clip_len
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        for idx, cls in enumerate(classes):
            class_dir = os.path.join(root_dir, cls)
            for ext in ("*.mp4", "*.avi", "*.mov"):
                self.samples.extend(
                    (path, idx) for path in glob.glob(os.path.join(class_dir, ext))
                )

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
            clip = [self.transform(img) for img in clip]

        clip = torch.stack(clip, dim=0)  # (T, C, H, W)
        return clip

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        clip = self._load_video_frames(path)
        return clip, label


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for clips, labels in loader:
        clips = clips.to(device)
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
    all_preds, all_labels = [], []

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate CNN+LSTM for action recognition.")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the root folder of the dataset (e.g. UCF101 subset).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results_metrics.csv",
        help="Path to CSV file where metrics per epoch are stored.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    classes = ["running", "walking", "jumping", "falling", "punching"]
    num_classes = len(classes)
    clip_len = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Dataset root:", args.data_root)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = VideoFolderDataset(
        root_dir=args.data_root,
        classes=classes,
        clip_len=clip_len,
        transform=transform,
    )

    train_len = int(0.7 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = CNNLSTM(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    best_state = None

    # abrir CSV y escribir cabecera
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"]
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, val_f1, cm = evaluate(
                model, val_loader, criterion, device
            )

            print(f"Epoch {epoch}/{args.epochs}")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, f1={train_f1:.4f}")
            print(f"  Val  : loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}")
            print("  Confusion matrix:")
            print(cm)

            writer.writerow(
                [epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1]
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()

    os.makedirs("saved_models", exist_ok=True)
    if best_state is not None:
        torch.save(best_state, "saved_models/cnn_lstm_best.pth")
        print(f"Best val accuracy: {best_val_acc:.4f}")
        print("Model saved to saved_models/cnn_lstm_best.pth")
    print(f"Metrics saved to {args.output_csv}")


if __name__ == "__main__":
    main()

    print("Model saved to saved_models/cnn_lstm_best.pth")


if __name__ == "__main__":
    main()
