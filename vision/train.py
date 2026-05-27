"""
麻将牌 CNN 训练脚本。

用法:
  python -m mahjong_bot.vision.train

输出:
  models/tile_cnn.pth    训练好的模型权重
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mahjong_bot.vision.dataset import create_dataloaders, NUM_CLASSES, ALL_CLASSES
from mahjong_bot.vision.model import TileCNN

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "tile_cnn.pth")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def main():
    device = get_device()
    print(f"device: {device}")
    print(f"classes: {NUM_CLASSES}")

    # ── data ────────────────────────────────────────────────────
    # repeats=24 → 34*24=816 训练样本，配合强增广
    train_dl, val_dl = create_dataloaders(batch_size=16, repeats=16)
    print(f"train samples: {len(train_dl.dataset)}")
    print(f"val samples:   {len(val_dl.dataset)}")

    # ── model ───────────────────────────────────────────────────
    model = TileCNN(num_classes=NUM_CLASSES).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs(MODEL_DIR, exist_ok=True)

    best_acc = 0.0
    epochs = 50

    print(f"\n{'Epoch':>6} {'Train Loss':>10} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>10} {'Best':>10}")
    print("-" * 58)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_dl, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

        marker = "*" if val_acc >= best_acc else ""
        print(f"{epoch:6d} {train_loss:10.4f} {train_acc:9.4f} {val_loss:10.4f} {val_acc:9.4f} {marker:>10}")

    elapsed = time.time() - t0
    print(f"\ntraining done in {elapsed:.1f}s")
    print(f"best val accuracy: {best_acc:.4f}")
    print(f"model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
