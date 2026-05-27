"""
麻将牌数据集模块。

从 templates/ 目录加载 34 类牌图片，提供训练/验证数据加载。

数据增强模拟真实截图中的偏移、缩放、光照变化，
让 CNN 对粗切手牌具备容错能力。
"""

import os
import sys
import random
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F_tv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "牌")
HAND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "hand")

# 目录名到 tile_code 的映射
SUIT_MAP = {"万": "m", "筒": "p", "条": "s"}
HONOR_MAP = {
    "东": "east", "南": "south", "西": "west", "北": "north",
    "白": "haku", "发": "hatsu", "中": "chun",
}

# 34 种牌的统一分类索引
ALL_CLASSES = (
    [f"m{i}" for i in range(1, 10)]
    + [f"p{i}" for i in range(1, 10)]
    + [f"s{i}" for i in range(1, 10)]
    + ["east", "south", "west", "north", "haku", "hatsu", "chun"]
)
CLASS_TO_IDX = {code: i for i, code in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)

INPUT_H, INPUT_W = 96, 64  # 模型输入尺寸 (H, W)


def _collect_images() -> list:
    """收集所有模板 + 手牌图片，返回 [(filepath, tile_code), ...]"""
    pairs = []

    # 1. 清洁模板
    for suit_dir, prefix in SUIT_MAP.items():
        dir_path = os.path.join(TEMPLATE_DIR, suit_dir)
        if not os.path.isdir(dir_path):
            continue
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".png"):
                continue
            num = fname.replace(".png", "")
            if num.isdigit() and 1 <= int(num) <= 9:
                pairs.append((os.path.join(dir_path, fname), f"{prefix}{num}"))

    honor_dir = os.path.join(TEMPLATE_DIR, "其他")
    if os.path.isdir(honor_dir):
        for fname in sorted(os.listdir(honor_dir)):
            if not fname.endswith(".png"):
                continue
            name = fname.replace(".png", "")
            if name in HONOR_MAP:
                pairs.append((os.path.join(honor_dir, fname), HONOR_MAP[name]))

    # 2. 真实手牌（dataset/hand/ 下按类别分目录）
    if os.path.isdir(HAND_DIR):
        for label_dir in sorted(os.listdir(HAND_DIR)):
            label_path = os.path.join(HAND_DIR, label_dir)
            if not os.path.isdir(label_path):
                continue
            code = _label_to_code(label_dir)
            if code is None:
                continue
            for fname in sorted(os.listdir(label_path)):
                if fname.endswith(".png"):
                    pairs.append((os.path.join(label_path, fname), code))

    return pairs


def _label_to_code(name: str) -> Optional[str]:
    """目录名 → tile_code。m1→m1, west→west, 东→east"""
    # 已是英文 tile code
    if name in CLASS_TO_IDX:
        return name
    # 中文名
    if name in HONOR_MAP:
        return HONOR_MAP[name]
    return None


def get_augmentations(train: bool = True) -> T.Compose:
    """数据增强 pipeline。训练时加随机扰动，验证时仅 resize + normalize。"""
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((INPUT_H, INPUT_W)),
            T.RandomAffine(degrees=5, translate=(0.08, 0.08), scale=(0.85, 1.15)),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((INPUT_H, INPUT_W)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class TileDataset(Dataset):
    """34 类麻将牌数据集。"""

    def __init__(self, pairs: list = None, train: bool = True, repeats: int = 16):
        if pairs is None:
            pairs = _collect_images()

        self.images: list = []
        self.train = train
        self.img_size = (INPUT_W, INPUT_H)

        for path, code in pairs:
            img = cv2.imread(path)
            if img is None:
                continue
            label = CLASS_TO_IDX.get(code, -1)
            if label < 0:
                continue
            if train:
                for _ in range(repeats):
                    self.images.append((img.copy(), label))
            else:
                self.images.append((img.copy(), label))

    def _apply_clahe(self, img_bgr):
        return img_bgr

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.images[idx]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.train:
            tensor = self._train_transform(img_rgb)
        else:
            img_rgb = cv2.resize(img_rgb, self.img_size, interpolation=cv2.INTER_AREA)
            tensor = T.ToTensor()(img_rgb)
        return tensor, label

    @staticmethod
    def _train_transform(img_rgb):
        return T.Compose([
            T.ToPILImage(),
            T.RandomAffine(degrees=2, translate=(0.03, 0.03), scale=(0.94, 1.06)),
            T.Resize((INPUT_H, INPUT_W)),
            T.ToTensor(),
        ])(img_rgb)


def create_dataloaders(
    batch_size: int = 16,
    repeats: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """全量训练 + 验证（同源，用于监控收敛）。真正测试看新鲜手牌。"""
    hand_pairs = [(p, c) for p, c in _collect_images() if "/hand/" in p]

    train_ds = TileDataset(pairs=hand_pairs, train=True, repeats=repeats)
    val_ds = TileDataset(pairs=hand_pairs, train=False, repeats=1)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl


def main():
    print(f"类别数: {NUM_CLASSES}")
    pairs = _collect_images()
    templates = [(p,c) for p,c in pairs if "/牌/" in p]
    hand = [(p,c) for p,c in pairs if "/hand/" in p]
    print(f"模板: {len(templates)}  手牌: {len(hand)}  合计: {len(pairs)}")
    train_dl, val_dl = create_dataloaders(batch_size=4, repeats=4)
    print(f"train: {len(train_dl.dataset)} samples")
    print(f"val:   {len(val_dl.dataset)} samples")
    if len(val_dl.dataset) > 0:
        x, y = next(iter(train_dl))
        print(f"batch: {x.shape}, labels: {y}")


if __name__ == "__main__":
    main()
