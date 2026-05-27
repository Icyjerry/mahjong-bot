"""
麻将牌 CNN 分类模型。

一个小型卷积网络，3 层卷积 + 2 层全连接。

为什么小网络足够：
  - 34 类麻将牌图案差异清晰（数字 1-9、风牌字符）
  - 无需检测物体位置，只需分类固定输入
  - 数据量有限（每类 1 张种子图），大网络易过拟合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TileCNN(nn.Module):
    """
    输入: (B, 3, 96, 64)   96=H 牌高, 64=W 牌宽
    输出: (B, 34)          34 类 logits
    """

    def __init__(self, num_classes: int = 34, dropout: float = 0.35):
        super().__init__()

        # Block 1: 96x64 → 48x32 → 24x16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        # Block 2: 24x16 → 12x8
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        # Block 3: 12x8 → 6x4
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        # 动态计算 FC 输入维度
        self._fc_in = self._compute_fc_in()

        self.fc1 = nn.Linear(self._fc_in, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def _compute_fc_in(self) -> int:
        """前向一次空数据，得到 flatten 尺寸。"""
        with torch.no_grad():
            x = torch.zeros(1, 3, 96, 64)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def predict(model: nn.Module, img_tensor: torch.Tensor) -> tuple:
    """
    单张推理。

    Args:
        model: 训练好的 TileCNN
        img_tensor: (1, 3, H, W) 或 (3, H, W)

    Returns:
        (class_code, confidence)
    """
    from mahjong_bot.vision.dataset import ALL_CLASSES

    model.eval()
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    return ALL_CLASSES[pred.item()], conf.item()


def predict_tile(
    model: nn.Module,
    tile_img: "np.ndarray",
    device: torch.device = None,
) -> tuple:
    """输入 BGR 手牌图像，输出 (tile_code, confidence)。"""
    from mahjong_bot.vision.dataset import INPUT_H, INPUT_W
    import cv2

    if device is None:
        device = next(model.parameters()).device

    img_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    return predict(model, tensor)
