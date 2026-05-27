# Mahjong Bot (雀魂半自动日麻Bot)

AI-powered semi-automatic Mahjong assistant with computer vision and strategy engine. Designed for **Mahjong Soul (雀魂)**, it captures the game window in real-time (3 FPS), recognizes tiles via a CNN model, and provides discard recommendations based on Shanten number calculation and heuristic evaluation.

## Features

- **Real-time screen capture**: Captures the hand region from the Mahjong Soul window
- **CNN tile recognition**: Custom TileCNN model trained on labeled tile images
- **Game state tracking**: Maintains internal game state from recognition results
- **Strategy engine**: Calculates Shanten numbers and recommends optimal discards
- **Heuristic evaluation**: Multi-factor heuristic for discard ranking (ukeire, tile efficiency, etc.)
- **Debug overlay**: Visual overlay showing recognition results and recommendations

## Project Structure

```
mahjong_bot/
├── main.py                  # Main entry point (3 FPS loop)
├── requirements.txt         # Python dependencies
├── calibrated_regions.json  # Screen region calibration
├── actions/                 # Action execution (clicking tiles)
├── capture/                 # Screen capture utilities
├── dataset/                 # Training dataset (labeled tile images)
├── models/                  # Trained model weights (place tile_cnn.pth here)
├── state/                   # Game state management
├── strategy/                # Strategy module (Shanten, Ukeire, Heuristic, Advisor)
├── tests/                   # Unit tests
├── utils/                   # Utilities (region calibrator, etc.)
├── vision/                  # Computer vision (CNN model, dataset, hand splitter)
├── templates/               # Template images
└── 牌/                      # Tile reference images
```

## Requirements

- Python 3.8+
- macOS (Retina display support)

Install dependencies:

```bash
pip install -r requirements.txt
```

Note: PyTorch is also required for the CNN model. Install separately:

```bash
pip install torch torchvision
```

## Setup

1. Place the trained model file (`tile_cnn.pth`) in the `models/` directory
2. Run the region calibrator to set up screen capture regions:
   ```bash
   python -m mahjong_bot.utils.region_calibrator
   ```
3. Launch Mahjong Soul and run:
   ```bash
   python main.py
   ```

## How It Works

The bot operates at ~3 FPS:
- Captures the hand region from the game window
- Splits the hand image into individual tile images
- Runs each tile through the CNN for classification
- Builds a game state from recognized tiles
- Calculates Shanten number (how many tiles away from Tenpai)
- Recommends the optimal discard based on heuristic evaluation
- Displays recommendations via an overlay window

---

# 雀魂半自动日麻Bot

基于计算机视觉和策略引擎的AI半自动麻将助手。专为**雀魂**设计，实时捕获游戏窗口（3 FPS），通过CNN模型识别手牌，基于向听数计算和启发式评估提供弃牌建议。

## 功能特点

- **实时屏幕捕获**：从雀魂窗口捕获手牌区域
- **CNN牌识别**：基于标注牌图像训练的自定义TileCNN模型
- **游戏状态追踪**：从识别结果维护内部游戏状态
- **策略引擎**：计算向听数并推荐最佳弃牌
- **启发式评估**：多因素启发式弃牌排序
- **调试叠加层**：显示识别结果和推荐的可视化窗口

## 环境要求

- Python 3.8+
- macOS（支持Retina显示）

## 安装

```bash
pip install -r requirements.txt
pip install torch torchvision
```

## 使用

1. 将训练好的模型文件 `tile_cnn.pth` 放入 `models/` 目录
2. 运行区域校准器设置屏幕捕获区域
3. 启动雀魂并运行 `python main.py`
