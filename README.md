# 麻将机器人 Mahjong Bot 🀄

日式麻将 AI 机器人，通过屏幕识别 + 策略引擎自动打牌。

A Japanese mahjong AI bot that plays automatically via screen recognition and strategy engine.

## 功能概述 / Overview

- **屏幕识别** — 使用 OpenCV 识别牌面、牌河、舍牌
- **策略引擎** — 基于牌效、防守、役种判断的出牌决策
- **模板匹配** — 校准区域识别，适配多种麻将客户端
- **状态管理** — 跟踪手牌、副露、牌河、点数

## 项目结构 / Structure

```
mahjong_bot/
├── main.py              # 主入口
├── actions/             # 操作模块
├── capture/             # 屏幕截图
├── vision/              # 视觉识别
├── strategy/            # 策略引擎
├── state/               # 状态管理
├── utils/               # 工具函数
├── tests/               # 测试
├── dataset/             # 训练数据
├── templates/           # 模板图片
├── 牌/                  # 牌图片资源
├── models/              # 模型文件（不含 .pth）
├── calibrated_regions.json  # 屏幕区域校准
└── requirements.txt     # Python 依赖
```

## 依赖 / Requirements

```
pip install -r requirements.txt
```

## 许可证 / License

MIT
