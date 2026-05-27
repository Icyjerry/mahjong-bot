"""
全局配置模块。

所有坐标、尺寸、阈值集中管理，便于校准和调试。

优先加载 calibrated_regions.json（由 region_calibrator.py 生成），
不存在时回退到硬编码默认值。
"""

import os
import json
from dataclasses import dataclass
from typing import Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CALIB_PATH = os.path.join(PROJECT_ROOT, "calibrated_regions.json")


def _load_calibrated() -> dict:
    """加载校准文件，不存在则返回空。"""
    if os.path.exists(_CALIB_PATH):
        with open(_CALIB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@dataclass(frozen=True)
class Region:
    """矩形区域，描述屏幕上的一组坐标。"""
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


def _get_region(data: dict, key: str, default: Region) -> Region:
    if key in data:
        d = data[key]
        return Region(
            x=int(d["x"]),
            y=int(d["y"]),
            width=int(d["width"]),
            height=int(d["height"]),
        )
    return default


@dataclass
class Config:
    """
    系统配置。

    坐标优先从 calibrated_regions.json 加载，
    文件不存在时使用内置默认值。
    """

    # ---- 路径 ----
    screenshot_dir: str = os.path.join(PROJECT_ROOT, "screenshots")
    calibrated_path: str = _CALIB_PATH

    # ---- 手牌区域 ----
    hand_region: Region = Region(x=350, y=850, width=1200, height=130)

    # ---- 弃牌区域 ----
    discard_region: Region = Region(x=200, y=400, width=800, height=400)

    # ---- 宝牌区域 ----
    dora_region: Region = Region(x=800, y=0, width=300, height=60)

    # ---- 操作按钮区域 ----
    action_region: Region = Region(x=600, y=700, width=300, height=100)

    # ---- 手牌切分基准（几何推算法） ----
    first_tile_region: Region = Region(x=350, y=850, width=120, height=200)
    tile_gap_region: Region = Region(x=1500, y=850, width=40, height=200)

    # ---- 牌尺寸（由 hand_region 自动推算） ----
    tile_width: int = 52
    tile_height: int = 78

    # ---- 调试 ----
    debug: bool = True

    def __post_init__(self):
        self._apply_calibrated()

    def _apply_calibrated(self):
        calib = _load_calibrated()
        if not calib:
            return

        obj_hand = _get_region(calib, "hand", self.hand_region)
        object.__setattr__(self, "hand_region", obj_hand)
        object.__setattr__(self, "discard_region", _get_region(calib, "discard", self.discard_region))
        object.__setattr__(self, "dora_region", _get_region(calib, "dora", self.dora_region))
        object.__setattr__(self, "action_region", _get_region(calib, "action", self.action_region))
        object.__setattr__(self, "first_tile_region", _get_region(calib, "first_tile", self.first_tile_region))
        object.__setattr__(self, "tile_gap_region", _get_region(calib, "tile_gap", self.tile_gap_region))

        # 从手牌宽度反推单牌尺寸
        tw = obj_hand.width // 14
        th = obj_hand.height
        object.__setattr__(self, "tile_width", tw)
        object.__setattr__(self, "tile_height", th)


config = Config()
