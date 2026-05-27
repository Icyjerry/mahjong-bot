"""
手牌区域切分 —— 几何推算法。

校准期内只需测量两个量：
  1. first_tile  — 手牌区域内的第一张牌（确定牌宽高 + 左起点）
  2. tile_gap   — 第 13 和 14 张牌之间的间隙（确定间距）

其余所有牌的位置均由几何计算得出，无需逐张校准。
"""

import os
import sys
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mahjong_bot.utils.config import config

CROP_RATIO = 0.72


def _tile_params() -> dict:
    """
    从配置中提取手牌切分几何参数（均为手牌区域内相对坐标）。

    Returns:
        dict with keys: tile_x, tile_w, tile_h, gap_start, gap_w
    """
    hand = config.hand_region
    first = config.first_tile_region
    gap = config.tile_gap_region

    return {
        "tile_x": first.x - hand.x,
        "tile_w": first.width,
        "tile_h": first.height,
        "gap_start": gap.x - hand.x,
        "gap_w": gap.width,
    }


def _extract(hand_img: np.ndarray, center_x: float, tile_w: float, crop_ratio: float) -> np.ndarray:
    """从手牌图像中以 center_x 为中心提取 crop_ratio 宽度的单牌。"""
    h, w = hand_img.shape[:2]
    inner_w = tile_w * crop_ratio
    x1 = max(0, int(center_x - inner_w / 2))
    x2 = min(w, int(center_x + inner_w / 2))
    return hand_img[0:h, x1:x2]


def split_hand(
    hand_img: np.ndarray,
    num_tiles: int = 14,
    crop_ratio: float = CROP_RATIO,
) -> List[np.ndarray]:
    """
    几何推算切分手牌。

    前提：first_tile 和 tile_gap 已通过 calibrator 校准。

    14 牌时：
      - 前 13 张均匀分布于 [tile_x, gap_start]
      - 第 14 张位于 gap 右侧，向右对齐
    13 牌时：
      - 全宽均匀切分（使用 tile_w 和 tile_x 作为参考）
    """
    h, w = hand_img.shape[:2]
    params = _tile_params()
    tile_x = params["tile_x"]
    tile_w = params["tile_w"]
    gap_start = params["gap_start"]
    gap_w = params["gap_w"]

    if num_tiles == 14 and tile_w > 0 and gap_start > tile_x:
        # 13 tiles 均匀分布在 [tile_x, gap_start]，step = 相邻牌左边缘间距
        step = (gap_start - tile_w) / 12
        tiles = []
        for i in range(13):
            cx = tile_x + tile_w / 2 + i * step
            tiles.append(_extract(hand_img, cx, tile_w, crop_ratio))
        cx_14 = gap_start + gap_w + tile_w / 2
        tiles.append(_extract(hand_img, cx_14, tile_w, crop_ratio))
        return tiles
    else:
        # fallback：均匀切分
        tile_w_fallback = w / num_tiles
        tiles = []
        for i in range(num_tiles):
            cx = (i + 0.5) * tile_w_fallback
            tiles.append(_extract(hand_img, cx, tile_w_fallback, crop_ratio))
        return tiles


def split_and_save(
    hand_img: np.ndarray,
    num_tiles: int = 14,
    output_dir: Optional[str] = None,
) -> List[str]:
    tiles = split_hand(hand_img, num_tiles)
    if output_dir is None:
        output_dir = os.path.join(config.screenshot_dir, "tiles")
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, tile in enumerate(tiles):
        filepath = os.path.join(output_dir, f"tile_{i:02d}.png")
        cv2.imwrite(filepath, tile)
        paths.append(filepath)
    return paths


def main():
    from mahjong_bot.capture.screenshot import capture_hand_region

    print("=" * 50)
    print("  手牌切分测试（几何推算法）")
    print("=" * 50)

    params = _tile_params()
    print(f"\n参数: tile_x={params['tile_x']} tile_w={params['tile_w']}")
    print(f"      gap_start={params['gap_start']} gap_w={params['gap_w']}")

    hand = capture_hand_region()
    h, w = hand.shape[:2]
    print(f"\n手牌区域: {w}x{h}")

    if params["tile_w"] > 0 and params["gap_start"] > params["tile_x"]:
        step = (params["gap_start"] - params["tile_w"]) / 12
        print(f"间距 step={step:.1f}px  (13 tiles in {params['gap_start']-params['tile_x']}px)")

    for n in (13, 14):
        tiles = split_hand(hand, num_tiles=n)
        print(f"\n{n} 牌:")
        for i, t in enumerate(tiles):
            print(f"  [{i:2d}] {t.shape[1]:3d}x{t.shape[0]}", end="")
            if i % 7 == 6:
                print()
        print()

    print("\n完成。先校准 first_tile 和 tile_gap 后重测。")


if __name__ == "__main__":
    main()
