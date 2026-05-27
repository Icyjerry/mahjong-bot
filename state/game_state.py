"""
麻将局面状态模块。

GameState: 从 CNN 识别结果恢复的完整局面
BotState:  系统自身状态（阶段、待确认动作等）
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

TILE_TO_34 = {
    "m1": 0, "m2": 1, "m3": 2, "m4": 3, "m5": 4, "m6": 5, "m7": 6, "m8": 7, "m9": 8,
    "p1": 9, "p2": 10, "p3": 11, "p4": 12, "p5": 13, "p6": 14, "p7": 15, "p8": 16, "p9": 17,
    "s1": 18, "s2": 19, "s3": 20, "s4": 21, "s5": 22, "s6": 23, "s7": 24, "s8": 25, "s9": 26,
    "east": 27, "south": 28, "west": 29, "north": 30,
    "haku": 31, "hatsu": 32, "chun": 33,
}


@dataclass
class GameState:
    """
    从 CV + CNN 恢复的牌局状态。

    hand_tiles: 手牌 tile_code 列表 (e.g. ["m1","m2","p3",...])
    hand_34:    34 维数组，每种牌的张数
    """
    hand_tiles: List[str] = field(default_factory=list)
    hand_34: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.hand_34 and self.hand_tiles:
            self.hand_34 = tiles_to_34(self.hand_tiles)

    @property
    def hand_count(self) -> int:
        return sum(self.hand_34)


@dataclass
class BotState:
    """系统自身状态。"""
    pending_discard: Optional[str] = None
    pending_action: Optional[str] = None


def tiles_to_34(tiles: List[str]) -> List[int]:
    """tile_code 列表 → 34 维计数数组。"""
    arr = [0] * 34
    for t in tiles:
        if t in TILE_TO_34:
            arr[TILE_TO_34[t]] += 1
    return arr


def tiles_from_34(arr: List[int]) -> List[str]:
    """34 维计数数组 → tile_code 列表。"""
    idx_to_tile = {v: k for k, v in TILE_TO_34.items()}
    result = []
    for i, count in enumerate(arr):
        code = idx_to_tile.get(i)
        if code:
            result.extend([code] * count)
    return result


def cnn_results_to_state(predictions: List[Optional[str]]) -> GameState:
    """CNN 预测结果 → GameState。None 跳过。"""
    tiles = [p for p in predictions if p is not None]
    return GameState(hand_tiles=tiles)
