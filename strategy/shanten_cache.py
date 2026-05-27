"""
Shanten 计算缓存。

tuple(hand_34) → shanten，避免重复计算相同牌组。
"""

from typing import List
from mahjong.shanten import Shanten

_calc = Shanten()
_cache: dict = {}


def shanten(hand_34: List[int]) -> int:
    key = tuple(hand_34)
    if key in _cache:
        return _cache[key]
    s = _calc.calculate_shanten(list(hand_34))
    _cache[key] = s
    return s


def cache_stats() -> str:
    return f"{len(_cache)} entries"
