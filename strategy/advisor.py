"""
麻将策略模块 —— 基于 Improvement Ukeire 的最优弃牌 AI。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict

from mahjong_bot.state.game_state import GameState
from mahjong_bot.strategy.ukeire import (
    calculate_shanten,
    evaluate_all_discards,
    format_results as _fmt,
)


def recommend_discard(state: GameState) -> List[Dict]:
    return evaluate_all_discards(state.hand_34)


def format_advice(results: List[Dict]) -> str:
    return _fmt(results)
