"""
Ukeire（受入）—— 高性能日麻牌效率 AI。

三重优化：
  1. Shanten Cache — 避免重复计算
  2. Effective Tiles — 只枚举可能影响手牌结构的牌
  3. Shape Heuristic — 启发式评分替代 expensive ukeire recursion
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Tuple
from mahjong_bot.state.game_state import tiles_to_34, TILE_TO_34
from mahjong_bot.strategy.shanten_cache import shanten as _cached_shanten
from mahjong_bot.strategy.heuristic import shape_score, value_score

_CODE_TO_34 = TILE_TO_34
_IDX_TO_CODE = {v: k for k, v in _CODE_TO_34.items()}
KOKUSHI_IDXS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


def validate_hand_34(hand_34: List[int], expected_count: int = None) -> None:
    if len(hand_34) != 34:
        raise ValueError(f"hand_34 must contain 34 counts, got {len(hand_34)}")
    for i, count in enumerate(hand_34):
        if not isinstance(count, int):
            raise ValueError(f"hand_34[{i}] must be int, got {type(count).__name__}")
        if count < 0 or count > 4:
            code = _IDX_TO_CODE.get(i, str(i))
            raise ValueError(f"invalid tile count for {code}: {count}")
    if expected_count is not None:
        actual = sum(hand_34)
        if actual != expected_count:
            raise ValueError(f"hand_34 must contain {expected_count} tiles, got {actual}")


def calculate_shanten(hand_34: List[int]) -> int:
    validate_hand_34(hand_34)
    return _cached_shanten(hand_34)


def chiitoi_shanten(hand_34: List[int]) -> int:
    """七对子向听数。13 张六对子一单骑为听牌 (0-shanten)。"""
    validate_hand_34(hand_34)
    pairs = sum(1 for c in hand_34 if c >= 2)
    unique = sum(1 for c in hand_34 if c > 0)
    return 6 - pairs + max(0, 7 - unique)


def kokushi_shanten(hand_34: List[int]) -> int:
    """国士无双向听数。13 种幺九字齐全且有对子时为和牌 (-1)。"""
    validate_hand_34(hand_34)
    unique = sum(1 for i in KOKUSHI_IDXS if hand_34[i] > 0)
    has_pair = any(hand_34[i] >= 2 for i in KOKUSHI_IDXS)
    return 13 - unique - (1 if has_pair else 0)


def _remaining_tiles(hand_34: List[int], idx: int) -> int:
    return max(0, 4 - hand_34[idx])


def _effective_draws(hand_13: List[int]) -> List[int]:
    """筛选可能影响手牌的有效摸牌（不枚举 34 张全量）。"""
    candidates = set()
    for suit_base in (0, 9, 18):
        for i in range(suit_base, suit_base + 9):
            if hand_13[i] > 0:
                # 自身
                candidates.add(i)
                # ±2 相邻
                loc = i - suit_base
                for d in (-2, -1, 1, 2):
                    nb = loc + d
                    if 0 <= nb < 9:
                        candidates.add(suit_base + nb)
    # 字牌
    for i in range(27, 34):
        if hand_13[i] > 0:
            candidates.add(i)
    # 确保至少回到 34 全量
    if len(candidates) < 15:
        candidates = set(range(34))
    return sorted(candidates)


def simple_ukeire(hand_13: List[int]) -> Tuple[int, List[str]]:
    """
    纯降向听受入。返回 (remaining_tile_count, [tile_codes])。

    仅枚举有效摸牌。
    """
    validate_hand_34(hand_13, expected_count=13)
    return _simple_ukeire_with_visible(hand_13, hand_13)


def _simple_ukeire_with_visible(hand_13: List[int], visible_34: List[int]) -> Tuple[int, List[str]]:
    s = _cached_shanten(hand_13)
    draws = _effective_draws(hand_13)
    count = 0
    improvements = []
    for i in draws:
        remaining = _remaining_tiles(visible_34, i)
        if remaining == 0:
            continue
        test = hand_13.copy()
        test[i] += 1
        if _cached_shanten(test) < s:
            count += remaining
            improvements.append(_IDX_TO_CODE[i])
    return count, improvements


def chiitoi_ukeire(hand_13: List[int]) -> Tuple[int, List[str]]:
    """七对子降向听受入。返回 (remaining_tile_count, [tile_codes])。"""
    validate_hand_34(hand_13, expected_count=13)
    return _chiitoi_ukeire_with_visible(hand_13, hand_13)


def _chiitoi_ukeire_with_visible(hand_13: List[int], visible_34: List[int]) -> Tuple[int, List[str]]:
    s = chiitoi_shanten(hand_13)
    count = 0
    improvements = []
    for i in range(34):
        remaining = _remaining_tiles(visible_34, i)
        if remaining == 0:
            continue
        test = hand_13.copy()
        test[i] += 1
        if chiitoi_shanten(test) < s:
            count += remaining
            improvements.append(_IDX_TO_CODE[i])
    return count, improvements


def kokushi_ukeire(hand_13: List[int]) -> Tuple[int, List[str]]:
    """国士无双降向听受入。返回 (remaining_tile_count, [tile_codes])。"""
    validate_hand_34(hand_13, expected_count=13)
    return _kokushi_ukeire_with_visible(hand_13, hand_13)


def _kokushi_ukeire_with_visible(hand_13: List[int], visible_34: List[int]) -> Tuple[int, List[str]]:
    s = kokushi_shanten(hand_13)
    count = 0
    improvements = []
    for i in KOKUSHI_IDXS:
        remaining = _remaining_tiles(visible_34, i)
        if remaining == 0:
            continue
        test = hand_13.copy()
        test[i] += 1
        if kokushi_shanten(test) < s:
            count += remaining
            improvements.append(_IDX_TO_CODE[i])
    return count, improvements


def _route_label(route: str, shanten: int) -> str:
    if route == "kokushi":
        return "TENPAI-13" if shanten == 0 else f"{shanten}-13"
    if route == "chiitoi":
        return "TENPAI-7P" if shanten == 0 else f"{shanten}-7P"
    return "TENPAI-STD" if shanten == 0 else f"{shanten}-STD"


def evaluate_all_discards(hand_34: List[int]) -> List[Dict]:
    """
    枚举弃牌，用缓存 + 有效牌 + 形状启发式评分。

    Returns:
        [{code, shanten, total, decreases, improves}, ...]
    """
    validate_hand_34(hand_34, expected_count=14)
    results = []

    for i in range(34):
        if hand_34[i] == 0:
            continue

        # 模拟弃牌
        discarded = hand_34.copy()
        discarded[i] -= 1

        s = _cached_shanten(discarded)
        u_count, u_tiles = _simple_ukeire_with_visible(discarded, hand_34)
        chiitoi_s = chiitoi_shanten(discarded)
        chiitoi_count = 0
        chiitoi_tiles = []
        if chiitoi_s == s:
            chiitoi_count, chiitoi_tiles = _chiitoi_ukeire_with_visible(discarded, hand_34)
            u_idxs = {TILE_TO_34[code] for code in u_tiles}
            for code in chiitoi_tiles:
                idx = TILE_TO_34[code]
                if idx not in u_idxs:
                    u_count += _remaining_tiles(hand_34, idx)
                    u_tiles.append(code)
                    u_idxs.add(idx)

        kokushi_s = kokushi_shanten(discarded)
        kokushi_count = 0
        kokushi_tiles = []
        if kokushi_s == s:
            kokushi_count, kokushi_tiles = _kokushi_ukeire_with_visible(discarded, hand_34)
            u_idxs = {TILE_TO_34[code] for code in u_tiles}
            for code in kokushi_tiles:
                idx = TILE_TO_34[code]
                if idx not in u_idxs:
                    u_count += _remaining_tiles(hand_34, idx)
                    u_tiles.append(code)
                    u_idxs.add(idx)

        # 形状改良：用启发式评分替代 expensive ukeire recursion
        base_shape = shape_score(discarded)
        base_value = value_score(discarded)
        count_improve = 0
        improve_tiles = []
        decrease_idxs = {TILE_TO_34[code] for code in u_tiles}

        if s > 0:
            for j in _effective_draws(discarded):
                remaining = _remaining_tiles(hand_34, j)
                if remaining == 0:
                    continue
                if j in decrease_idxs:
                    continue  # already counted as decrease
                test = discarded.copy()
                test[j] += 1
                s2 = _cached_shanten(test)
                if s2 < s:
                    continue  # already counted as decrease (shouldn't happen with effective_draws)
                if s2 == s:
                    new_shape = shape_score(test)
                    if new_shape > base_shape:
                        count_improve += remaining
                        improve_tiles.append(_IDX_TO_CODE[j])

        route = "standard"
        route_tiles = u_tiles
        if kokushi_count > 0:
            route = "kokushi"
            route_tiles = kokushi_tiles
        elif chiitoi_count > 0:
            route = "chiitoi"
            route_tiles = chiitoi_tiles
        winning_tiles = route_tiles if s == 0 else []
        advance_tiles = u_tiles
        shape_tiles = improve_tiles

        results.append({
            "code": _IDX_TO_CODE[i],
            "shanten": s,
            "ukeire_raw": u_count,
            "ukeire_improve": count_improve,
            "decreases": u_tiles,
            "improves": improve_tiles,
            "total": u_count + count_improve,
            "shape": base_shape,
            "value": base_value,
            "chiitoi_shanten": chiitoi_s,
            "chiitoi_ukeire": chiitoi_count,
            "chiitoi_tiles": chiitoi_tiles,
            "kokushi_shanten": kokushi_s,
            "kokushi_ukeire": kokushi_count,
            "kokushi_tiles": kokushi_tiles,
            "route": route,
            "route_label": _route_label(route, s),
            "route_tiles": route_tiles,
            "winning_tiles": winning_tiles,
            "advance_tiles": advance_tiles,
            "shape_tiles": shape_tiles,
        })

    results.sort(key=lambda x: (
        x["shanten"], -x["total"], -x["ukeire_raw"],
        -x["chiitoi_ukeire"], -x["kokushi_ukeire"], -x["shape"], -x["value"],
    ))
    return results


def format_results(results: List[Dict]) -> str:
    lines = []
    for r in results[:8]:
        sh = r.get("route_label") or ("TENPAI" if r["shanten"] == 0 else f"{r['shanten']}-shanten")
        dec_tiles = r.get("winning_tiles") or r.get("route_tiles") or r["decreases"]
        dec = ",".join(dec_tiles[:4])
        imp = ",".join((r.get("shape_tiles") or r["improves"])[:3])
        parts = []
        if dec: parts.append(f"↓{dec}")
        if imp: parts.append(f"→{imp}")
        detail = "  ".join(parts) if parts else "-"
        lines.append(
            f"  {r['code']:6s} → {sh:12s}  ukeire={r['total']:2d}"
            f"  [{detail}]"
        )
    return "\n".join(lines)


def main():
    from mahjong_bot.capture.screenshot import capture_hand_region
    from mahjong_bot.vision.hand_splitter import split_hand
    from mahjong_bot.vision.model import TileCNN, predict_tile
    from mahjong_bot.vision.dataset import NUM_CLASSES
    from mahjong_bot.state.game_state import cnn_results_to_state
    from mahjong_bot.strategy.shanten_cache import cache_stats
    import torch

    print("=" * 55)
    print("  Optimized Ukeire AI")
    print("=" * 55)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TileCNN(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "tile_cnn.pth"),
        map_location=device, weights_only=True))
    model.eval()

    hand = capture_hand_region()
    tiles = split_hand(hand, 14)
    predictions = [predict_tile(model, t, device)[0] for t in tiles]
    print(f"\nCNN: {predictions}")

    hand_34 = tiles_to_34([p for p in predictions if p])
    print(f"Shanten: {calculate_shanten(hand_34)}")

    import time
    t0 = time.time()
    results = evaluate_all_discards(hand_34)
    elapsed = time.time() - t0

    print(f"\n弃牌推荐 ({elapsed*1000:.0f}ms, {cache_stats()}):")
    print(format_results(results))

    if results:
        r = results[0]
        print(f"\nBest: {r['code']} (shanten={r['shanten']}, ukeire={r['total']})")


if __name__ == "__main__":
    main()
