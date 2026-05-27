"""
牌型结构启发式评分 v2。

核心哲学：
  1. 两面搭子（ryammen）是日麻最宝贵资产 → 高奖励
  2. 孤张字牌无连接潜力 → 明确惩罚
  3. 幺九牌连接性弱 → 轻微惩罚
  4. 中张连接性强 → 轻度奖励
  5. 搭子复合结构 → 额外奖励
"""

from typing import List


def shape_score(hand_34: List[int]) -> float:
    """
    对 13 张手牌做结构评分。越高越好。

    评分体系：
      完整面子（刻子/顺子）:  +6
      两面搭子:               +5
      对子:                   +1
      坎张:                   +1
      边张:                   +0.5
      中张孤立:               -0.5
      幺九孤立:               -1.0
      字牌孤立:               -2.0
      搭子复合（34567等）:     +2
    """
    score = 0.0

    for base in (0, 9, 18):  # 万/筒/条
        nums = hand_34[base:base + 9]

        i = 0
        while i < 9:
            if nums[i] == 0:
                i += 1
                continue

            # 刻子
            if nums[i] >= 3:
                score += 6.0
                nums[i] -= 3
                continue

            # 顺子
            if i + 2 < 9 and nums[i] >= 1 and nums[i + 1] >= 1 and nums[i + 2] >= 1:
                score += 6.0
                nums[i] -= 1
                nums[i + 1] -= 1
                nums[i + 2] -= 1
                continue

            # 对子 → 先记对子，继续看能否形成搭子
            if nums[i] >= 2:
                score += 1.0
                nums[i] -= 1  # 留一张继续匹配

            # 两面搭子 (i, i+1 都存在)
            if i + 1 < 9 and nums[i] >= 1 and nums[i + 1] >= 1:
                if 1 <= i <= 6:   # 中张两面 23~78
                    score += 5.0
                    if i + 2 < 9 and nums[i + 2] >= 1:
                        score += 2.0  # 345 复合
                    if i > 0 and nums[i - 1] >= 1:
                        score += 2.0  # 456 复合
                elif i == 0:       # 12 边张
                    score += 0.5
                else:              # 89 边张
                    score += 0.5
                nums[i] -= 1
                nums[i + 1] -= 1
                continue

            # 坎张 (i, i+2 都存在)
            if i + 2 < 9 and nums[i] >= 1 and nums[i + 2] >= 1:
                score += 1.0
                nums[i] -= 1
                nums[i + 2] -= 1
                continue

            # 孤张
            if nums[i] >= 1:
                if i == 0 or i == 8:
                    score -= 1.0   # 幺九孤立
                elif 1 <= i <= 7:
                    score -= 0.5   # 中张孤立
                nums[i] -= 1
            i += 1

    # 字牌 (27-33)
    for i in range(27, 34):
        cnt = hand_34[i]
        if cnt >= 3:
            score += 6.0   # 刻子
        elif cnt >= 2:
            score += 1.0   # 对子
        elif cnt >= 1:
            score -= 2.0   # 孤张字牌惩罚

    return score


def value_score(hand_34: List[int]) -> float:
    """
    对不依赖场况的役倾向做保守评分。只用于同分排序。

    覆盖：
      役牌：三元牌对子/刻子
      染手：同一花色数量较多
      断幺：中张多且幺九字牌少
      七对：对子数量接近七对子
    """
    score = 0.0

    # 三元牌确定有役，风牌缺少场风/自风信息，只给很小权重。
    for i in range(31, 34):
        if hand_34[i] >= 3:
            score += 4.0
        elif hand_34[i] >= 2:
            score += 2.0

    for i in range(27, 31):
        if hand_34[i] >= 3:
            score += 1.5
        elif hand_34[i] >= 2:
            score += 0.5

    suit_counts = [sum(hand_34[base:base + 9]) for base in (0, 9, 18)]
    max_suit = max(suit_counts)
    honor_count = sum(hand_34[27:34])
    if max_suit >= 10:
        score += 4.0
    elif max_suit >= 8:
        score += 2.0
    if max_suit >= 8:
        score += min(honor_count, 3) * 0.3

    terminal_count = sum(hand_34[i] for i in (0, 8, 9, 17, 18, 26))
    simple_count = sum(hand_34[base + i] for base in (0, 9, 18) for i in range(1, 8))
    if terminal_count == 0 and honor_count == 0:
        score += 3.0
    elif terminal_count + honor_count <= 2 and simple_count >= 9:
        score += 1.5

    pair_count = sum(1 for c in hand_34 if c >= 2)
    if pair_count >= 6:
        score += 4.0
    elif pair_count == 5:
        score += 2.5
    elif pair_count == 4:
        score += 1.0

    return score
