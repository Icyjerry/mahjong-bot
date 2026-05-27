"""
模板匹配模块。

将手牌切分出的单牌与 templates/ 下的模板比对，识别牌面。

模板存放规则：
  牌/万/1-9.png  牌/筒/1-9.png  牌/条/1-9.png  牌/其他/东南西北白发中.png
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "牌")

# 目录名 -> (suit prefix, index shift)
SUIT_MAP: Dict[str, Tuple[str, int]] = {
    "万": ("m", 0),
    "筒": ("p", 0),
    "条": ("s", 0),
}

# 风牌箭牌文件名 -> tile code
HONOR_MAP = {
    "东": "east", "南": "south", "西": "west", "北": "north",
    "白": "haku", "发": "hatsu", "中": "chun",
}


def load_templates() -> Dict[str, np.ndarray]:
    """
    加载所有模板，返回 {tile_code: grayscale_image}。

    tile_code 格式: m1-m9, p1-p9, s1-s9, east/south/west/north/haku/hatsu/chun
    """
    templates: Dict[str, np.ndarray] = {}

    for suit_dir, (prefix, _) in SUIT_MAP.items():
        dir_path = os.path.join(TEMPLATE_DIR, suit_dir)
        if not os.path.isdir(dir_path):
            continue
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".png"):
                continue
            num = fname.replace(".png", "")
            if num.isdigit() and 1 <= int(num) <= 9:
                code = f"{prefix}{num}"
                img = cv2.imread(os.path.join(dir_path, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[code] = img

    honor_dir = os.path.join(TEMPLATE_DIR, "其他")
    if os.path.isdir(honor_dir):
        for fname in os.listdir(honor_dir):
            if not fname.endswith(".png"):
                continue
            name = fname.replace(".png", "")
            if name in HONOR_MAP:
                code = HONOR_MAP[name]
                img = cv2.imread(os.path.join(honor_dir, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[code] = img

    return templates


def match_tile(
    tile_img: np.ndarray,
    templates: Dict[str, np.ndarray],
    threshold: float = 0.35,
) -> Optional[str]:
    """
    模板匹配。双方均中心裁剪 72% 宽度，聚焦牌面图案区。
    """
    tile_gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
    th, tw = tile_gray.shape

    best_code = None
    best_score = -1.0

    for code, tpl in templates.items():
        if code == "haku":
            score = 1.0 if (np.mean(tile_gray) > 200 and np.std(tile_gray) < 20) else 0.0
        else:
            # 模板中心裁剪 72%（与手牌裁剪比例一致）
            th_t, tw_t = tpl.shape
            from mahjong_bot.vision.hand_splitter import CROP_RATIO
            inner = int(tw_t * CROP_RATIO)
            x0 = max(0, (tw_t - inner) // 2)
            tpl_cropped = tpl[:, x0:x0 + inner]

            tpl_rs = cv2.resize(tpl_cropped, (tw, th), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(tile_gray, tpl_rs, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]

        if score > best_score:
            best_score = score
            best_code = code

    if best_code == "haku":
        return best_code if best_score > 0.5 else None
    if best_score >= threshold:
        return best_code
    return None


def match_hand(
    hand_img: np.ndarray,
    templates: Optional[Dict[str, np.ndarray]] = None,
) -> List[Optional[str]]:
    """
    识别整副手牌。

    匹配用满宽提取（crop_ratio=1.0），获得更多牌面信息。
    """
    from mahjong_bot.vision.hand_splitter import split_hand

    if templates is None:
        templates = load_templates()

    tiles = split_hand(hand_img)
    return [match_tile(t, templates) for t in tiles]


def main():
    """独立测试。"""
    from mahjong_bot.capture.screenshot import capture_hand_region

    print("=" * 50)
    print("  模板匹配测试")
    print("=" * 50)

    templates = load_templates()
    print(f"\n已加载 {len(templates)} 张模板:")
    for code in sorted(templates.keys()):
        h, w = templates[code].shape
        print(f"  {code:6s} {w}x{h}")

    print("\n截取手牌...")
    hand = capture_hand_region()
    results = match_hand(hand, templates)

    print("\n识别结果:")
    for i, r in enumerate(results):
        status = r if r else "?"
        print(f"  [{i:2d}] {status}")

    ratio = sum(1 for r in results if r is not None) / max(len(results), 1)
    print(f"\n识别率: {ratio:.0%}")


if __name__ == "__main__":
    main()
