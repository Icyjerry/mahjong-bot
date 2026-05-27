#!/usr/bin/env python3
"""
雀魂半自动日麻Bot - 可视化区域校准系统（Region Calibrator）

功能：
  实时截图 + Overlay 标注 → 键盘调节区域坐标 → 保存为 JSON

键位：
  方向键 / WASD    移动选中区域
  IJKL              调整 width / height
  Shift + 方向键    快速移动
  Shift + IJKL      快速缩放
  1-6               选择区域
  Space             重新截图
  Enter             保存
  R                 重置
  Q / ESC           退出

运行：
  python -m mahjong_bot.utils.region_calibrator
"""

import sys
import os
import json

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PROJ_ROOT)
sys.path.insert(0, _PARENT)

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from mahjong_bot.capture.screenshot import capture_fullscreen

# ── keyboard detection ─────────────────────────────────────────────
# macOS 上 keyboard 库 import 可能因权限弹窗卡死，所以默认不启用。
# 如果你已经给终端授予了「辅助功能」权限，可以手动设为 True。
_KB_AVAILABLE = False

# ── constants ─────────────────────────────────────────────────────
MOVE_STEP = 5
FAST_MOVE_STEP = 20
RESIZE_STEP = 2
FAST_RESIZE_STEP = 10

WINDOW_NAME = "Region Calibrator [雀魂半自动日麻Bot]"

REGION_COLORS = {
    "hand": (0, 255, 0),
    "discard": (0, 140, 255),
    "dora": (0, 255, 255),
    "action": (255, 0, 255),
    "first_tile": (255, 255, 0),   # yellow
    "tile_gap": (255, 128, 0),      # orange
}

OUTPUT_PATH = os.path.join(_PROJ_ROOT, "calibrated_regions.json")


# ── Region dataclass ──────────────────────────────────────────────
@dataclass
class Region:
    """可调节的屏幕区域。"""
    name: str
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def move(self, dx: int, dy: int):
        self.x += dx
        self.y += dy

    def resize(self, dw: int, dh: int):
        self.width = max(10, self.width + dw)
        self.height = max(10, self.height + dh)

    def reset(self):
        self.x = 0
        self.y = 0
        self.width = 200
        self.height = 100


# ── Calibrator ────────────────────────────────────────────────────
class RegionCalibrator:
    def __init__(self):
        # 尝试加载上次保存的坐标作为起始值
        defaults = {
            "hand": (350, 850, 1200, 130),
            "discard": (200, 400, 800, 400),
            "dora": (800, 0, 300, 60),
            "action": (600, 700, 300, 100),
            "first_tile": (350, 850, 120, 200),
            "tile_gap": (1500, 850, 40, 200),
        }
        if os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, "r") as f:
                saved = json.load(f)
            for name in defaults:
                if name in saved:
                    d = saved[name]
                    defaults[name] = (d["x"], d["y"], d["width"], d["height"])

        self.regions: Dict[str, Region] = {
            "hand": Region("hand", *defaults["hand"]),
            "discard": Region("discard", *defaults["discard"]),
            "dora": Region("dora", *defaults["dora"]),
            "action": Region("action", *defaults["action"]),
            "first_tile": Region("first_tile", *defaults["first_tile"]),
            "tile_gap": Region("tile_gap", *defaults["tile_gap"]),
        }
        self.region_names = list(self.regions.keys())
        self.selected_idx = 0
        self.running = True
        self._shift_held = False
        self._scale = 0.5

    @property
    def selected_name(self) -> str:
        return self.region_names[self.selected_idx]

    @property
    def selected_region(self) -> Region:
        return self.regions[self.selected_name]

    # ── overlay ────────────────────────────────────────────────────
    def _draw_overlay(self, screen: np.ndarray) -> np.ndarray:
        display = screen.copy()
        h, w = display.shape[:2]

        for i, name in enumerate(self.region_names):
            r = self.regions[name]
            is_selected = (i == self.selected_idx)
            color = REGION_COLORS.get(name, (255, 255, 255))
            thickness = 3 if is_selected else 2

            x1, y1 = r.x, r.y
            x2, y2 = r.x + r.width, r.y + r.height
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

            # label above
            label = f"[{i + 1}] {name.upper()}"
            cv2.putText(display, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # coords below
            coord = f"x={r.x} y={r.y}  w={r.width} h={r.height}"
            cv2.putText(display, coord, (x1, min(y2 + 22, h - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # top-left HUD
        mode_hint = "←↑↓→ / WASD: move   IJKL: resize   Shift: fast"
        hints = [
            f"[{self.selected_idx + 1}] {self.selected_name.upper()}  x={self.selected_region.x} y={self.selected_region.y}  w={self.selected_region.width} h={self.selected_region.height}",
            "1-6: select   Enter: save   R: reset   Space: recapture   Q: quit",
        ]
        for j, text in enumerate(hints):
            cv2.putText(display, text, (10, 25 + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(display, mode_hint, (10, 25 + len(hints) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        return display

    # ── key handling ───────────────────────────────────────────────
    def _shift_pressed(self) -> bool:
        return self._shift_held

    def _get_step(self, base: int, fast: int) -> int:
        return fast if self._shift_pressed() else base

    def handle_key(self, key: int):
        if key == -1:
            return

        r = self.selected_region
        shift = self._shift_pressed()

        # ── quit ──
        if key in (ord("q"), ord("Q"), 27):  # ESC
            self.running = False
            return

        # ── save (Enter) ──
        if key == 13:
            self._save()
            return

        # ── reset ──
        if key in (ord("r"), ord("R"), 8):  # Backspace too
            r.reset()
            print(f"[reset] {r.name} → x=0 y=0 w=200 h=100")
            return

        # ── region selection (1-6) ──
        for i in range(len(self.region_names)):
            if key == ord(str(i + 1)):
                self.selected_idx = i
                print(f"[select] {self.selected_name}")
                return

        # ── movement (arrow / WASD, always active) ──
        move_step = self._get_step(MOVE_STEP, FAST_MOVE_STEP)
        if key in (65361, ord("a"), ord("A")):
            r.move(-move_step, 0)
        elif key in (65363, ord("d"), ord("D")):
            r.move(move_step, 0)
        elif key in (65362, ord("w"), ord("W")):
            r.move(0, -move_step)
        elif key in (65364, ord("s"), ord("S")):
            r.move(0, move_step)

        # ── resize (IJKL, always active) ──
        resize_step = self._get_step(RESIZE_STEP, FAST_RESIZE_STEP)
        if key in (ord("j"), ord("J")):
            r.resize(-resize_step, 0)
        elif key in (ord("l"), ord("L")):
            r.resize(resize_step, 0)
        elif key in (ord("i"), ord("I")):
            r.resize(0, -resize_step)
        elif key in (ord("k"), ord("K")):
            r.resize(0, resize_step)

    # ── save ────────────────────────────────────────────────────────
    def _save(self):
        inv_scale = 1.0 / self._scale
        data = {}
        for name, r in self.regions.items():
            data[name] = {
                "x": int(r.x * inv_scale),
                "y": int(r.y * inv_scale),
                "width": int(r.width * inv_scale),
                "height": int(r.height * inv_scale),
            }
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved to: {OUTPUT_PATH}")
        print(json.dumps(data, indent=2, ensure_ascii=False))

    # ── main loop ──────────────────────────────────────────────────
    def run(self):
        import time

        print("=" * 55)
        print("  雀魂半自动日麻Bot - Region Calibrator")
        print("  Arrow/WASD: move  IJKL: resize  Shift: fast  Enter: save  Q: quit")
        print("=" * 55)
        print()

        # 倒计时
        countdown = 2
        for i in range(countdown, 0, -1):
            print(f"  Capturing in {i}...", flush=True)
            time.sleep(1)

        print("  Capturing now!")
        img_native = capture_fullscreen()
        nh, nw = img_native.shape[:2]

        # 精确 50% 缩放，显示清晰，1 显示像素 = 2 原生像素
        self._scale = 0.5
        dw = int(nw * self._scale)
        dh = int(nh * self._scale)
        base = cv2.resize(img_native, (dw, dh), interpolation=cv2.INTER_AREA)

        # 初始坐标也缩放到 display 坐标
        for r in self.regions.values():
            r.x = int(r.x * self._scale)
            r.y = int(r.y * self._scale)
            r.width = int(r.width * self._scale)
            r.height = int(r.height * self._scale)

        print(f"  native: {nw}x{nh}  →  display: {dw}x{dh}  (50%)  Space to re-capture")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(1512, dw), min(900, dh))
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.waitKey(1)

        while self.running:
            display = self._draw_overlay(base)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKeyEx(16)
            self.handle_key(key)

            if key == 32:
                print("Re-capturing...")
                img_native = capture_fullscreen()
                base = cv2.resize(img_native, (dw, dh), interpolation=cv2.INTER_AREA)
                print(f"  refreshed")

            if not _KB_AVAILABLE:
                if key in (65505, 65506, 0xFFE1, 0xFFE2, 225, 226):
                    self._shift_held = True
                elif key != -1:
                    self._shift_held = False

        cv2.destroyAllWindows()
        print("\nCalibrator exited.")


# ── entry ──────────────────────────────────────────────────────────
def main():
    calib = RegionCalibrator()
    calib.run()


if __name__ == "__main__":
    main()
