"""
截图模块。

优先使用 macOS 原生 screencapture 命令（无需手动授权弹窗），
pyautogui 作为跨平台回退方案。
"""

import os
import subprocess
import tempfile
import platform
from datetime import datetime
from typing import Optional

import numpy as np
import cv2

from mahjong_bot.utils.config import config, Region

_IS_MACOS = platform.system() == "Darwin"
_RETINA_SCALE = None


def _get_retina_scale() -> float:
    global _RETINA_SCALE
    if _RETINA_SCALE is not None:
        return _RETINA_SCALE
    if not _IS_MACOS:
        _RETINA_SCALE = 1.0
        return _RETINA_SCALE
    native = _capture_native()
    nh, nw = native.shape[:2]
    import mss
    with mss.mss() as sct:
        logical = sct.monitors[0]
    _RETINA_SCALE = nw / logical["width"] if logical["width"] > 0 else 2.0
    return _RETINA_SCALE


def _capture_quartz_region(region: Region) -> np.ndarray:
    """Quartz 原生内存截图（极快，14ms）。"""
    from Quartz import (
        CGWindowListCreateImage, CGRectMake,
        kCGWindowListOptionOnScreenOnly, kCGWindowImageDefault,
        CGImageGetWidth, CGImageGetHeight, CGImageGetBitsPerPixel,
        CGImageGetBytesPerRow, CGImageGetDataProvider, CGDataProviderCopyData,
    )
    scale = _get_retina_scale()
    rect = CGRectMake(
        region.x / scale,
        region.y / scale,
        region.width / scale,
        region.height / scale,
    )
    img_ref = CGWindowListCreateImage(rect, kCGWindowListOptionOnScreenOnly, 0, kCGWindowImageDefault)
    if img_ref is None:
        raise RuntimeError("Quartz capture returned None")

    w = CGImageGetWidth(img_ref)
    h = CGImageGetHeight(img_ref)
    bpp = CGImageGetBitsPerPixel(img_ref)
    bpr = CGImageGetBytesPerRow(img_ref)
    dp = CGImageGetDataProvider(img_ref)
    data = CGDataProviderCopyData(dp)
    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, bpr // max(1, bpp // 8), max(1, bpp // 8))
    arr = arr[:, :w, :]
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr


def capture_region(region: Region) -> np.ndarray:
    """截取指定区域。macOS 优先 Quartz（14ms），fallback screencapture -R。"""
    if _IS_MACOS:
        try:
            return _capture_quartz_region(region)
        except Exception:
            pass
        if _is_native_available():
            scale = _get_retina_scale()
            lx, ly = int(region.x / scale), int(region.y / scale)
            lw, lh = int(region.width / scale), int(region.height / scale)
            fd, tmp = tempfile.mkstemp(suffix=".png", prefix="mjbot_reg_")
            os.close(fd)
            try:
                cmd = ["/usr/sbin/screencapture", "-x", "-R", f"{lx},{ly},{lw},{lh}", tmp]
                subprocess.run(cmd, check=True, capture_output=True)
                img = cv2.imread(tmp)
                if img is not None:
                    return img
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    full = capture_fullscreen()
    h, w = full.shape[:2]
    x1, y1 = max(0, region.x), max(0, region.y)
    x2, y2 = min(w, region.x + region.width), min(h, region.y + region.height)
    return full[y1:y2, x1:x2]


def _capture_native() -> np.ndarray:
    """
    macOS 原生全屏截图：调用 /usr/sbin/screencapture。

    -x  静默模式（无快门声）
    -C  包含鼠标光标
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="mjbot_")
    os.close(fd)

    try:
        cmd = ["/usr/sbin/screencapture", "-x", "-C", tmp_path]
        subprocess.run(cmd, check=True, capture_output=True)
        img_bgr = cv2.imread(tmp_path)
        if img_bgr is None:
            raise RuntimeError(f"screencapture 输出文件为空: {tmp_path}")
        return img_bgr
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _capture_pyautogui() -> np.ndarray:
    """pyautogui 全屏截图回退方案。"""
    import pyautogui

    img = pyautogui.screenshot()
    img_np = np.array(img)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


def _is_native_available() -> bool:
    """检测 macOS screencapture 是否可用（是否已授权）。"""
    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="mjbot_test_")
    os.close(fd)
    try:
        result = subprocess.run(
            ["/usr/sbin/screencapture", "-x", tmp_path],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False
        img = cv2.imread(tmp_path)
        return img is not None and img.size > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _capture_mss() -> np.ndarray:
    """mss 截屏（快速，CoreGraphics 原生）。"""
    import mss
    with mss.mss() as sct:
        monitor = sct.monitors[0]  # 主屏幕
        img = np.array(sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def capture_fullscreen() -> np.ndarray:
    """截取整个屏幕，返回原生分辨率 BGR 格式。"""
    if _IS_MACOS and _is_native_available():
        return _capture_native()
    try:
        return _capture_mss()
    except Exception:
        try:
            return _capture_pyautogui()
        except Exception:
            raise


def capture_hand_region() -> np.ndarray:
    """截取手牌区域。"""
    return capture_region(config.hand_region)


def save_image(
    img: np.ndarray,
    filename: Optional[str] = None,
    subdir: str = "",
) -> str:
    """保存图像到截图目录，返回文件路径。"""
    if filename is None:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    save_dir = os.path.join(config.screenshot_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, f"{filename}.png")
    cv2.imwrite(filepath, img)
    return filepath
