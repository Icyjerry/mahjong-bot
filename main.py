"""
雀魂半自动日麻Bot 主入口。

3fps 自动识别 + 推荐。14牌推荐弃牌，13牌等待摸牌。
"""

import sys, os
_PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_PROJ))

import cv2, numpy as np, torch
from collections import deque
from mahjong_bot.capture.screenshot import capture_hand_region
from mahjong_bot.vision.hand_splitter import split_hand
from mahjong_bot.vision.model import TileCNN, predict_tile
from mahjong_bot.vision.dataset import NUM_CLASSES
from mahjong_bot.state.game_state import cnn_results_to_state
from mahjong_bot.strategy.advisor import recommend_discard, calculate_shanten
from mahjong_bot.capture.screenshot import _get_retina_scale

MODEL_PATH = os.path.join(_PROJ, "models", "tile_cnn.pth")
WINDOW = "MahjongBot"


def _route_label(rec):
    return rec.get("route_label") or ("TENPAI" if rec.get("shanten") == 0 else f"{rec.get('shanten')}-shanten")


def _alt_label(rec):
    return f"[{rec['code']}:{_route_label(rec)}/{rec['total']}]"


def _status_text(tile_count, error=None):
    if error:
        return f"Strategy error: {error}"
    if tile_count == 13:
        return "Waiting draw"
    if tile_count < 13:
        return f"Recognition incomplete ({tile_count}/14)"
    if tile_count > 14:
        return f"Recognition unstable ({tile_count}/14)"
    return "Waiting..."


class PredictionStabilizer:
    def __init__(self, window=3, required=2):
        self.window = window
        self.required = required
        self.history = deque(maxlen=window)

    def update(self, preds):
        key = tuple(preds)
        self.history.append(key)
        if sum(1 for h in self.history if h == key) >= self.required:
            return list(key)
        return None


def _load_model(device):
    m = TileCNN(NUM_CLASSES).to(device)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    m.eval()
    return m


def _analyze(model, device):
    hand = capture_hand_region()
    tiles = split_hand(hand, 14)
    preds = []
    for i, t in enumerate(tiles):
        code, conf = predict_tile(model, t, device)
        # 空牌检测：位置 13 且 (haku 或 p8) 高置信 + 图很暗
        if i == 13:
            gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            if gray.mean() < 55 and gray.std() < 15:
                preds.append(None)
                continue
        # 其他位置也防误判空牌
        if code in ("haku", "p8"):
            gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            if gray.mean() < 55 and gray.std() < 15:
                preds.append(None)
                continue
        preds.append(code)
    return tiles, preds


def _advise(preds):
    state = cnn_results_to_state(preds)
    s = calculate_shanten(state.hand_34) if state.hand_count in (13, 14) else None
    recs = recommend_discard(state) if state.hand_count == 14 else []
    return s, recs


def _draw(tiles, preds, shanten, recs):
    h0 = tiles[0].shape[0]
    gap, scale = 8, 0.5
    tws = [max(1, int(t.shape[1] * scale)) for t in tiles]
    ths = max(1, int(h0 * scale))
    info_h = 70
    pw = sum(tws) + (len(tiles) + 1) * gap
    ph = ths + info_h
    panel = np.ones((ph, pw, 3), dtype=np.uint8) * 30

    disc_code = recs[0]["code"]
    try:
        di = preds.index(disc_code)
    except ValueError:
        di = -1

    x = gap
    for i, t in enumerate(tiles):
        ts = cv2.resize(t, (tws[i], ths), interpolation=cv2.INTER_AREA)
        panel[0:ths, x:x + tws[i]] = ts
        label = preds[i] or "?"
        color = (0, 0, 255) if i == di else (0, 200, 0)
        cv2.putText(panel, label, (x + tws[i] // 2 - 12, ths + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2)
        if i == di:
            cv2.rectangle(panel, (x, 0), (x + tws[i], ths), (0, 0, 255), 3)
        x += tws[i] + gap

    best = recs[0]
    sh_label = _route_label(best)
    cv2.putText(panel, f"<<< DISCARD: {disc_code} >>>   {sh_label}  ukeire={best['total']}",
                (10, ths + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    alts = " | ".join(_alt_label(r) for r in recs[1:5])
    cv2.putText(panel, "Alts: " + alts, (10, ths + 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
    return panel


def _draw_wait(tiles, preds, status="Waiting..."):
    h0 = tiles[0].shape[0]
    gap, scale = 8, 0.5
    tws = [max(1, int(t.shape[1] * scale)) for t in tiles]
    ths = max(1, int(h0 * scale))
    info_h = 60
    pw = sum(tws) + (len(tiles) + 1) * gap
    ph = ths + info_h
    panel = np.ones((ph, pw, 3), dtype=np.uint8) * 30
    x = gap
    for i, t in enumerate(tiles):
        ts = cv2.resize(t, (tws[i], ths), interpolation=cv2.INTER_AREA)
        panel[0:ths, x:x + tws[i]] = ts
        label = preds[i] or "?"
        cv2.putText(panel, label, (x + tws[i] // 2 - 12, ths + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 0), 1)
        x += tws[i] + gap
    cv2.putText(panel, status, (10, ths + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    return panel


def _draw_error(status):
    panel = np.ones((120, 520, 3), dtype=np.uint8) * 30
    cv2.putText(panel, status, (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 180, 255), 1)
    return panel


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = _load_model(device)
    scale = _get_retina_scale()
    print(f"Model loaded. Retina scale: {scale:.1f}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(WINDOW, 50, 50)

    frame = 0
    stabilizer = PredictionStabilizer()
    while True:
        t0 = __import__('time').time()
        error = None
        try:
            tiles, raw_preds = _analyze(model, device)
            stable_preds = stabilizer.update(raw_preds)
            if stable_preds is None:
                preds, s, recs = raw_preds, None, []
            else:
                preds = stable_preds
                s, recs = _advise(preds)
        except Exception as e:
            error = str(e)
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
            try:
                tiles = split_hand(capture_hand_region(), 14)
                preds, s, recs = [None] * len(tiles), None, []
            except Exception:
                tiles, preds, s, recs = [], [], None, []

        elapsed = __import__('time').time() - t0

        tile_count = len([p for p in preds if p])
        if not tiles:
            panel = _draw_error(_status_text(tile_count, error))
        elif tile_count == 14 and recs:
            panel = _draw(tiles, preds, s, recs)
        elif error is None and stable_preds is None:
            panel = _draw_wait(tiles, preds, "Stabilizing recognition")
        else:
            panel = _draw_wait(tiles, preds, _status_text(tile_count, error))

        cv2.imshow(WINDOW, panel)

        frame += 1
        if frame == 1:
            discard = recs[0]["code"] if recs else "-"
            print(f"[start] {tile_count} tiles  shanten={s}  discard={discard}  {elapsed*1000:.0f}ms")
        elif frame % 30 == 0:
            print(f"[{frame}] {elapsed*1000:.0f}ms/frame")

        key = cv2.waitKey(1) & 0xFF  # 不限帧率
        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
