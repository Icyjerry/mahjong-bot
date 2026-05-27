import os
import sys
import unittest

PROJECT_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_PARENT)

from mahjong_bot.main import PredictionStabilizer, _alt_label, _route_label, _status_text


class MainDisplayTest(unittest.TestCase):
    def test_route_label_uses_strategy_label(self):
        self.assertEqual("TENPAI-13", _route_label({
            "route_label": "TENPAI-13",
            "shanten": 0,
        }))

    def test_route_label_falls_back_to_shanten(self):
        self.assertEqual("2-shanten", _route_label({"shanten": 2}))

    def test_alt_label_includes_route_and_ukeire(self):
        self.assertEqual("[m5:TENPAI-13/39]", _alt_label({
            "code": "m5",
            "route_label": "TENPAI-13",
            "total": 39,
        }))

    def test_status_text_distinguishes_waiting_states(self):
        self.assertEqual("Waiting draw", _status_text(13))
        self.assertEqual("Recognition incomplete (12/14)", _status_text(12))
        self.assertEqual("Recognition unstable (15/14)", _status_text(15))

    def test_status_text_reports_strategy_error(self):
        self.assertEqual("Strategy error: bad hand", _status_text(14, "bad hand"))

    def test_prediction_stabilizer_waits_for_repeated_prediction(self):
        stabilizer = PredictionStabilizer(window=3, required=2)
        first = ["m1"] * 14
        second = ["m2"] * 14

        self.assertIsNone(stabilizer.update(first))
        self.assertIsNone(stabilizer.update(second))
        self.assertEqual(second, stabilizer.update(second))

    def test_prediction_stabilizer_accepts_alternating_recent_match(self):
        stabilizer = PredictionStabilizer(window=3, required=2)
        first = ["m1"] * 14
        second = ["m2"] * 14

        self.assertIsNone(stabilizer.update(first))
        self.assertIsNone(stabilizer.update(second))
        self.assertEqual(first, stabilizer.update(first))


if __name__ == "__main__":
    unittest.main()
