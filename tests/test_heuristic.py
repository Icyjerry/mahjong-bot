import os
import sys
import unittest

PROJECT_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_PARENT)

from mahjong_bot.state.game_state import tiles_to_34
from mahjong_bot.strategy.heuristic import value_score


class ValueScoreTest(unittest.TestCase):
    def test_dragon_pair_is_more_valuable_than_wind_pair(self):
        dragon_pair = tiles_to_34(["haku", "haku"])
        wind_pair = tiles_to_34(["east", "east"])

        self.assertGreater(value_score(dragon_pair), value_score(wind_pair))

    def test_tanyao_shape_scores_above_terminal_heavy_shape(self):
        tanyao = tiles_to_34([
            "m2", "m3", "m4", "p3", "p4", "p5", "s4",
            "s5", "s6", "m6", "m7", "p6", "p7",
        ])
        terminals = tiles_to_34([
            "m1", "m9", "p1", "p9", "s1", "s9", "east",
            "south", "west", "north", "haku", "hatsu", "chun",
        ])

        self.assertGreater(value_score(tanyao), value_score(terminals))

    def test_flush_tendency_scores_above_mixed_suits(self):
        flushish = tiles_to_34([
            "m1", "m2", "m3", "m4", "m5", "m6", "m7",
            "m8", "m9", "east", "haku", "hatsu", "chun",
        ])
        mixed = tiles_to_34([
            "m1", "m2", "m3", "p4", "p5", "p6", "s7",
            "s8", "s9", "east", "haku", "hatsu", "chun",
        ])

        self.assertGreater(value_score(flushish), value_score(mixed))

    def test_chiitoi_tendency_scores_pair_heavy_shape(self):
        pair_heavy = tiles_to_34([
            "m2", "m2", "m5", "m5", "p3", "p3", "p7",
            "p7", "s4", "s4", "east", "haku", "chun",
        ])
        no_pairs = tiles_to_34([
            "m2", "m3", "m4", "m5", "p2", "p3", "p4",
            "p5", "s2", "s3", "s4", "s5", "east",
        ])

        self.assertGreater(value_score(pair_heavy), value_score(no_pairs))


if __name__ == "__main__":
    unittest.main()
