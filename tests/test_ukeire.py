import os
import sys
import unittest

PROJECT_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_PARENT)

from mahjong_bot.state.game_state import TILE_TO_34, tiles_to_34
from mahjong_bot.strategy.ukeire import (
    chiitoi_shanten,
    chiitoi_ukeire,
    evaluate_all_discards,
    kokushi_shanten,
    kokushi_ukeire,
    simple_ukeire,
    validate_hand_34,
)


class UkeireTest(unittest.TestCase):
    def test_validate_hand_rejects_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "34 counts"):
            validate_hand_34([0] * 33)

    def test_validate_hand_rejects_impossible_tile_count(self):
        hand_34 = [0] * 34
        hand_34[TILE_TO_34["m1"]] = 5

        with self.assertRaisesRegex(ValueError, "m1"):
            validate_hand_34(hand_34)

    def test_validate_hand_rejects_wrong_total(self):
        with self.assertRaisesRegex(ValueError, "13 tiles"):
            validate_hand_34(tiles_to_34(["m1", "m2"]), expected_count=13)

    def test_evaluate_all_discards_requires_14_tiles(self):
        hand = [
            "m1", "m2", "m3", "m4", "m5", "m6", "m7",
            "m8", "m9", "p5", "p5", "s6", "s7",
        ]

        with self.assertRaisesRegex(ValueError, "14 tiles"):
            evaluate_all_discards(tiles_to_34(hand))

    def test_simple_ukeire_requires_13_tiles(self):
        hand = [
            "m1", "m2", "m3", "m4", "m5", "m6", "m7",
            "m8", "m9", "p5", "p5", "s6", "s7", "east",
        ]

        with self.assertRaisesRegex(ValueError, "13 tiles"):
            simple_ukeire(tiles_to_34(hand))

    def test_improvement_draws_are_recomputed_per_discard(self):
        hand = [
            "m2", "m9", "p6", "p7", "p9", "p9", "s1",
            "s1", "s3", "s7", "s9", "east", "west", "haku",
        ]

        results = {r["code"]: r for r in evaluate_all_discards(tiles_to_34(hand))}

        self.assertEqual(74, results["p6"]["total"])
        self.assertTrue({"m1", "m3", "m4"}.issubset(results["p6"]["improves"]))

    def test_ukeire_counts_remaining_tiles_not_tile_kinds(self):
        hand = [
            "m6", "p2", "p4", "s3", "s3", "s4", "s5",
            "s5", "s5", "s5", "s9", "north", "chun",
        ]

        hand_34 = tiles_to_34(hand)
        count, tiles = simple_ukeire(hand_34)

        self.assertNotIn("s5", tiles)
        self.assertEqual(0, 4 - hand_34[TILE_TO_34["s5"]])
        self.assertEqual(57, count)

    def test_discarded_tile_is_still_visible_for_remaining_count(self):
        hand = [
            "m1", "m6", "p2", "p4", "s3", "s3", "s4",
            "s5", "s5", "s5", "s5", "s9", "north", "chun",
        ]

        results = evaluate_all_discards(tiles_to_34(hand))

        for result in results:
            self.assertNotIn("s5", result["decreases"])
            self.assertNotIn("s5", result["improves"])

    def test_tenpai_total_counts_winning_tiles_only(self):
        hand = [
            "m1", "m2", "m3", "m4", "m5", "m6", "m7",
            "m8", "m9", "p5", "p5", "s6", "s7", "east",
        ]

        results = {r["code"]: r for r in evaluate_all_discards(tiles_to_34(hand))}
        discard_east = results["east"]

        self.assertEqual(0, discard_east["shanten"])
        self.assertEqual(["s5", "s8"], discard_east["decreases"])
        self.assertEqual(8, discard_east["ukeire_raw"])
        self.assertEqual(0, discard_east["ukeire_improve"])
        self.assertEqual(8, discard_east["total"])
        self.assertEqual(["s5", "s8"], discard_east["winning_tiles"])
        self.assertEqual(["s5", "s8"], discard_east["advance_tiles"])
        self.assertEqual([], discard_east["shape_tiles"])

    def test_shape_breaks_otherwise_equal_ties(self):
        hand = [
            "m1", "m2", "m2", "m9", "p1", "p1", "p7",
            "p8", "s4", "s4", "s7", "east", "haku", "hatsu",
        ]

        order = [r["code"] for r in evaluate_all_discards(tiles_to_34(hand))]

        self.assertLess(order.index("east"), order.index("m1"))

    def test_value_breaks_otherwise_equal_ties(self):
        hand = [
            "m2", "m4", "m7", "m7", "p1", "p2", "p5",
            "p5", "p8", "p9", "s4", "s7", "s7", "s9",
        ]

        results = evaluate_all_discards(tiles_to_34(hand))
        order = [r["code"] for r in results]

        self.assertLess(order.index("s9"), order.index("p2"))

    def test_chiitoi_shanten_counts_pairs_and_unique_tiles(self):
        tenpai = tiles_to_34([
            "m1", "m1", "m2", "m2", "m5", "m5", "p3",
            "p3", "p7", "p7", "s4", "s4", "east",
        ])
        two_shanten = tiles_to_34([
            "m1", "m1", "m2", "m2", "m5", "m5", "p3",
            "p3", "p7", "s4", "east", "haku", "chun",
        ])

        self.assertEqual(0, chiitoi_shanten(tenpai))
        self.assertEqual(2, chiitoi_shanten(two_shanten))

    def test_chiitoi_ukeire_counts_singletons(self):
        hand = [
            "m1", "m1", "m2", "m2", "m5", "m5", "p3",
            "p3", "p7", "s4", "east", "haku", "chun",
        ]

        count, tiles = chiitoi_ukeire(tiles_to_34(hand))

        self.assertEqual(["p7", "s4", "east", "haku", "chun"], tiles)
        self.assertEqual(15, count)

    def test_chiitoi_route_is_reported_for_pair_heavy_discards(self):
        hand = [
            "m1", "m1", "m2", "m2", "m5", "m5", "p3",
            "p3", "p7", "p7", "s4", "s4", "east", "haku",
        ]

        results = {r["code"]: r for r in evaluate_all_discards(tiles_to_34(hand))}
        discard_east = results["east"]

        self.assertEqual(0, discard_east["chiitoi_shanten"])
        self.assertEqual(["haku"], discard_east["chiitoi_tiles"])
        self.assertEqual(3, discard_east["chiitoi_ukeire"])
        self.assertEqual("chiitoi", discard_east["route"])
        self.assertEqual("TENPAI-7P", discard_east["route_label"])
        self.assertEqual(["haku"], discard_east["route_tiles"])
        self.assertEqual(["haku"], discard_east["winning_tiles"])

    def test_kokushi_shanten_handles_thirteen_wait_and_pair_wait(self):
        thirteen_wait = tiles_to_34([
            "m1", "m9", "p1", "p9", "s1", "s9", "east",
            "south", "west", "north", "haku", "hatsu", "chun",
        ])
        pair_wait = tiles_to_34([
            "m1", "m9", "p1", "p9", "s1", "s9", "east",
            "south", "west", "north", "haku", "hatsu", "hatsu",
        ])

        self.assertEqual(0, kokushi_shanten(thirteen_wait))
        self.assertEqual(0, kokushi_shanten(pair_wait))

    def test_kokushi_ukeire_counts_only_terminal_and_honor_tiles(self):
        hand = [
            "m1", "m9", "p1", "p9", "s1", "s9", "east",
            "south", "west", "north", "haku", "hatsu", "chun",
        ]

        count, tiles = kokushi_ukeire(tiles_to_34(hand))

        self.assertEqual([
            "m1", "m9", "p1", "p9", "s1", "s9", "east",
            "south", "west", "north", "haku", "hatsu", "chun",
        ], tiles)
        self.assertEqual(39, count)

    def test_kokushi_route_discards_middle_tile(self):
        hand = [
            "m1", "m9", "p1", "p9", "s1", "s9", "east",
            "south", "west", "north", "haku", "hatsu", "chun", "m5",
        ]

        best = evaluate_all_discards(tiles_to_34(hand))[0]

        self.assertEqual("m5", best["code"])
        self.assertEqual(0, best["kokushi_shanten"])
        self.assertEqual(39, best["kokushi_ukeire"])
        self.assertEqual("kokushi", best["route"])
        self.assertEqual("TENPAI-13", best["route_label"])
        self.assertEqual(best["kokushi_tiles"], best["winning_tiles"])

    def test_standard_route_label_is_reported(self):
        hand = [
            "m1", "m2", "m3", "m4", "m5", "m6", "m7",
            "m8", "m9", "p5", "p5", "s6", "s7", "east",
        ]

        discard_east = {
            r["code"]: r for r in evaluate_all_discards(tiles_to_34(hand))
        }["east"]

        self.assertEqual("standard", discard_east["route"])
        self.assertEqual("TENPAI-STD", discard_east["route_label"])
        self.assertEqual(["s5", "s8"], discard_east["route_tiles"])

    def test_non_tenpai_result_separates_advance_and_shape_tiles(self):
        hand = [
            "m2", "m9", "p6", "p7", "p9", "p9", "s1",
            "s1", "s3", "s7", "s9", "east", "west", "haku",
        ]

        result = {
            r["code"]: r for r in evaluate_all_discards(tiles_to_34(hand))
        }["p6"]

        self.assertEqual([], result["winning_tiles"])
        self.assertEqual(result["decreases"], result["advance_tiles"])
        self.assertEqual(result["improves"], result["shape_tiles"])
        self.assertTrue({"m1", "m3", "m4"}.issubset(result["shape_tiles"]))


if __name__ == "__main__":
    unittest.main()
