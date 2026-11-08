import unittest

from kanachan.simulation.statistics import summarize


def game(proposed):
    return [
        {
            "ranking": rank,
            "score": 40000 - rank * 10000,
            "proposed": rank in proposed,
        }
        for rank in range(4)
    ]


class SimulationStatisticsTests(unittest.TestCase):
    def test_one_game_one_proposed_player_has_undefined_variance(self):
        result = summarize([game({0})])
        self.assertEqual(
            result["proposed"]["ranking"], {"average": 1.0, "variance": None}
        )
        self.assertEqual(
            result["difference"]["ranking"], {"average": 2.0, "variance": None}
        )
        self.assertEqual(result["proposed"]["num_samples"], 1)

    def test_variance_has_the_same_sample_definition_for_rates_and_ranks(self):
        result = summarize([game({0}), game({3})])
        self.assertEqual(
            result["proposed"]["top_rate"], {"average": 0.5, "variance": 0.5}
        )
        self.assertEqual(
            result["proposed"]["ranking"], {"average": 2.5, "variance": 4.5}
        )
        self.assertEqual(
            result["difference"]["ranking"], {"average": 0.0, "variance": 8.0}
        )

    def test_empty_results_and_invalid_ranks_are_rejected(self):
        for results in ([], [game(set())], [[game({0})[0]] * 4]):
            with self.subTest(results=results), self.assertRaises(ValueError):
                summarize(results)
