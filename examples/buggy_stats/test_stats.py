import unittest

from stats import mean, median, mode


class TestStats(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([1, 2, 3, 4]), 2.5)

    def test_median_odd(self):
        self.assertEqual(median([3, 1, 2]), 2)

    def test_median_even(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_mode_first_wins_ties(self):
        self.assertEqual(mode([2, 1, 1, 2, 3]), 2)

    def test_empty(self):
        for f in (mean, median, mode):
            with self.assertRaises(ValueError):
                f([])


if __name__ == "__main__":
    unittest.main()
