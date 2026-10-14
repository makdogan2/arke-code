import unittest
from stats import median

class Hidden(unittest.TestCase):
    def test_even(self):
        self.assertEqual(median([10, 2]), 6)
        self.assertEqual(median([1, 2, 3, 4, 5, 6]), 3.5)

    def test_odd_and_unsorted(self):
        self.assertEqual(median([9, 1, 5, 3, 7]), 5)

    def test_input_not_mutated(self):
        data = [3, 1, 2, 4]
        median(data)
        self.assertEqual(data, [3, 1, 2, 4])
