import calendar
import unittest
from calendar_utils import days_in_month, is_leap

class Hidden(unittest.TestCase):
    def test_matches_stdlib(self):
        for y in [1600, 1700, 1900, 2000, 2023, 2024, 2100, 2400]:
            self.assertEqual(is_leap(y), calendar.isleap(y), y)

    def test_february_2000(self):
        self.assertEqual(days_in_month(2000, 2), 29)
