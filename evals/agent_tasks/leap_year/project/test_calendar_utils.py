import unittest

from calendar_utils import days_in_month, days_in_year


class TestCalendar(unittest.TestCase):
    def test_regular_months(self):
        self.assertEqual(days_in_month(2023, 4), 30)
        self.assertEqual(days_in_month(2024, 2), 29)

    def test_century_year(self):
        self.assertEqual(days_in_month(1900, 2), 28)
        self.assertEqual(days_in_year(1900), 365)


if __name__ == "__main__":
    unittest.main()
