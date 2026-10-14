import unittest
from pages import paginate

class Hidden(unittest.TestCase):
    def test_middle_and_last(self):
        items = list(range(10))
        self.assertEqual(paginate(items, 2, 3), [3, 4, 5])
        self.assertEqual(paginate(items, 4, 3), [9])

    def test_past_end(self):
        self.assertEqual(paginate(list(range(10)), 5, 3), [])

    def test_pages_cover_everything_once(self):
        items = list(range(17))
        seen = [x for p in range(1, 5) for x in paginate(items, p, 5)]
        self.assertEqual(seen, items)
