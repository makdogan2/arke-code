import unittest

from pages import page_count, paginate


class TestPages(unittest.TestCase):
    def test_first_page(self):
        self.assertEqual(paginate(list(range(10)), 1, 3), [0, 1, 2])

    def test_page_count(self):
        self.assertEqual(page_count(10, 3), 4)
        self.assertEqual(page_count(0, 3), 0)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            paginate([1], 0, 3)


if __name__ == "__main__":
    unittest.main()
