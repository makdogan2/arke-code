import unittest

from text_utils import slugify, truncate


class TestTextUtils(unittest.TestCase):
    def test_slugify_simple(self):
        self.assertEqual(slugify("Python"), "python")

    def test_slugify_punctuation(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_truncate(self):
        self.assertEqual(truncate("abcdefgh", 6), "abc...")
        self.assertEqual(truncate("abc", 6), "abc")


if __name__ == "__main__":
    unittest.main()
