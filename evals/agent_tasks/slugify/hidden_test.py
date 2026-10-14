import unittest
from text_utils import slugify

class Hidden(unittest.TestCase):
    def test_spaces_and_dashes(self):
        self.assertEqual(slugify("  Many   spaces  "), "many-spaces")
        self.assertEqual(slugify("a -- b"), "a-b")

    def test_edges(self):
        self.assertEqual(slugify(""), "")
        self.assertEqual(slugify("!!!"), "")
        self.assertEqual(slugify("Version 2.0 (beta)"), "version-2-0-beta")
