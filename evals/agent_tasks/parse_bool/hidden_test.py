import unittest
from config import parse_bool

class Hidden(unittest.TestCase):
    def test_true_values(self):
        for v in ["true", "TRUE", "yes", "On", "1"]:
            self.assertIs(parse_bool(v), True, v)

    def test_false_values(self):
        for v in ["false", "No", "off", "0", "FALSE"]:
            self.assertIs(parse_bool(v), False, v)

    def test_invalid(self):
        for v in ["maybe", "", "2"]:
            with self.assertRaises(ValueError):
                parse_bool(v)
