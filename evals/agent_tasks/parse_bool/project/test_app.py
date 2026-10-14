import unittest

from app import load_settings


class TestSettings(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(load_settings({}), {"debug": False, "workers": 2})

    def test_values(self):
        s = load_settings({"DEBUG": "true", "WORKERS": "8"})
        self.assertTrue(s["debug"])
        self.assertEqual(s["workers"], 8)

    def test_debug_off(self):
        self.assertFalse(load_settings({"DEBUG": "False"})["debug"])


if __name__ == "__main__":
    unittest.main()
