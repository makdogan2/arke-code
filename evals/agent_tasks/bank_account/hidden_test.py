import unittest
from bank import Account

class Hidden(unittest.TestCase):
    def test_failed_transfer_changes_nothing(self):
        a, b = Account("a", 10), Account("b", 5)
        with self.assertRaises(ValueError):
            a.transfer(b, 50)
        self.assertEqual((a.balance, b.balance), (10, 5))

    def test_exact_balance_allowed(self):
        a = Account("a", 10)
        a.withdraw(10)
        self.assertEqual(a.balance, 0)

    def test_transfer_ok(self):
        a, b = Account("a", 10), Account("b", 0)
        a.transfer(b, 4)
        self.assertEqual((a.balance, b.balance), (6, 4))
