import unittest

from bank import Account


class TestAccount(unittest.TestCase):
    def test_deposit_and_withdraw(self):
        a = Account("ada", 100)
        a.deposit(50)
        a.withdraw(30)
        self.assertEqual(a.balance, 120)

    def test_overdraft_is_refused(self):
        a = Account("ada", 20)
        with self.assertRaises(ValueError):
            a.withdraw(50)
        self.assertEqual(a.balance, 20)


if __name__ == "__main__":
    unittest.main()
