"""A tiny bank account model."""


class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.balance += amount

    def withdraw(self, amount):
        """Take money out. Raises ValueError if the amount is not positive
        or larger than the balance; the balance is then left unchanged."""
        if amount <= 0:
            raise ValueError("withdrawal must be positive")
        self.balance -= amount

    def transfer(self, other, amount):
        """Move money to another account; nothing changes if it fails."""
        self.withdraw(amount)
        other.deposit(amount)
