"""Small calendar helpers."""

DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def is_leap(year):
    return year % 4 == 0


def days_in_month(year, month):
    if not 1 <= month <= 12:
        raise ValueError("month must be 1-12")
    if month == 2 and is_leap(year):
        return 29
    return DAYS[month - 1]


def days_in_year(year):
    return 366 if is_leap(year) else 365
