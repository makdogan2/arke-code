"""Custom eval set for Arke Code.

Two kinds of tasks:

- "implement": the model writes a function or class. We run its code followed by
  `tests` (plain asserts). The task passes if the script exits cleanly.
- "write_tests": the model writes tests for a given function. Good tests must pass
  on the correct `reference` AND fail on every buggy `mutant`. They must also avoid
  `== True` / `== False` comparisons (style check).

Every task carries a `reference` solution so `python eval.py --check` can prove the
eval set itself is correct before we trust any score.
"""

TASKS = [
    # ------------------------------------------------------------------ strings
    {
        "id": "palindrome",
        "kind": "implement",
        "prompt": "Write a Python function `is_palindrome(s: str) -> bool` that returns True if "
                  "`s` is a palindrome, ignoring case and any non-alphanumeric characters.",
        "reference": '''
def is_palindrome(s: str) -> bool:
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
''',
        "tests": '''
assert is_palindrome("A man, a plan, a canal: Panama")
assert is_palindrome("")
assert is_palindrome("No 'x' in Nixon")
assert not is_palindrome("race a car")
assert not is_palindrome("ab")
''',
    },
    {
        "id": "valid_parentheses",
        "kind": "implement",
        "prompt": "Write a Python function `is_balanced(s: str) -> bool` that checks whether the "
                  "brackets `()`, `[]` and `{}` in `s` are balanced and correctly nested. "
                  "Ignore all other characters.",
        "reference": '''
def is_balanced(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for c in s:
        if c in "([{":
            stack.append(c)
        elif c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack
''',
        "tests": '''
assert is_balanced("()[]{}")
assert is_balanced("f(a[b]{c})")
assert is_balanced("")
assert not is_balanced("(]")
assert not is_balanced("([)]")
assert not is_balanced("((")
assert not is_balanced("}")
''',
    },
    {
        "id": "run_length",
        "kind": "implement",
        "prompt": "Write two Python functions: `rle_encode(s: str) -> str` that run-length encodes a "
                  "string as count followed by character (\"aaabcc\" -> \"3a1b2c\"), and "
                  "`rle_decode(s: str) -> str` that reverses it. Counts can have several digits. "
                  "Input strings contain only letters.",
        "reference": '''
import re

def rle_encode(s: str) -> str:
    if not s:
        return ""
    out, prev, count = [], s[0], 1
    for c in s[1:]:
        if c == prev:
            count += 1
        else:
            out.append(f"{count}{prev}")
            prev, count = c, 1
    out.append(f"{count}{prev}")
    return "".join(out)

def rle_decode(s: str) -> str:
    return "".join(ch * int(n) for n, ch in re.findall(r"(\\d+)(\\D)", s))
''',
        "tests": '''
assert rle_encode("aaabcc") == "3a1b2c"
assert rle_encode("") == ""
assert rle_encode("a" * 12) == "12a"
assert rle_decode("3a1b2c") == "aaabcc"
assert rle_decode("12a1b") == "a" * 12 + "b"
for text in ["x", "xyz", "zzzzzzzzzzzyy", "abcabc"]:
    assert rle_decode(rle_encode(text)) == text
''',
    },
    {
        "id": "group_anagrams",
        "kind": "implement",
        "prompt": "Write a Python function `group_anagrams(words: list[str]) -> list[list[str]]` that "
                  "groups words that are anagrams of each other. Keep words inside each group in "
                  "their original order, and order the groups by the first appearance of their "
                  "first word.",
        "reference": '''
def group_anagrams(words):
    groups = {}
    for w in words:
        groups.setdefault("".join(sorted(w)), []).append(w)
    return list(groups.values())
''',
        "tests": '''
assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
assert group_anagrams([]) == []
assert group_anagrams(["a"]) == [["a"]]
assert group_anagrams(["ab", "ba", "ab"]) == [["ab", "ba", "ab"]]
''',
    },
    {
        "id": "parse_duration",
        "kind": "implement",
        "prompt": "Write a Python function `parse_duration(text: str) -> int` that converts strings "
                  "like \"1h30m\", \"45s\", \"2h5s\" or \"1h2m3s\" into a total number of seconds. "
                  "Units are h, m and s, each appears at most once, always in that order. Raise "
                  "ValueError for an empty or invalid string.",
        "reference": '''
import re

def parse_duration(text: str) -> int:
    m = re.fullmatch(r"(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", text or "")
    if not text or not m:
        raise ValueError(f"invalid duration: {text!r}")
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s
''',
        "tests": '''
assert parse_duration("1h30m") == 5400
assert parse_duration("45s") == 45
assert parse_duration("2h5s") == 7205
assert parse_duration("1h2m3s") == 3723
assert parse_duration("90m") == 5400
for bad in ["", "abc", "5x", "1m1h"]:
    try:
        parse_duration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"no ValueError for {bad!r}")
''',
    },
    {
        "id": "roman_to_int",
        "kind": "implement",
        "prompt": "Write a Python function `roman_to_int(s: str) -> int` that converts a valid Roman "
                  "numeral (1 to 3999) to an integer.",
        "reference": '''
def roman_to_int(s: str) -> int:
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, c in enumerate(s):
        if i + 1 < len(s) and v[c] < v[s[i + 1]]:
            total -= v[c]
        else:
            total += v[c]
    return total
''',
        "tests": '''
assert roman_to_int("III") == 3
assert roman_to_int("IV") == 4
assert roman_to_int("IX") == 9
assert roman_to_int("LVIII") == 58
assert roman_to_int("MCMXCIV") == 1994
assert roman_to_int("MMMCMXCIX") == 3999
''',
    },
    # ------------------------------------------------------------- algorithms
    {
        "id": "two_sum",
        "kind": "implement",
        "prompt": "Write a Python function `two_sum(nums: list[int], target: int) -> tuple[int, int] | None` "
                  "that returns the indices (i, j) with i < j of the first pair (smallest j, then smallest i) "
                  "whose values add up to `target`, or None if there is no such pair. It must run in O(n).",
        "reference": '''
def two_sum(nums, target):
    seen = {}
    for j, x in enumerate(nums):
        if target - x in seen:
            return (seen[target - x], j)
        seen.setdefault(x, j)
    return None
''',
        "tests": '''
assert two_sum([2, 7, 11, 15], 9) == (0, 1)
assert two_sum([3, 2, 4], 6) == (1, 2)
assert two_sum([3, 3], 6) == (0, 1)
assert two_sum([1, 2, 3], 100) is None
assert two_sum([], 0) is None
assert two_sum([1, 5, 1, 5], 6) == (0, 1)
''',
    },
    {
        "id": "binary_search",
        "kind": "implement",
        "prompt": "Write a Python function `lower_bound(arr: list[int], x: int) -> int` that returns "
                  "the first index i such that arr[i] >= x in the sorted list `arr` (len(arr) if no such "
                  "index). Use binary search, do not use the bisect module.",
        "reference": '''
def lower_bound(arr, x):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        "tests": '''
import bisect, random
assert lower_bound([], 5) == 0
assert lower_bound([1, 2, 2, 2, 3], 2) == 1
assert lower_bound([1, 2, 3], 10) == 3
assert lower_bound([1, 2, 3], -1) == 0
rng = random.Random(0)
for _ in range(200):
    arr = sorted(rng.randint(0, 20) for _ in range(rng.randint(0, 15)))
    x = rng.randint(-2, 22)
    assert lower_bound(arr, x) == bisect.bisect_left(arr, x)
''',
    },
    {
        "id": "merge_intervals",
        "kind": "implement",
        "prompt": "Write a Python function `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]` "
                  "that merges overlapping or touching closed intervals and returns them sorted by start. "
                  "The input may be unsorted.",
        "reference": '''
def merge_intervals(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out
''',
        "tests": '''
assert merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]
assert merge_intervals([(1, 4), (4, 5)]) == [(1, 5)]
assert merge_intervals([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]
assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]
assert merge_intervals([]) == []
''',
    },
    {
        "id": "top_k_words",
        "kind": "implement",
        "prompt": "Write a Python function `top_k_words(text: str, k: int) -> list[tuple[str, int]]` that "
                  "returns the k most frequent words with their counts. Words are case-insensitive "
                  "sequences of letters (a-z). Sort by count descending, then alphabetically.",
        "reference": '''
import re
from collections import Counter

def top_k_words(text, k):
    counts = Counter(re.findall(r"[a-z]+", text.lower()))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
''',
        "tests": '''
assert top_k_words("the cat and the hat. The END!", 2) == [("the", 3), ("and", 1)]
assert top_k_words("b a b a c", 3) == [("a", 2), ("b", 2), ("c", 1)]
assert top_k_words("", 5) == []
assert top_k_words("Hello, hello HELLO", 1) == [("hello", 3)]
''',
    },
    {
        "id": "rotate_matrix",
        "kind": "implement",
        "prompt": "Write a Python function `rotate(matrix: list[list[int]]) -> list[list[int]]` that "
                  "returns a new n x m matrix rotated 90 degrees clockwise. Do not modify the input.",
        "reference": '''
def rotate(matrix):
    return [list(row) for row in zip(*matrix[::-1])]
''',
        "tests": '''
m = [[1, 2, 3], [4, 5, 6]]
assert rotate(m) == [[4, 1], [5, 2], [6, 3]]
assert m == [[1, 2, 3], [4, 5, 6]]
assert rotate([[1]]) == [[1]]
assert rotate([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]
''',
    },
    {
        "id": "primes",
        "kind": "implement",
        "prompt": "Write a Python function `primes_up_to(n: int) -> list[int]` that returns all primes "
                  "<= n using the Sieve of Eratosthenes. It should handle n = 1_000_000 in well under a second.",
        "reference": '''
def primes_up_to(n):
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(range(i * i, n + 1, i)))
    return [i for i, p in enumerate(sieve) if p]
''',
        "tests": '''
import time
assert primes_up_to(1) == []
assert primes_up_to(2) == [2]
assert primes_up_to(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
t = time.time()
assert len(primes_up_to(1_000_000)) == 78498
assert time.time() - t < 3
''',
    },
    {
        "id": "fib_big",
        "kind": "implement",
        "prompt": "Write a Python function `fib(n: int) -> int` returning the n-th Fibonacci number "
                  "(fib(0) = 0, fib(1) = 1). It must be fast for n = 10_000 and must not hit the "
                  "recursion limit.",
        "reference": '''
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
        "tests": '''
assert [fib(i) for i in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
assert fib(90) == 2880067194370816120
assert len(str(fib(10_000))) == 2090
''',
    },
    # ------------------------------------------------------- data structures
    {
        "id": "flatten",
        "kind": "implement",
        "prompt": "Write a Python function `flatten(items: list) -> list` that flattens arbitrarily "
                  "nested lists and tuples into a single list. Strings must NOT be split into characters.",
        "reference": '''
def flatten(items):
    out = []
    for x in items:
        if isinstance(x, (list, tuple)):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out
''',
        "tests": '''
assert flatten([1, [2, [3, (4, 5)]], 6]) == [1, 2, 3, 4, 5, 6]
assert flatten([]) == []
assert flatten(["ab", ["cd", []]]) == ["ab", "cd"]
assert flatten([[[[[]]]]]) == []
''',
    },
    {
        "id": "dedupe",
        "kind": "implement",
        "prompt": "Write a Python function `dedupe(items: list) -> list` that removes duplicates while "
                  "keeping the first occurrence order. Items are hashable.",
        "reference": '''
def dedupe(items):
    return list(dict.fromkeys(items))
''',
        "tests": '''
assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
assert dedupe([]) == []
assert dedupe(["a", "A", "a"]) == ["a", "A"]
assert dedupe([1, True, 1.0]) == [1]
''',
    },
    {
        "id": "deep_merge",
        "kind": "implement",
        "prompt": "Write a Python function `deep_merge(a: dict, b: dict) -> dict` that returns a new dict "
                  "merging b into a. When both values for a key are dicts, merge them recursively; "
                  "otherwise the value from b wins. Neither input may be modified.",
        "reference": '''
import copy

def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
''',
        "tests": '''
a = {"x": 1, "cfg": {"lr": 0.1, "opt": {"name": "adam"}}}
b = {"y": 2, "cfg": {"lr": 0.01, "opt": {"beta": 0.9}}}
r = deep_merge(a, b)
assert r == {"x": 1, "y": 2, "cfg": {"lr": 0.01, "opt": {"name": "adam", "beta": 0.9}}}
assert a == {"x": 1, "cfg": {"lr": 0.1, "opt": {"name": "adam"}}}
assert b == {"y": 2, "cfg": {"lr": 0.01, "opt": {"beta": 0.9}}}
r["cfg"]["opt"]["name"] = "sgd"
assert a["cfg"]["opt"]["name"] == "adam"
assert deep_merge({"k": {"a": 1}}, {"k": 5}) == {"k": 5}
''',
    },
    {
        "id": "lru_cache",
        "kind": "implement",
        "prompt": "Write a Python class `LRUCache` with `__init__(self, capacity: int)`, "
                  "`get(self, key) -> value or -1 if missing` and `put(self, key, value)`. When the "
                  "cache is full, `put` evicts the least recently used key. Both `get` and `put` "
                  "count as a use. All operations must be O(1).",
        "reference": '''
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
''',
        "tests": '''
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
c.put(4, 4)
assert c.get(1) == -1
assert c.get(3) == 3
assert c.get(4) == 4
c.put(3, 30)
c.put(5, 5)
assert c.get(4) == -1
assert c.get(3) == 30
''',
    },
    {
        "id": "min_stack",
        "kind": "implement",
        "prompt": "Write a Python class `MinStack` with methods `push(x)`, `pop() -> x`, `top() -> x` and "
                  "`get_min() -> x`, all O(1). `pop`, `top` and `get_min` raise IndexError on an empty stack.",
        "reference": '''
class MinStack:
    def __init__(self):
        self.items = []

    def push(self, x):
        m = min(x, self.items[-1][1]) if self.items else x
        self.items.append((x, m))

    def pop(self):
        if not self.items:
            raise IndexError("pop from empty stack")
        return self.items.pop()[0]

    def top(self):
        if not self.items:
            raise IndexError("top of empty stack")
        return self.items[-1][0]

    def get_min(self):
        if not self.items:
            raise IndexError("min of empty stack")
        return self.items[-1][1]
''',
        "tests": '''
s = MinStack()
for x in [5, 3, 7, 3, 1]:
    s.push(x)
assert s.get_min() == 1
assert s.pop() == 1
assert s.get_min() == 3
s.pop()
assert s.get_min() == 3
assert s.top() == 7
s.pop(); s.pop()
assert s.get_min() == 5
s.pop()
for method in (s.pop, s.top, s.get_min):
    try:
        method()
    except IndexError:
        pass
    else:
        raise AssertionError(f"{method.__name__} on empty stack did not raise IndexError")
''',
    },
    # ---------------------------------------------------------------- debugging
    {
        "id": "fix_average",
        "kind": "implement",
        "prompt": "This function is supposed to return the average of the numbers, or None for an empty "
                  "list, but it has bugs. Return the fixed function, keeping its name.\n\n"
                  "```python\n"
                  "def average(nums):\n"
                  "    total = 0\n"
                  "    for i in range(1, len(nums)):\n"
                  "        total += nums[i]\n"
                  "    return total // len(nums)\n"
                  "```",
        "reference": '''
def average(nums):
    if not nums:
        return None
    return sum(nums) / len(nums)
''',
        "tests": '''
assert average([1, 2, 3, 4]) == 2.5
assert average([10]) == 10
assert average([]) is None
assert average([-1, 1]) == 0
''',
    },
    {
        "id": "fix_mutable_default",
        "kind": "implement",
        "prompt": "Calling this function twice gives surprising results. Explain nothing, just return the "
                  "fixed function with the same name and signature semantics (tags is optional).\n\n"
                  "```python\n"
                  "def add_tag(tag, tags=[]):\n"
                  "    tags.append(tag)\n"
                  "    return tags\n"
                  "```",
        "reference": '''
def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
''',
        "tests": '''
assert add_tag("a") == ["a"]
assert add_tag("b") == ["b"]
mine = ["x"]
assert add_tag("y", mine) == ["x", "y"]
assert mine == ["x", "y"]
''',
    },
    # ----------------------------------------------------------- writing tests
    {
        "id": "tests_clamp",
        "kind": "write_tests",
        "prompt": "Write tests for this function as plain Python assert statements (no pytest, no "
                  "unittest, do not redefine the function). Cover normal values and the edge cases.\n\n"
                  "```python\n"
                  "def clamp(x, lo, hi):\n"
                  "    \"\"\"Limit x to the closed range [lo, hi]. Raise ValueError if lo > hi.\"\"\"\n"
                  "    if lo > hi:\n"
                  "        raise ValueError(\"lo > hi\")\n"
                  "    return max(lo, min(x, hi))\n"
                  "```",
        "reference": '''
def clamp(x, lo, hi):
    if lo > hi:
        raise ValueError("lo > hi")
    return max(lo, min(x, hi))
''',
        "mutants": [
            # forgets the upper bound
            '''
def clamp(x, lo, hi):
    if lo > hi:
        raise ValueError("lo > hi")
    return max(lo, x)
''',
            # never raises
            '''
def clamp(x, lo, hi):
    return max(lo, min(x, hi))
''',
            # off by one at the lower edge
            '''
def clamp(x, lo, hi):
    if lo > hi:
        raise ValueError("lo > hi")
    return max(lo + 1, min(x, hi)) if x < lo else max(lo, min(x, hi))
''',
        ],
        "bug_notes": ["ignores the upper bound", "never raises ValueError",
                      "returns lo + 1 below the range"],
        "example_tests": '''
assert clamp(5, 0, 10) == 5
assert clamp(-3, 0, 10) == 0
assert clamp(42, 0, 10) == 10
assert clamp(0, 0, 0) == 0
try:
    clamp(1, 5, 2)
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError")
''',
    },
    {
        "id": "tests_is_leap",
        "kind": "write_tests",
        "prompt": "Write tests for this function as plain Python assert statements (no pytest, no "
                  "unittest, do not redefine the function).\n\n"
                  "```python\n"
                  "def is_leap(year: int) -> bool:\n"
                  "    \"\"\"Gregorian leap year rule.\"\"\"\n"
                  "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n"
                  "```",
        "reference": '''
def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
''',
        "mutants": [
            '''
def is_leap(year: int) -> bool:
    return year % 4 == 0
''',
            '''
def is_leap(year: int) -> bool:
    return year % 4 == 0 and year % 100 != 0
''',
            '''
def is_leap(year: int) -> bool:
    return year % 400 == 0
''',
        ],
        "bug_notes": ["calls 1900 a leap year", "misses the 400-year rule (2000)",
                      "only knows the 400-year rule (2024)"],
        "example_tests": '''
assert is_leap(2024)
assert is_leap(2000)
assert not is_leap(1900)
assert not is_leap(2023)
''',
    },
]
