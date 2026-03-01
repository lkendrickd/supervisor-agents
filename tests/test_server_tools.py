"""Tests for the MCP tool functions in mcp_tools.py.

These are plain functions under the @mcp.tool() decorator,
so we can call them directly.
"""

import datetime
import uuid

import pytest

from mcp_tools import (
    add,
    multiply,
    sqrt,
    divide,
    power,
    percentage,
    word_count,
    char_count,
    to_uppercase,
    to_lowercase,
    reverse_text,
    now,
    generate_uuid,
    date_diff,
    random_number,
)


# ── Math tools ──


class TestAdd:
    def test_positive(self):
        assert add(2, 3) == 5

    def test_negative(self):
        assert add(-1, -2) == -3

    def test_floats(self):
        assert add(1.5, 2.5) == 4.0

    def test_zero(self):
        assert add(0, 0) == 0


class TestMultiply:
    def test_basic(self):
        assert multiply(3, 4) == 12

    def test_by_zero(self):
        assert multiply(5, 0) == 0

    def test_negative(self):
        assert multiply(-2, 3) == -6

    def test_floats(self):
        assert multiply(2.5, 4) == 10.0


class TestDivide:
    def test_basic(self):
        assert divide(10, 2) == 5

    def test_float_result(self):
        assert divide(7, 2) == 3.5

    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(1, 0)

    def test_negative(self):
        assert divide(-6, 3) == -2


class TestSqrt:
    def test_perfect_square(self):
        assert sqrt(9) == 3.0

    def test_non_perfect(self):
        assert sqrt(2) == pytest.approx(1.4142135623730951)

    def test_zero(self):
        assert sqrt(0) == 0.0


class TestPower:
    def test_basic(self):
        assert power(2, 10) == 1024

    def test_zero_exponent(self):
        assert power(5, 0) == 1

    def test_fractional_exponent(self):
        assert power(4, 0.5) == 2.0


class TestPercentage:
    def test_basic(self):
        assert percentage(1, 4) == 25.0

    def test_full(self):
        assert percentage(100, 100) == 100.0

    def test_zero_value(self):
        assert percentage(0, 50) == 0.0

    def test_zero_total(self):
        with pytest.raises(ValueError, match="Total cannot be zero"):
            percentage(10, 0)


# ── Text tools ──


class TestWordCount:
    def test_basic(self):
        assert word_count("Foo Bar Baz") == 3

    def test_single_word(self):
        assert word_count("hello") == 1

    def test_empty(self):
        assert word_count("") == 0

    def test_extra_spaces(self):
        assert word_count("  hello   world  ") == 2


class TestCharCount:
    def test_basic(self):
        assert char_count("hello") == 5

    def test_with_spaces(self):
        assert char_count("hi there") == 8

    def test_empty(self):
        assert char_count("") == 0


class TestToUppercase:
    def test_basic(self):
        assert to_uppercase("hello") == "HELLO"

    def test_mixed(self):
        assert to_uppercase("Hello World") == "HELLO WORLD"

    def test_already_upper(self):
        assert to_uppercase("ABC") == "ABC"


class TestToLowercase:
    def test_basic(self):
        assert to_lowercase("HELLO") == "hello"

    def test_mixed(self):
        assert to_lowercase("Hello World") == "hello world"


class TestReverseText:
    def test_basic(self):
        assert reverse_text("hello") == "olleh"

    def test_palindrome(self):
        assert reverse_text("racecar") == "racecar"

    def test_empty(self):
        assert reverse_text("") == ""


# ── Utility tools ──


class TestNow:
    def test_returns_iso_format(self):
        result = now()
        # Should parse without error
        datetime.datetime.fromisoformat(result)

    def test_is_recent(self):
        result = now()
        parsed = datetime.datetime.fromisoformat(result)
        delta = abs(datetime.datetime.now() - parsed)
        assert delta.total_seconds() < 2


class TestGenerateUuid:
    def test_valid_uuid(self):
        result = generate_uuid()
        parsed = uuid.UUID(result)
        assert parsed.version == 4

    def test_unique(self):
        a = generate_uuid()
        b = generate_uuid()
        assert a != b


class TestDateDiff:
    def test_basic(self):
        assert date_diff("2024-01-01", "2024-01-31") == 30

    def test_reversed_order(self):
        assert date_diff("2024-01-31", "2024-01-01") == 30

    def test_same_date(self):
        assert date_diff("2024-06-15", "2024-06-15") == 0

    def test_invalid_date(self):
        with pytest.raises(ValueError):
            date_diff("not-a-date", "2024-01-01")


class TestRandomNumber:
    def test_within_default_range(self):
        for _ in range(50):
            val = random_number()
            assert 1 <= val <= 100

    def test_custom_range(self):
        for _ in range(50):
            val = random_number(10, 20)
            assert 10 <= val <= 20

    def test_single_value_range(self):
        assert random_number(5, 5) == 5
