"""
# ID: TEST-DATA-001
# Purpose: Unit tests for src/data_loader._clean_text().
#          No network access required.
"""
from src.data_loader import _clean_text


class TestCleanText:
    def test_strips_whitespace(self):
        assert _clean_text("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert _clean_text("foo   bar") == "foo bar"

    def test_collapses_newlines(self):
        assert _clean_text("a\n\nb") == "a b"

    def test_empty_string(self):
        assert _clean_text("") == ""

    def test_already_clean(self):
        assert _clean_text("hello world") == "hello world"
