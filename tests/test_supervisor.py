"""Tests for supervisor.py tool routing and filtering logic.

These test the constants and filter_tools() without requiring
an LLM or MCP server.
"""

from types import SimpleNamespace

from supervisor import (
    MATH_TOOL_NAMES,
    TEXT_TOOL_NAMES,
    UTILITY_TOOL_NAMES,
    FILE_READ_TOOL_NAMES,
    FILE_WRITE_TOOL_NAMES,
    filter_tools,
)


# All tool names that the MCP server exposes (must stay in sync with mcp_tools.py).
ALL_TOOL_NAMES = {
    "add",
    "multiply",
    "sqrt",
    "divide",
    "power",
    "percentage",
    "word_count",
    "char_count",
    "to_uppercase",
    "to_lowercase",
    "reverse_text",
    "now",
    "generate_uuid",
    "date_diff",
    "random_number",
    "read_file",
    "list_files",
    "create_file",
    "delete_file",
}


def _fake_tools(names: set[str]) -> list:
    """Create fake tool objects with a .name attribute."""
    return [SimpleNamespace(name=n) for n in names]


ALL_SETS = [
    MATH_TOOL_NAMES,
    TEXT_TOOL_NAMES,
    UTILITY_TOOL_NAMES,
    FILE_READ_TOOL_NAMES,
    FILE_WRITE_TOOL_NAMES,
]


class TestToolNameSets:
    def test_no_overlap_math_text(self):
        assert MATH_TOOL_NAMES & TEXT_TOOL_NAMES == set()

    def test_no_overlap_math_utility(self):
        assert MATH_TOOL_NAMES & UTILITY_TOOL_NAMES == set()

    def test_no_overlap_text_utility(self):
        assert TEXT_TOOL_NAMES & UTILITY_TOOL_NAMES == set()

    def test_no_overlap_file_read_write(self):
        assert FILE_READ_TOOL_NAMES & FILE_WRITE_TOOL_NAMES == set()

    def test_no_overlap_file_read_math(self):
        assert FILE_READ_TOOL_NAMES & MATH_TOOL_NAMES == set()

    def test_no_overlap_file_read_text(self):
        assert FILE_READ_TOOL_NAMES & TEXT_TOOL_NAMES == set()

    def test_no_overlap_file_read_utility(self):
        assert FILE_READ_TOOL_NAMES & UTILITY_TOOL_NAMES == set()

    def test_no_overlap_file_write_math(self):
        assert FILE_WRITE_TOOL_NAMES & MATH_TOOL_NAMES == set()

    def test_no_overlap_file_write_text(self):
        assert FILE_WRITE_TOOL_NAMES & TEXT_TOOL_NAMES == set()

    def test_no_overlap_file_write_utility(self):
        assert FILE_WRITE_TOOL_NAMES & UTILITY_TOOL_NAMES == set()

    def test_all_tools_assigned(self):
        assigned = (
            MATH_TOOL_NAMES
            | TEXT_TOOL_NAMES
            | UTILITY_TOOL_NAMES
            | FILE_READ_TOOL_NAMES
            | FILE_WRITE_TOOL_NAMES
        )
        assert assigned == ALL_TOOL_NAMES

    def test_random_number_in_math(self):
        assert "random_number" in MATH_TOOL_NAMES

    def test_generate_uuid_in_math(self):
        assert "generate_uuid" in MATH_TOOL_NAMES

    def test_now_in_utility(self):
        assert "now" in UTILITY_TOOL_NAMES

    def test_date_diff_in_utility(self):
        assert "date_diff" in UTILITY_TOOL_NAMES

    def test_read_file_in_file_read(self):
        assert "read_file" in FILE_READ_TOOL_NAMES

    def test_list_files_in_file_read(self):
        assert "list_files" in FILE_READ_TOOL_NAMES

    def test_create_file_in_file_write(self):
        assert "create_file" in FILE_WRITE_TOOL_NAMES

    def test_delete_file_in_file_write(self):
        assert "delete_file" in FILE_WRITE_TOOL_NAMES


class TestFilterTools:
    def test_filters_correctly(self):
        all_tools = _fake_tools(ALL_TOOL_NAMES)
        math_tools = filter_tools(all_tools, MATH_TOOL_NAMES)
        names = {t.name for t in math_tools}
        assert names == MATH_TOOL_NAMES

    def test_filters_text(self):
        all_tools = _fake_tools(ALL_TOOL_NAMES)
        text_tools = filter_tools(all_tools, TEXT_TOOL_NAMES)
        names = {t.name for t in text_tools}
        assert names == TEXT_TOOL_NAMES

    def test_filters_utility(self):
        all_tools = _fake_tools(ALL_TOOL_NAMES)
        util_tools = filter_tools(all_tools, UTILITY_TOOL_NAMES)
        names = {t.name for t in util_tools}
        assert names == UTILITY_TOOL_NAMES

    def test_empty_names(self):
        all_tools = _fake_tools(ALL_TOOL_NAMES)
        assert filter_tools(all_tools, set()) == []

    def test_empty_tools(self):
        assert filter_tools([], MATH_TOOL_NAMES) == []

    def test_filters_file_read(self):
        all_tools = _fake_tools(ALL_TOOL_NAMES)
        file_read = filter_tools(all_tools, FILE_READ_TOOL_NAMES)
        names = {t.name for t in file_read}
        assert names == FILE_READ_TOOL_NAMES

    def test_filters_file_write(self):
        all_tools = _fake_tools(ALL_TOOL_NAMES)
        file_write = filter_tools(all_tools, FILE_WRITE_TOOL_NAMES)
        names = {t.name for t in file_write}
        assert names == FILE_WRITE_TOOL_NAMES

    def test_filters_all_file_tools(self):
        all_tools = _fake_tools(ALL_TOOL_NAMES)
        file_tools = filter_tools(
            all_tools, FILE_READ_TOOL_NAMES | FILE_WRITE_TOOL_NAMES
        )
        names = {t.name for t in file_tools}
        assert names == FILE_READ_TOOL_NAMES | FILE_WRITE_TOOL_NAMES

    def test_unknown_names_ignored(self):
        all_tools = _fake_tools({"add", "multiply"})
        result = filter_tools(all_tools, {"add", "nonexistent"})
        assert len(result) == 1
        assert result[0].name == "add"
