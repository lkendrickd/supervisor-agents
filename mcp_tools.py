"""
mcp_tools.py
============
MCP tool server. Runs over stdio — supervisor.py spawns this as a subprocess.

Tools are organized by category so supervisor.py can assign subsets
to specialist agents:
  - Math:    add, multiply, sqrt, divide, power, percentage, random_number, generate_uuid
  - Text:    word_count, char_count, to_uppercase, to_lowercase, reverse_text
  - Time:    now, date_diff
  - File:    read_file, list_files, create_file, delete_file

  uv run python mcp_tools.py         # standalone (prints to stdout)
"""

import math
import datetime
import uuid
import random
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my_tools")


# ─────────────────────────────────────────────
# MATH TOOLS
# ─────────────────────────────────────────────


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@mcp.tool()
def sqrt(n: float) -> float:
    """Return the square root of a number."""
    return math.sqrt(n)


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b. Returns an error if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base**exponent


@mcp.tool()
def percentage(value: float, total: float) -> float:
    """Calculate what percentage `value` is of `total`. Returns (value/total)*100. E.g. percentage(1, 4) = 25.0 meaning 1 is 25% of 4. To compute '25% of 400', use multiply(400, 25) then divide by 100 instead."""
    if total == 0:
        raise ValueError("Total cannot be zero")
    return (value / total) * 100


@mcp.tool()
def random_number(min_val: int = 1, max_val: int = 100) -> int:
    """Generate a random integer between min_val and max_val (inclusive). Defaults to 1-100."""
    return random.randint(min_val, max_val)


@mcp.tool()
def generate_uuid() -> str:
    """Generate a random UUID."""
    return str(uuid.uuid4())


# ─────────────────────────────────────────────
# TEXT TOOLS
# ─────────────────────────────────────────────


@mcp.tool()
def word_count(text: str) -> int:
    """Count the number of words in a text string."""
    return len(text.split())


@mcp.tool()
def char_count(text: str) -> int:
    """Count the number of characters in a text string (including spaces)."""
    return len(text)


@mcp.tool()
def to_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


@mcp.tool()
def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


@mcp.tool()
def reverse_text(text: str) -> str:
    """Reverse a string. 'hello' becomes 'olleh'."""
    return text[::-1]


# ─────────────────────────────────────────────
# TIME TOOLS
# ─────────────────────────────────────────────


@mcp.tool()
def now() -> str:
    """Return the current date and time in ISO-8601 format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@mcp.tool()
def date_diff(date1: str, date2: str) -> int:
    """Return the number of days between two ISO dates (e.g. '2024-01-01'). Result is absolute."""
    d1 = datetime.date.fromisoformat(date1)
    d2 = datetime.date.fromisoformat(date2)
    return abs((d2 - d1).days)


# ─────────────────────────────────────────────
# FILE TOOLS
# ─────────────────────────────────────────────

PROJECT_DIR = os.environ.get(
    "MCP_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__))
)


def _safe_path(path: str) -> str:
    """Resolve *path* relative to PROJECT_DIR and reject traversal attempts."""
    resolved = os.path.normpath(os.path.join(PROJECT_DIR, path))
    if not resolved.startswith(os.path.normpath(PROJECT_DIR)):
        raise ValueError(f"Path '{path}' escapes the project directory")
    return resolved


@mcp.tool()
def read_file(path: str) -> str:
    """Read and return the contents of a file. Path is relative to the project root."""
    resolved = _safe_path(path)
    if not os.path.isfile(resolved):
        return f"Error: file '{path}' does not exist."
    with open(resolved, encoding="utf-8") as f:
        return f.read()


@mcp.tool()
def list_files(directory: str = ".") -> str:
    """List entries in a directory. Returns one entry per line. Path is relative to the project root."""
    resolved = _safe_path(directory)
    if not os.path.isdir(resolved):
        return f"Error: directory '{directory}' does not exist."
    entries = sorted(os.listdir(resolved))
    return "\n".join(entries)


@mcp.tool()
def create_file(path: str, content: str) -> str:
    """Create (or overwrite) a file with the given content. Path is relative to the project root."""
    resolved = _safe_path(path)
    os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Created {path}"


@mcp.tool()
def delete_file(path: str) -> str:
    """Delete a file. Path is relative to the project root."""
    resolved = _safe_path(path)
    if not os.path.isfile(resolved):
        return f"Error: file '{path}' does not exist."
    os.remove(resolved)
    return f"Deleted {path}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
