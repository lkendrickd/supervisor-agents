# Supervisor-Agents

A multi-agent system built with LangGraph. One supervisor agent decides which of three specialists to call and stitches their answers together. Each specialist gets its own subset of tools from an MCP server. If two tasks are independent the supervisor calls both at once; if one needs the other's output it chains them.

## How it works

The supervisor is a ReAct agent. Its only "tools" are the three specialists, wrapped as `@tool` functions with string-in/string-out interfaces. It never sees their internals, just the docstrings. When the LLM emits multiple tool calls in one response, LangGraph runs them concurrently.

### Specialists

| Agent | Tools | Purpose |
|---|---|---|
| Mathematician | add, multiply, divide, sqrt, power, percentage, random_number, generate_uuid | Arithmetic, exponentiation, percentages, random numbers, UUIDs |
| Wordsmith | word_count, char_count, to_uppercase, to_lowercase, reverse_text | Text processing, string manipulation |
| Timekeeper | now, date_diff | Time queries, date calculations |

### Parallel vs sequential

The supervisor's ReAct loop handles both. No extra routing logic:

- "Calculate 2^10 and reverse 'hello'" calls Mathematician and Wordsmith in parallel (two tool calls in one LLM response).
- "Count the words in 'Foo Bar Baz' and multiply that count by 3" calls Wordsmith first, reads the result (3), then passes it to Mathematician (3 x 3 = 9). The chaining happens on its own.

### Architecture diagram

Open `blueprint.html` in a browser. It walks through the supervisor pattern and traces a chained-dependency example step by step.

## Prerequisites

I chose openrouter for the LLM provider to show how to swap models and providers easily. You can switch to any OpenAI-compatible API by changing the `model` argument in `create_agent()`.

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [OpenRouter](https://openrouter.ai) API key

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` with your values:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | yes | — | API key from [OpenRouter](https://openrouter.ai) |
| `SUPERVISOR_MODEL` | no | `google/gemini-2.5-flash` | Model for the supervisor (routing and synthesis) |
| `SPECIALIST_MODEL` | no | same as `SUPERVISOR_MODEL` | Model for the three specialist agents. Set to a cheaper model to save costs. |

## Running

```bash
# interactive chat
make run

# or directly
uv run python supervisor.py
```

## Files

| File | What it does |
|---|---|
| `supervisor.py` | Supervisor agent, specialist wrappers |
| `server.py` | MCP tool server (spawned via stdio) |
| `blueprint.html` | Architecture diagram |
| `Makefile` | Shortcuts |

## Notes

Keep in mind that the performance and correct tool calling depend heavily on the LLM knowledge, reasoning, and tool use capabilities.

- LLM calls go through [OpenRouter](https://openrouter.ai). Set `SUPERVISOR_MODEL` and `SPECIALIST_MODEL` in `.env` to swap models.
- The supervisor only sees final answers from specialists, not intermediate reasoning. If it misreads a result, you can change the `@tool` wrappers to return steps alongside the answer.
- Tools are served via MCP over stdio. The client spawns `server.py` as a subprocess on startup, loads all tools, then `filter_tools()` splits them across specialists.
- In a real deployment you'd run the MCP server as a standalone service over Streamable HTTP instead of spawning it inline. The subprocess approach here keeps the demo to one `uv run` command.
