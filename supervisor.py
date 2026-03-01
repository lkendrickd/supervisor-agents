"""
supervisor.py
=============
Multiple specialized agents coordinated by a supervisor, with tools loaded
from an MCP server. The supervisor is a ReAct agent whose "tools" are the
sub-agents — it decides which specialists to invoke, can call multiple in
parallel, and synthesizes their results into a unified response.

HOW IT WORKS (Agents-as-tools / LLM-driven parallelism):
  The supervisor is itself a ReAct agent whose "tools" are the sub-agents.
  When it sees a request that spans specialties, the LLM can call multiple
  agent-tools in one response. LangGraph runs those calls concurrently.
  The model decides when to parallelize.

SPECIALIST AGENTS:
  - Mathematician — math/number tools (add, multiply, sqrt, divide, power, percentage, random_number, generate_uuid)
  - Wordsmith     — text tools (word_count, char_count, to_uppercase, to_lowercase, reverse_text)
  - Timekeeper    — temporal tools (now, date_diff)

Run:
  uv run python supervisor.py
"""

import os
import asyncio
from typing import Any

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import tool

# create_agent from langchain.agents — builds a ReAct agent (CompiledStateGraph)
# with a model + tools. Accepts `system_prompt` for persona instructions and
# `name` for labeling the agent in multi-agent setups.
from langchain.agents import create_agent

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv(override=True)


# ─────────────────────────────────────────────
# 1. MODEL CONFIG
#    Uses OpenRouter as the LLM provider via the
#    OpenAI-compatible API. SUPERVISOR_MODEL runs
#    the orchestrator, SPECIALIST_MODEL runs the
#    agents. Set SPECIALIST_MODEL to a cheaper
#    model to save costs on tool-calling tasks.
# ─────────────────────────────────────────────

SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "google/gemini-2.5-flash")
SPECIALIST_MODEL = os.environ.get("SPECIALIST_MODEL", SUPERVISOR_MODEL)

_openrouter_kwargs = {
    "max_completion_tokens": 1024,
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": SecretStr(os.environ["OPENROUTER_API_KEY"]),
}

supervisor_llm = ChatOpenAI(model=SUPERVISOR_MODEL, **_openrouter_kwargs)
specialist_llm = ChatOpenAI(model=SPECIALIST_MODEL, **_openrouter_kwargs)


# ─────────────────────────────────────────────
# 2. MCP SERVER CONFIG
#    The MCP client spawns mcp_tools.py as a
#    subprocess and talks to it over stdin/stdout.
# ─────────────────────────────────────────────

MCP_SERVERS: dict[str, dict[str, Any]] = {
    "my_tools": {
        "command": "uv",
        "args": ["run", "python", "mcp_tools.py"],
        "transport": "stdio",
    },
}


# ─────────────────────────────────────────────
# 3. LOAD & FILTER TOOLS
#    The MCP server exposes all tools in one flat
#    list. We load them all, then split them into
#    subsets so each specialist agent only sees
#    the tools relevant to its specialty.
# ─────────────────────────────────────────────

# Which tools belong to which specialist.
MATH_TOOL_NAMES = {
    "add",
    "multiply",
    "sqrt",
    "divide",
    "power",
    "percentage",
    "random_number",
    "generate_uuid",
}
TEXT_TOOL_NAMES = {
    "word_count",
    "char_count",
    "to_uppercase",
    "to_lowercase",
    "reverse_text",
}
UTILITY_TOOL_NAMES = {"now", "date_diff"}


def filter_tools(all_tools: list, names: set[str]) -> list:
    """Pick tools whose name is in the given set."""
    return [t for t in all_tools if t.name in names]


# ─────────────────────────────────────────────
# 4. CREATE SPECIALIST AGENTS
#    Each agent gets a filtered subset of MCP
#    tools, a system prompt defining its persona,
#    and a name for identification in logs.
# ─────────────────────────────────────────────


def create_specialists(all_tools: list) -> dict:
    """Build the three specialist agents from the full MCP tool list.

    Returns a dict mapping persona name → compiled agent.
    """
    math_tools = filter_tools(all_tools, MATH_TOOL_NAMES)
    text_tools = filter_tools(all_tools, TEXT_TOOL_NAMES)
    utility_tools = filter_tools(all_tools, UTILITY_TOOL_NAMES)

    mathematician = create_agent(
        model=specialist_llm,
        tools=math_tools,
        system_prompt="You are the Mathematician — a precise, methodical specialist. "
        "Show your work step by step. You have tools for arithmetic, "
        "exponentiation, square roots, percentages, random number generation, "
        "and UUID generation. "
        "IMPORTANT: Only answer the math parts of a question. "
        "Silently skip anything outside your specialty — "
        "never say 'I cannot' or mention missing capabilities.",
        name="Mathematician",
    )

    wordsmith = create_agent(
        model=specialist_llm,
        tools=text_tools,
        system_prompt="You are the Wordsmith — a text processing specialist. "
        "You handle word counts, character counts, case conversion, "
        "and string reversal. Be concise and accurate. "
        "IMPORTANT: Only answer the text-processing parts of a question. "
        "Silently skip anything outside your specialty — "
        "never say 'I cannot' or mention missing capabilities.",
        name="Wordsmith",
    )

    timekeeper = create_agent(
        model=specialist_llm,
        tools=utility_tools,
        system_prompt="You are the Timekeeper — a temporal specialist. "
        "You handle time queries and date calculations. Be precise with formats. "
        "IMPORTANT: Only answer the time/date parts of a question. "
        "Silently skip anything outside your specialty — "
        "never say 'I cannot' or mention missing capabilities.",
        name="Timekeeper",
    )

    return {
        "Mathematician": mathematician,
        "Wordsmith": wordsmith,
        "Timekeeper": timekeeper,
    }


# ─────────────────────────────────────────────
# 5. SUPERVISOR (AGENTS-AS-TOOLS)
#    The supervisor is itself a ReAct agent.
#    Its "tools" are wrapper functions that call
#    the sub-agents. When the LLM returns multiple
#    tool calls in one response, LangGraph runs
#    them concurrently — that's how we get
#    LLM-driven parallelism for free.
#
#    We use @tool wrappers instead of .as_tool()
#    because .as_tool() exposes the agent's
#    MessagesState schema, which confuses the
#    supervisor's LLM. Simple string-in/string-out
#    wrappers are cleaner and easier to understand.
# ─────────────────────────────────────────────


def create_supervisor(specialists: dict):
    """Build a supervisor agent that delegates to specialists.

    Each specialist is wrapped as a LangChain @tool so the supervisor
    can call it like any other tool. The supervisor's LLM decides
    which specialist(s) to invoke based on the user's request.
    """

    # We need to capture the agent references in closures.
    # Each wrapper: takes a string request → invokes the sub-agent → returns the answer.

    mathematician_agent = specialists["Mathematician"]
    wordsmith_agent = specialists["Wordsmith"]
    timekeeper_agent = specialists["Timekeeper"]

    @tool
    async def ask_mathematician(request: str) -> str:
        """Delegate a math/number question to the Mathematician specialist.
        Use for arithmetic, exponentiation, square roots, percentages, random numbers, and UUIDs.
        """
        result = await mathematician_agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        return result["messages"][-1].content

    @tool
    async def ask_wordsmith(request: str) -> str:
        """Delegate a text processing question to the Wordsmith specialist.
        Use for word counts, character counts, case conversion, and string reversal.
        """
        result = await wordsmith_agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        return result["messages"][-1].content

    @tool
    async def ask_timekeeper(request: str) -> str:
        """Delegate a time/date question to the Timekeeper specialist.
        Use for current time and date differences.
        """
        result = await timekeeper_agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        return result["messages"][-1].content

    supervisor = create_agent(
        model=supervisor_llm,
        tools=[ask_mathematician, ask_wordsmith, ask_timekeeper],
        system_prompt="You are a supervisor coordinating three specialist agents:\n"
        "  - Mathematician: math, calculations, random numbers, UUIDs\n"
        "  - Wordsmith: text processing (counts, case, reversal)\n"
        "  - Timekeeper: current time, date differences\n\n"
        "For each user request, delegate to the appropriate specialist(s).\n\n"
        "IMPORTANT — before calling specialists, check for dependencies:\n"
        "  - If one specialist's INPUT requires another specialist's OUTPUT, "
        "call them sequentially. Wait for the first result, then pass it "
        "to the second. Example: 'get the time and count its characters' "
        "means Timekeeper first, then Wordsmith with the time string.\n"
        "  - Only call specialists in parallel when their inputs are fully "
        "independent. Example: 'calculate 2^10 and reverse hello' — "
        "neither needs the other's result.\n\n"
        "Synthesize their answers into a single coherent response.",
        name="Supervisor",
    )

    return supervisor


# ─────────────────────────────────────────────
# 6. INTERACTIVE CLI
#    Chat loop with conversation memory. The
#    supervisor streams its responses, showing
#    delegation steps as they happen.
# ─────────────────────────────────────────────


async def cli(supervisor) -> None:
    """Interactive chat loop with conversation memory and delegation tracing."""
    console = Console()

    console.print(
        Panel(
            f"Supervisor: [bold]{SUPERVISOR_MODEL}[/bold]\n"
            f"Specialists: [bold]{SPECIALIST_MODEL}[/bold]\n"
            f"Pattern: [bold]supervisor[/bold] (agents-as-tools)\n"
            f"Specialists: Mathematician, Wordsmith, Timekeeper\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to leave.\n\n"
            "[bold]Parallel[/bold] (independent tasks, agents run concurrently):\n"
            "  \"Calculate 2^10 and count the words in 'hello world'\"  → Mathematician + Wordsmith\n"
            '  "What time is it and what is the square root of 144?"   → Timekeeper + Mathematician\n'
            "  \"Multiply 7 by 8, uppercase 'hello', and get the time\" → Mathematician + Wordsmith + Timekeeper\n\n"
            "[bold]Chained[/bold] (result from one agent feeds into another):\n"
            "  \"Count the words in 'Foo Bar Baz' and multiply that count by 3\" → Wordsmith then Mathematician\n"
            '  "Get the current time and count the characters in it"            → Timekeeper then Wordsmith',
            title="Supervisor Agent",
            border_style="blue",
        )
    )

    messages: list[dict] = []

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye!")
            break

        if user_input.strip().lower() in ("exit", "quit"):
            console.print("Goodbye!")
            break

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        last_msg = None
        async for chunk in supervisor.astream(
            {"messages": messages},  # type: ignore[arg-type]
            stream_mode="values",
        ):
            last_msg = chunk["messages"][-1]

            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    console.print(
                        f"  [dim italic]delegating to: {tc['name']}({tc['args']})[/dim italic]"
                    )
            elif last_msg.type == "tool":
                content = last_msg.content
                preview = content[:200] + "..." if len(content) > 200 else content
                console.print(f"  [dim italic]result: {preview}[/dim italic]")

        if last_msg:
            answer = last_msg.content or "(no response)"
            messages.append({"role": "assistant", "content": answer})
            console.print(
                Panel(
                    Markdown(answer),
                    title="Supervisor",
                    border_style="cyan",
                )
            )


async def main():
    console = Console()

    console.print("[dim]Loading tools from MCP server...[/dim]")
    client = MultiServerMCPClient(MCP_SERVERS)  # type: ignore[arg-type]
    all_tools = await client.get_tools()
    tool_names = [t.name for t in all_tools]
    console.print(f"[dim]Loaded {len(all_tools)} tools: {tool_names}[/dim]")

    specialists = create_specialists(all_tools)
    supervisor = create_supervisor(specialists)

    await cli(supervisor)


if __name__ == "__main__":
    asyncio.run(main())
