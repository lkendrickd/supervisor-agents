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
  - Scribe        — file tools (read_file, list_files, create_file, delete_file)

HUMAN-IN-THE-LOOP:
  File write operations (create_file, delete_file) require human approval.
  The supervisor uses HumanInTheLoopMiddleware to interrupt before executing
  write tools. The CLI prompts the user to approve, reject, or edit the
  operation before resuming.

Run:
  uv run python supervisor.py
"""

import os
import asyncio
import uuid as uuid_mod
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

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv(override=True)


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
MATH_TOOL_NAMES: set[str] = {
    "add",
    "multiply",
    "sqrt",
    "divide",
    "power",
    "percentage",
    "random_number",
    "generate_uuid",
}
TEXT_TOOL_NAMES: set[str] = {
    "word_count",
    "char_count",
    "to_uppercase",
    "to_lowercase",
    "reverse_text",
}
UTILITY_TOOL_NAMES: set[str] = {"now", "date_diff"}
FILE_READ_TOOL_NAMES: set[str] = {"read_file", "list_files"}
FILE_WRITE_TOOL_NAMES: set[str] = {"create_file", "delete_file"}


def filter_tools(all_tools: list, names: set[str]) -> list:
    """Pick tools whose name is in the given set."""
    return [t for t in all_tools if t.name in names]


# ─────────────────────────────────────────────
# 4. CREATE SPECIALIST AGENTS
#    Each agent gets a filtered subset of MCP
#    tools, a system prompt defining its persona,
#    and a name for identification in logs.
# ─────────────────────────────────────────────


def create_specialists(all_tools: list, llm: ChatOpenAI) -> dict:
    """Build the specialist agents from the full MCP tool list.

    Returns a dict mapping persona name → compiled agent.
    """
    math_tools: list = filter_tools(all_tools, MATH_TOOL_NAMES)
    text_tools: list = filter_tools(all_tools, TEXT_TOOL_NAMES)
    utility_tools: list = filter_tools(all_tools, UTILITY_TOOL_NAMES)
    file_tools: list = filter_tools(
        all_tools, FILE_READ_TOOL_NAMES | FILE_WRITE_TOOL_NAMES
    )

    mathematician = create_agent(
        model=llm,
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
        model=llm,
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
        model=llm,
        tools=utility_tools,
        system_prompt="You are the Timekeeper — a temporal specialist. "
        "You handle time queries and date calculations. Be precise with formats. "
        "IMPORTANT: Only answer the time/date parts of a question. "
        "Silently skip anything outside your specialty — "
        "never say 'I cannot' or mention missing capabilities.",
        name="Timekeeper",
    )

    scribe = create_agent(
        model=llm,
        tools=file_tools,
        system_prompt="You are the Scribe — a file operations specialist. "
        "You handle reading files, listing directory contents, creating files, "
        "and deleting files. All paths are relative to the project root. "
        "CRITICAL: When a tool returns file contents, return the COMPLETE, "
        "UNMODIFIED output exactly as received. Never summarize, truncate, "
        "paraphrase, or interpret file contents. Never claim a file is empty "
        "if the tool returned content. Your job is to relay the raw tool "
        "output faithfully. "
        "IMPORTANT: Only answer the file-related parts of a question. "
        "Silently skip anything outside your specialty — "
        "never say 'I cannot' or mention missing capabilities.",
        name="Scribe",
    )

    return {
        "Mathematician": mathematician,
        "Wordsmith": wordsmith,
        "Timekeeper": timekeeper,
        "Scribe": scribe,
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


def create_supervisor(specialists: dict, llm: ChatOpenAI, checkpointer=None):
    """Build a supervisor agent that delegates to specialists.

    Each specialist is wrapped as a LangChain @tool so the supervisor
    can call it like any other tool. The supervisor's LLM decides
    which specialist(s) to invoke based on the user's request.

    If *checkpointer* is provided, enables human-in-the-loop approval
    for file-write operations via HumanInTheLoopMiddleware.
    """

    # We need to capture the agent references in closures.
    # Each wrapper: takes a string request → invokes the sub-agent → returns the answer.

    mathematician_agent = specialists["Mathematician"]
    wordsmith_agent = specialists["Wordsmith"]
    timekeeper_agent = specialists["Timekeeper"]
    scribe_agent = specialists["Scribe"]

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

    @tool
    async def ask_scribe_read(request: str) -> str:
        """Delegate a file reading question to the Scribe specialist.
        Use for reading file contents and listing directory entries.
        Never use for creating or deleting files.
        """
        result = await scribe_agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        return result["messages"][-1].content

    @tool
    async def ask_scribe_write(request: str) -> str:
        """Delegate a file write or delete operation to the Scribe specialist.
        Use for creating new files or deleting existing files.
        Requires human approval.
        """
        result = await scribe_agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        return result["messages"][-1].content

    hitl = HumanInTheLoopMiddleware(interrupt_on={"ask_scribe_write": True})

    supervisor = create_agent(
        model=llm,
        tools=[
            ask_mathematician,
            ask_wordsmith,
            ask_timekeeper,
            ask_scribe_read,
            ask_scribe_write,
        ],
        middleware=[hitl],
        checkpointer=checkpointer,
        system_prompt="You are a supervisor coordinating four specialist agents:\n"
        "  - Mathematician: math, calculations, random numbers, UUIDs\n"
        "  - Wordsmith: text processing (counts, case, reversal)\n"
        "  - Timekeeper: current time, date differences\n"
        "  - Scribe (read): reading files, listing directories\n"
        "  - Scribe (write): creating/deleting files\n\n"
        "For each user request, delegate to the appropriate specialist(s).\n"
        "Use ask_scribe_read for reading/listing files.\n"
        "Use ask_scribe_write for creating/deleting files.\n\n"
        "NEVER ask the user for confirmation before calling a tool. "
        "Just call the tool directly — the system handles approval "
        "automatically for write operations. Do not say things like "
        "'please approve' or 'shall I proceed'. Act immediately.\n\n"
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
#    Uses a checkpointer for history and supports
#    human-in-the-loop interrupts for file writes.
# ─────────────────────────────────────────────


async def handle_interrupt(interrupts, console) -> Command:
    """Prompt the user to approve/reject each interrupted tool call.

    Returns a ``Command(resume=...)`` with an ``HITLResponse`` payload.
    """
    import json
    from langchain.agents.middleware.human_in_the_loop import (
        ApproveDecision,
        EditDecision,
        RejectDecision,
        HITLResponse,
        Decision,
    )

    decisions: list[Decision] = []

    for intr in interrupts:
        hitl_request = intr.value
        for action in hitl_request["action_requests"]:
            console.print(
                Panel(
                    f"[bold]Tool:[/bold]  {action['name']}\n"
                    f"[bold]Args:[/bold]  {action['args']}",
                    title="Approval Required",
                    border_style="yellow",
                )
            )

            while True:
                choice = (
                    console.input(
                        "[bold yellow]Approve, reject, or edit? (a/r/e):[/bold yellow] "
                    )
                    .strip()
                    .lower()
                )

                if choice in ("a", "approve"):
                    decisions.append(ApproveDecision(type="approve"))
                    console.print("  [green]Approved[/green]")
                    break
                elif choice in ("r", "reject"):
                    reason = console.input("[dim]Reason (optional):[/dim] ").strip()
                    decision = RejectDecision(type="reject")
                    if reason:
                        decision["message"] = reason
                    decisions.append(decision)
                    console.print("  [red]Rejected[/red]")
                    break
                elif choice in ("e", "edit"):
                    new_args_str = console.input(
                        "[dim]New args as JSON (e.g. "
                        '{"path": "x.txt", "content": "hi"}):[/dim] '
                    ).strip()
                    try:
                        new_args = json.loads(new_args_str)
                    except json.JSONDecodeError:
                        console.print("  [red]Invalid JSON, try again[/red]")
                        continue
                    decisions.append(
                        EditDecision(
                            type="edit",
                            edited_action={
                                "name": action["name"],
                                "args": new_args,
                            },
                        )
                    )
                    console.print("  [blue]Edited[/blue]")
                    break
                else:
                    console.print("  [dim]Please enter a, r, or e[/dim]")

    response: HITLResponse = {"decisions": decisions}
    return Command(resume=response)


async def _stream_and_trace(supervisor, input_val, config, console):
    """Stream the supervisor, printing delegation/result traces.

    Returns the last message from the stream (or None).
    """
    last_msg = None
    async for chunk in supervisor.astream(
        input_val,
        config,
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

    return last_msg


async def cli(supervisor, supervisor_model: str, specialist_model: str) -> None:
    """Interactive chat loop with conversation memory and delegation tracing."""
    console = Console()

    console.print(
        Panel(
            f"Supervisor: [bold]{supervisor_model}[/bold]\n"
            f"Specialists: [bold]{specialist_model}[/bold]\n"
            f"Pattern: [bold]supervisor[/bold] (agents-as-tools)\n"
            f"Specialists: Mathematician, Wordsmith, Timekeeper, Scribe\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to leave.\n\n"
            "[bold]Parallel[/bold] (independent tasks, agents run concurrently):\n"
            "  \"Calculate 2^10 and count the words in 'hello world'\"  → Mathematician + Wordsmith\n"
            '  "What time is it and what is the square root of 144?"   → Timekeeper + Mathematician\n'
            "  \"Multiply 7 by 8, uppercase 'hello', and get the time\" → Mathematician + Wordsmith + Timekeeper\n\n"
            "[bold]Chained[/bold] (result from one agent feeds into another):\n"
            "  \"Count the words in 'Foo Bar Baz' and multiply that count by 3\" → Wordsmith then Mathematician\n"
            '  "Get the current time and count the characters in it"            → Timekeeper then Wordsmith\n\n'
            "[bold]File read[/bold] (auto-approved):\n"
            '  "List the files in the project directory"                        → Scribe\n'
            '  "Read the contents of README.md"                                 → Scribe\n\n'
            "[bold]File write[/bold] (requires approval):\n"
            "  \"Create a file called notes.txt with 'hello world'\"             → Scribe (approval required)\n"
            '  "Delete the file notes.txt"                                      → Scribe (approval required)\n\n'
            "[bold]Cross-specialist[/bold]:\n"
            '  "Count the words in README.md"                                   → Scribe then Wordsmith\n'
            '  "Create a file with today\'s date as the filename"               → Timekeeper then Scribe (approval required)',
            title="Supervisor Agent",
            border_style="blue",
        )
    )

    thread_id = str(uuid_mod.uuid4())
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

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

        # First invocation with user message
        last_msg = await _stream_and_trace(
            supervisor,
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            console,
        )

        # Check for interrupts and resume until none remain
        while True:
            state = await supervisor.aget_state(config)
            if not state.interrupts:
                break

            command = await handle_interrupt(state.interrupts, console)
            last_msg = await _stream_and_trace(supervisor, command, config, console)

        if last_msg:
            answer = last_msg.content or "(no response)"
            console.print(
                Panel(
                    Markdown(answer),
                    title="Supervisor",
                    border_style="cyan",
                )
            )


async def main():
    console = Console()

    # ── Model config ──────────────────────────────────
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in .env (see .env.example)")

    supervisor_model = os.environ.get("SUPERVISOR_MODEL", "google/gemini-2.5-flash")
    specialist_model = os.environ.get("SPECIALIST_MODEL", supervisor_model)

    openrouter_kwargs: dict[str, Any] = {
        "max_completion_tokens": 2048,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": SecretStr(api_key),
    }

    supervisor_llm = ChatOpenAI(model=supervisor_model, **openrouter_kwargs)
    specialist_llm = ChatOpenAI(model=specialist_model, **openrouter_kwargs)

    # ── Load tools & build agents ─────────────────────
    console.print("[dim]Loading tools from MCP server...[/dim]")
    client = MultiServerMCPClient(MCP_SERVERS)  # type: ignore[arg-type]
    all_tools = await client.get_tools()
    tool_names = [t.name for t in all_tools]
    console.print(f"[dim]Loaded {len(all_tools)} tools: {tool_names}[/dim]")

    checkpointer = MemorySaver()
    specialists = create_specialists(all_tools, specialist_llm)
    supervisor = create_supervisor(specialists, supervisor_llm, checkpointer)

    await cli(supervisor, supervisor_model, specialist_model)


if __name__ == "__main__":
    asyncio.run(main())
