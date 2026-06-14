# This lesson is the client-side complement to file 39 — instead of calling the MCP session
# directly, you wrap list_prompts() and get_prompt() inside an MCPClient class so the rest
# of your application never touches protocol types: list_prompts() returns a plain Python
# list of Prompt objects, and get_prompt(name, args) returns the rendered message list that
# feeds straight into Claude without any transformation by the caller.
# The signature of get_prompt matters: arguments is a plain dict[str, str] matching the
# parameter names the server declared, and the SDK interpolates those values into the prompt
# template on the server side before returning the messages — the client never builds prompts.
# The practical pattern from the course is a slash-command CLI: when the user types "/" the
# app lists every available prompt as a command, when they type "/format_document" the app
# collects the required arguments interactively (e.g. which document to format), calls
# get_prompt to fetch the expert-written messages, then sends those messages to Claude with
# the server's tools attached so Claude can execute the task end-to-end autonomously.
# The separation is clean: MCPClient owns the protocol, the CLI owns the interaction,
# Claude owns the execution — each layer knows nothing about the others' internals.

# Full example: the server exposes three prompts and three document tools; MCPClient wraps
# all session calls and returns clean Python objects; the chat loop accepts normal messages
# AND slash commands — type "/" to list prompts, "/summarise_document" to run a specific one,
# or just chat normally and Claude answers without any tools.


import asyncio
import os
import sys
from typing import Any
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

load_dotenv()

API_KEY    = os.getenv("ANTHROPIC_API_KEY")
MODEL      = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = max(int(os.getenv("MAX_TOKENS", "1024")), 2048)


# ══════════════════════════════════════════════════════════════════════════════
#  MCP SERVER  —  python 40_mcp_prompts_client.py --server
# ══════════════════════════════════════════════════════════════════════════════

if "--server" in sys.argv:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.prompts.base import UserMessage
    from pydantic import Field

    mcp = FastMCP("PromptClientDemo", log_level="ERROR")

    docs: dict[str, str] = {
        "project_plan.md": (
            "project alpha timeline phase one q1 infrastructure setup "
            "four backend engineers phase two q2 beta rollout fifty customers "
            "phase three q3 general availability sla budget four eighty thousand"
        ),
        "budget.txt": (
            "total budget four hundred eighty thousand dollars "
            "engineering two hundred eighty thousand fifty eight percent "
            "marketing one twenty thousand twenty five percent "
            "operations eighty thousand seventeen percent q1 spend ninety five thousand"
        ),
        "team_notes.md": (
            "lead sarah chen backend alice bob carol dave frontend eve frank grace "
            "qa henry irene next demo friday three pm "
            "blocker vendor api rate limits ticket eight nine two"
        ),
    }

    @mcp.tool(name="list_documents", description="List all document IDs.")
    def list_documents() -> str:
        return "\n".join(docs.keys())

    @mcp.tool(name="read_doc_contents", description="Read full text of a document.")
    def read_document(doc_id: str = Field(description="Document ID, e.g. 'budget.txt'")) -> str:
        if doc_id not in docs:
            raise ValueError(f"'{doc_id}' not found.")
        return docs[doc_id]

    @mcp.tool(name="edit_document", description="Replace text inside a document.")
    def edit_document(
        doc_id:  str = Field(description="Document ID to edit."),
        old_str: str = Field(description="Exact text to find."),
        new_str: str = Field(description="Replacement text."),
    ) -> str:
        if doc_id not in docs:
            raise ValueError(f"'{doc_id}' not found.")
        if old_str not in docs[doc_id]:
            raise ValueError(f"Text not found in '{doc_id}'.")
        docs[doc_id] = docs[doc_id].replace(old_str, new_str)
        return f"'{doc_id}' updated."

    @mcp.prompt(name="format_document",
                description="Rewrites a document in clean Markdown with headers, bullets, tables.")
    def format_document(doc_id: str = Field(description="Document ID to reformat.")) -> list:
        return [UserMessage(
            f"Reformat the document '{doc_id}' into clean Markdown.\n"
            "1. Use read_doc_contents to get the current text.\n"
            "2. Rewrite it with # headings, **bold** key terms, bullet points, and tables for any data.\n"
            "3. Use edit_document to save the reformatted version.\n"
            "4. Confirm what was changed."
        )]

    @mcp.prompt(name="summarise_document",
                description="3-bullet executive summary — faster and more consistent than ad-hoc requests.")
    def summarise_document(doc_id: str = Field(description="Document ID to summarise.")) -> list:
        return [UserMessage(
            f"Summarise '{doc_id}' in exactly 3 bullet points.\n"
            "1. Use read_doc_contents to read it.\n"
            "2. Write EXACTLY 3 bullets: key fact, key numbers, main risk/next action.\n"
            "3. Each bullet must be one sentence under 20 words. No other formatting."
        )]

    @mcp.prompt(name="review_document",
                description="Reviews a document for issues and gives actionable improvement suggestions.")
    def review_document(
        doc_id: str = Field(description="Document ID to review."),
        focus:  str = Field(description="'clarity', 'completeness', or 'consistency'", default="completeness"),
    ) -> list:
        return [UserMessage(
            f"Review '{doc_id}' focusing on: {focus}\n"
            "1. Use read_doc_contents to read it.\n"
            "2. Output:\n"
            "   SCORE: X/10\n"
            "   ISSUES:\n   - issue 1\n   - issue 2\n"
            "   FIXES:\n   - specific fix 1\n   - specific fix 2"
        )]

    mcp.run()
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  MCPClient CLASS
#  Each method returns a plain Python object — no MCP types leak to callers.
# ══════════════════════════════════════════════════════════════════════════════

class MCPClient:
    def __init__(self, session: ClientSession):
        self._session = session

    # ── Prompts ───────────────────────────────────────────────────────────────

    async def list_prompts(self) -> list[types.Prompt]:
        """Return all prompts the server exposes."""
        result = await self._session.list_prompts()
        return result.prompts                          # list[types.Prompt]

    async def get_prompt(self, prompt_name: str, args: dict[str, str]) -> list:
        """
        Fetch a rendered prompt from the server.
        The server interpolates args into the prompt template and returns messages.
        Returns the messages list — feed it directly into Claude.
        """
        result = await self._session.get_prompt(prompt_name, args)
        return result.messages                         # list[PromptMessage]

    # ── Tools (needed so Claude can execute the prompt's instructions) ─────────

    async def list_tools(self) -> list[dict]:
        result = await self._session.list_tools()
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
            for t in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict) -> str:
        result = await self._session.call_tool(name, arguments)
        return "\n".join(getattr(c, "text", "") for c in result.content)


# ══════════════════════════════════════════════════════════════════════════════
#  SLASH-COMMAND CHAT
# ══════════════════════════════════════════════════════════════════════════════

def prompt_messages_to_claude(mcp_messages) -> list[dict]:
    """Convert MCP PromptMessage objects to Claude's dict format."""
    return [
        {"role": m.role, "content": getattr(m.content, "text", str(m.content))}
        for m in mcp_messages
    ]


async def run_slash_chat(client: MCPClient, claude: Anthropic):
    """
    Chat loop with slash-command support.
    /          → list all available prompts
    /name      → run that prompt (collects args interactively)
    anything   → normal Claude conversation
    """
    tools          = await client.list_tools()
    prompts        = await client.list_prompts()
    prompt_map     = {p.name: p for p in prompts}
    messages: list = []

    print("\nType a message, '/' to list prompts, or '/prompt_name' to run one.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You : ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        # ── Slash command ─────────────────────────────────────────────────────
        if user_input.startswith("/"):
            command = user_input[1:].strip()   # everything after "/"

            # Bare "/" → show all prompts
            if not command:
                print("\nAvailable prompts (type /name to use):")
                for p in prompts:
                    args_info = ""
                    if p.arguments:
                        req  = [a.name for a in p.arguments if a.required]
                        opt  = [a.name for a in p.arguments if not a.required]
                        args_info = f"  args: {req}" + (f"  optional: {opt}" if opt else "")
                    print(f"  /{p.name}")
                    print(f"      {p.description}")
                    if args_info:
                        print(f"      {args_info}")
                print()
                continue

            # "/name" → find the prompt
            if command not in prompt_map:
                print(f"  Unknown prompt '/{command}'. Type '/' to see available prompts.")
                continue

            prompt = prompt_map[command]

            # Collect required (and optional) arguments interactively
            args: dict[str, str] = {}
            print(f"\n  Running /{prompt.name}  —  {prompt.description}")
            for arg in (prompt.arguments or []):
                label = f"  {arg.name} [{arg.description}]"
                label += "" if arg.required else " (Enter to skip)"
                label += ": "
                val = input(label).strip()
                if val:
                    args[arg.name] = val
                elif arg.required:
                    print(f"  '{arg.name}' is required.")
                    break
            else:
                # Fetch the expert prompt from the MCP server
                print(f"\n  Fetching prompt '{prompt.name}' from server...")
                mcp_msgs        = await client.get_prompt(prompt.name, args)
                claude_messages = prompt_messages_to_claude(mcp_msgs)

                print(f"  Got {len(claude_messages)} message(s). Sending to Claude with {len(tools)} tools...\n")
                print("=" * 60)

                # Multi-turn loop — Claude may call tools multiple times
                while True:
                    response = claude.messages.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        tools=tools,
                        messages=claude_messages,
                    )

                    for block in response.content:
                        if block.type == "text" and block.text.strip():
                            print(block.text)
                        elif block.type == "tool_use":
                            print(f"\n  [tool] {block.name}({block.input})")

                    if response.stop_reason == "end_turn":
                        break

                    if response.stop_reason == "tool_use":
                        claude_messages.append({"role": "assistant", "content": response.content})
                        results = []
                        for block in response.content:
                            if block.type != "tool_use":
                                continue
                            out = await client.call_tool(block.name, block.input)
                            print(f"  [result] {out[:100]}{'...' if len(out) > 100 else ''}")
                            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
                        claude_messages.append({"role": "user", "content": results})

                print("=" * 60 + "\n")
            continue

        # ── Normal chat message ───────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        response = claude.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
        )

        reply = response.content[-1].text
        messages.append({"role": "assistant", "content": reply})
        print(f"\nClaude: {reply}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def run():
    claude        = Anthropic(api_key=API_KEY)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[__file__, "--server"],
    )

    print("=== MCP Prompts Client — Slash Command Chat ===")
    print(f"Model: {MODEL}\n")
    print("Connecting to MCP server...")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            client = MCPClient(session)

            prompts = await client.list_prompts()
            tools   = await client.list_tools()
            print(f"Connected. {len(prompts)} prompt(s), {len(tools)} tool(s) available.")

            await run_slash_chat(client, claude)


if __name__ == "__main__":
    asyncio.run(run())
