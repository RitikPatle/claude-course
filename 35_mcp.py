# MCP (Model Context Protocol) solves the problem of tool integration: instead of you writing
# dozens of tool schemas and implementation functions for every external service, you connect
# to an MCP server where someone else has already done that work — your code just asks
# "what tools do you have?" (ListToolsRequest) and "run this tool" (CallToolRequest).
# Communication is transport-agnostic, but the most common setup is stdio — your client
# starts an MCP server as a subprocess and talks to it over stdin/stdout, exactly like this
# file does when it launches itself with the --server flag.
# The Python MCP SDK makes building an MCP server trivial: one line creates the server
# (FastMCP("name")), then a @mcp.tool decorator on any function auto-generates the full
# JSON schema Claude needs from Python type hints and Pydantic Field descriptions — no
# manual schema writing, no boilerplate, just regular Python functions.
# In production you typically implement only ONE side — either a client that connects to
# existing MCP servers (GitHub, AWS, databases), or a server that exposes your own service;
# this file implements both in one script purely so you can see how the two halves talk.

# Full example: this single file acts as two programs depending on how it is invoked.
# Run without arguments and it is the MCP client + Claude chat: it spawns itself as a
# subprocess in --server mode, calls list_tools to discover three document tools, converts
# those to Claude's tool format, and runs an interactive chat where every tool call Claude
# makes is routed through the MCP client to the MCP server and the result fed back to Claude.
# Run with --server and it is the FastMCP server: it exposes three tools over stdio for
# listing, reading, and editing an in-memory document store with no database required.


import asyncio
import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

API_KEY    = os.getenv("ANTHROPIC_API_KEY")
MODEL      = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = max(int(os.getenv("MAX_TOKENS", "1024")), 1024)

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")


# ══════════════════════════════════════════════════════════════════════════════
#  MCP SERVER HALF
#  When this file is launched with --server it runs as a FastMCP server over
#  stdio, exposing three tools: list_documents, read_doc_contents, edit_document
# ══════════════════════════════════════════════════════════════════════════════

if "--server" in sys.argv:
    from mcp.server.fastmcp import FastMCP
    from pydantic import Field

    mcp = FastMCP("DocumentMCP", log_level="ERROR")

    # In-memory document store — no database needed
    docs: dict[str, str] = {
        "project_plan.md":  "The plan outlines a 3-phase rollout: Alpha (Q1), Beta (Q2), GA (Q3). "
                            "Phase Alpha focuses on core infrastructure and internal testing. "
                            "Phase Beta expands to selected enterprise customers. "
                            "Phase GA marks full public availability with SLA guarantees.",
        "budget.txt":       "Total project budget: $480,000. "
                            "Engineering: $280,000 (58%). Marketing: $120,000 (25%). "
                            "Operations: $80,000 (17%). Current spend as of Q1: $95,000.",
        "team_notes.md":    "Engineering lead: Sarah Chen. Backend team: 4 engineers. "
                            "Frontend team: 3 engineers. QA: 2 testers. "
                            "Next sync: Monday 10am. Blockers: API rate limits from vendor.",
        "release_notes.txt":"v1.2.0 — Fixed login timeout bug. Improved dashboard load time by 40%%. "
                            "v1.1.0 — Added CSV export feature. Fixed mobile layout issues. "
                            "v1.0.0 — Initial release. Core features complete.",
    }

    # ── Tool 1: list all document IDs ────────────────────────────────────────
    @mcp.tool(
        name="list_documents",
        description="List all available document IDs in the document store.",
    )
    def list_documents() -> str:
        return "\n".join(docs.keys())

    # ── Tool 2: read a document's content ────────────────────────────────────
    @mcp.tool(
        name="read_doc_contents",
        description="Read the full contents of a document and return it as a string.",
    )
    def read_document(
        doc_id: str = Field(description="The document ID to read (e.g. 'budget.txt')"),
    ) -> str:
        if doc_id not in docs:
            raise ValueError(f"Document '{doc_id}' not found. Use list_documents to see available IDs.")
        return docs[doc_id]

    # ── Tool 3: edit a document with find-and-replace ─────────────────────────
    @mcp.tool(
        name="edit_document",
        description="Edit a document by replacing a specific string with new text.",
    )
    def edit_document(
        doc_id:  str = Field(description="The document ID to edit."),
        old_str: str = Field(description="The exact text to find and replace. Must match exactly including whitespace."),
        new_str: str = Field(description="The replacement text to insert in place of old_str."),
    ) -> str:
        if doc_id not in docs:
            raise ValueError(f"Document '{doc_id}' not found.")
        if old_str not in docs[doc_id]:
            raise ValueError(f"Text '{old_str[:40]}...' not found in '{doc_id}'.")
        docs[doc_id] = docs[doc_id].replace(old_str, new_str)
        return f"Document '{doc_id}' updated successfully."

    # Start the MCP server — it blocks here, communicating via stdio
    mcp.run()
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  MCP CLIENT + CLAUDE CHAT HALF
#  When this file is run normally, it starts itself as a subprocess in --server
#  mode, connects via stdio, lists tools, then runs an interactive Claude chat.
# ══════════════════════════════════════════════════════════════════════════════

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def mcp_tool_to_claude_format(tool) -> dict:
    """Convert an MCP tool definition to the dict format Claude expects."""
    return {
        "name":         tool.name,
        "description":  tool.description or "",
        "input_schema": tool.inputSchema,   # MCP already returns a valid JSON schema dict
    }


async def run_mcp_chat():
    claude = Anthropic(api_key=API_KEY)

    # ── Launch this same file as the MCP server subprocess ───────────────────
    server_params = StdioServerParameters(
        command=sys.executable,          # same Python interpreter
        args=[__file__, "--server"],     # but with --server flag → runs the server half
    )

    print("=== MCP Document Chat ===")
    print(f"Model: {MODEL}")
    print("\nConnecting to MCP server...")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # ── Step 1: Initialize the MCP connection ─────────────────────────
            await session.initialize()

            # ── Step 2: ListTools — ask the MCP server what tools it provides ─
            tools_result = await session.list_tools()
            claude_tools = [mcp_tool_to_claude_format(t) for t in tools_result.tools]

            print(f"MCP server connected. Tools available ({len(claude_tools)}):")
            for t in claude_tools:
                print(f"  - {t['name']}: {t['description']}")

            print("\nAsk anything about the documents. Type 'exit' to quit.")
            print("Hint: try 'what documents do you have?' or 'summarise the budget'\n")

            messages: list = []

            while True:
                user_input = input("You : ").strip()
                if user_input.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break
                if not user_input:
                    continue

                messages.append({"role": "user", "content": user_input})

                # ── Multi-turn loop until Claude stops calling tools ──────────
                while True:
                    response = claude.messages.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        tools=claude_tools,
                        messages=messages,
                    )

                    # ── Final answer — no more tool calls ────────────────────
                    if response.stop_reason == "end_turn":
                        text_blocks = [b for b in response.content if b.type == "text"]
                        reply = text_blocks[-1].text if text_blocks else "(no text response)"
                        messages.append({"role": "assistant", "content": response.content})
                        print(f"\nClaude: {reply}\n")
                        break

                    # ── Tool use — Claude wants to call an MCP tool ──────────
                    elif response.stop_reason == "tool_use":
                        messages.append({"role": "assistant", "content": response.content})

                        tool_results = []
                        for block in response.content:
                            if block.type != "tool_use":
                                continue

                            print(f"  [MCP] Calling tool '{block.name}' with {block.input}")

                            # ── Step 3: CallTool — ask MCP server to execute it
                            mcp_result = await session.call_tool(block.name, block.input)

                            # Extract text from MCP result content
                            result_text = "\n".join(
                                c.text for c in mcp_result.content if hasattr(c, "text")
                            )

                            print(f"  [MCP] Result: {result_text[:120]}{'...' if len(result_text) > 120 else ''}")

                            tool_results.append({
                                "type":        "tool_result",
                                "tool_use_id": block.id,
                                "content":     result_text,
                            })

                        # Feed all tool results back to Claude in one message
                        messages.append({"role": "user", "content": tool_results})

                    else:
                        # Unexpected stop reason — just print what we have and move on
                        print(f"  (stop_reason={response.stop_reason})")
                        break


if __name__ == "__main__":
    asyncio.run(run_mcp_chat())
