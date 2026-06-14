# MCP Prompts are pre-written, expert-tested instruction templates stored on the server
# that clients can request and use instead of inventing prompts from scratch — the
# difference between a user typing "convert this to markdown" and the server providing
# a carefully worded, fully tested prompt that tells Claude exactly how to structure
# headers, tables, and bullet points for consistent, high-quality output every time.
# On the server side you define them with @mcp.prompt(name, description) and return a
# list of UserMessage / AssistantMessage objects; arguments work exactly like tool
# parameters using Field() and the SDK interpolates them into the prompt text at request time.
# On the client side, list_prompts() discovers what prompts the server exposes (with their
# required arguments), and get_prompt(name, arguments={...}) fetches the fully rendered
# message list — you then pass those messages straight to Claude exactly as they are,
# optionally alongside the server's tools so Claude can act on the prompt immediately.
# The key value: prompt authors encode their domain expertise once on the server; every
# client that connects gets that expertise for free without writing a single prompt word.

# Full example: the server exposes three prompts — format_document (rewrites in Markdown),
# summarise_document (tight executive summary), and review_document (finds issues and
# suggests fixes) — plus three tools so Claude can actually read and edit documents.
# Mode 1 lists all prompts and their arguments; mode 2 previews the raw messages a prompt
# generates before sending anything to Claude; mode 3 executes a chosen prompt end-to-end:
# fetches the prompt messages, sends them to Claude with the tools attached, and Claude
# reads the document and edits or summarises it in one uninterrupted autonomous run.


import asyncio
import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

API_KEY    = os.getenv("ANTHROPIC_API_KEY")
MODEL      = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = max(int(os.getenv("MAX_TOKENS", "1024")), 2048)


# ══════════════════════════════════════════════════════════════════════════════
#  MCP SERVER  —  python 39_mcp_prompts.py --server
# ══════════════════════════════════════════════════════════════════════════════

if "--server" in sys.argv:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.prompts.base import UserMessage, AssistantMessage
    from pydantic import Field

    mcp = FastMCP("PromptsDemo", log_level="ERROR")

    # In-memory document store
    docs: dict[str, str] = {
        "project_plan.md": (
            "project alpha timeline phase one q1 infrastructure setup backend engineers four people "
            "phase two q2 beta rollout fifty enterprise customers phase three q3 general availability "
            "sla ninety nine point nine percent ten thousand users at launch budget four hundred eighty thousand"
        ),
        "budget.txt": (
            "total budget four hundred eighty thousand dollars "
            "engineering two hundred eighty thousand dollars fifty eight percent "
            "marketing one hundred twenty thousand dollars twenty five percent "
            "operations eighty thousand dollars seventeen percent "
            "q1 actual spend ninety five thousand q2 forecast one hundred fifteen thousand"
        ),
        "team_notes.md": (
            "engineering lead sarah chen backend alice bob carol dave frontend eve frank grace "
            "qa henry irene sprint velocity forty two points next demo friday three pm "
            "blockers vendor api rate limits ticket eight nine two staging environment flaky ticket nine zero one"
        ),
    }

    # ── Tools (Claude will use these when executing prompts) ──────────────────

    @mcp.tool(name="list_documents", description="List all document IDs.")
    def list_documents() -> str:
        return "\n".join(docs.keys())

    @mcp.tool(name="read_doc_contents", description="Read the full text of a document.")
    def read_document(
        doc_id: str = Field(description="Document ID to read, e.g. 'budget.txt'"),
    ) -> str:
        if doc_id not in docs:
            raise ValueError(f"Document '{doc_id}' not found.")
        return docs[doc_id]

    @mcp.tool(name="edit_document", description="Replace text inside a document.")
    def edit_document(
        doc_id:  str = Field(description="Document ID to edit."),
        old_str: str = Field(description="Exact text to find."),
        new_str: str = Field(description="Replacement text."),
    ) -> str:
        if doc_id not in docs:
            raise ValueError(f"Document '{doc_id}' not found.")
        if old_str not in docs[doc_id]:
            raise ValueError(f"Text not found in '{doc_id}'.")
        docs[doc_id] = docs[doc_id].replace(old_str, new_str)
        return f"'{doc_id}' updated successfully."

    # ── Prompts — the pre-written expert templates ────────────────────────────

    @mcp.prompt(
        name="format_document",
        description="Rewrites a document using proper Markdown formatting with headers, "
                    "bullet points, and tables. Much better than asking Claude yourself.",
    )
    def format_document(
        doc_id: str = Field(description="ID of the document to reformat."),
    ) -> list:
        prompt = f"""Your task is to reformat a document into clean, professional Markdown.

The document ID you need to reformat is: {doc_id}

Steps:
1. Use the read_doc_contents tool to read the current document content.
2. Rewrite it in Markdown with proper structure:
   - Use # and ## for headings
   - Use bullet points or numbered lists where appropriate
   - Use **bold** for key terms and figures
   - Add a table if there is tabular data (like budget breakdowns)
3. Use the edit_document tool to save the entire reformatted content back.
4. Confirm what changes you made.

Be thorough — add structure even where the original has none."""
        return [UserMessage(prompt)]

    @mcp.prompt(
        name="summarise_document",
        description="Produces a tight 3-bullet executive summary of any document. "
                    "Faster and more consistent than ad-hoc summarisation requests.",
    )
    def summarise_document(
        doc_id: str = Field(description="ID of the document to summarise."),
    ) -> list:
        prompt = f"""Produce a concise executive summary of the document: {doc_id}

1. Use read_doc_contents to retrieve the document.
2. Write EXACTLY 3 bullet points:
   - Bullet 1: The single most important fact or decision
   - Bullet 2: Key numbers, dates, or names
   - Bullet 3: The main risk or next action
3. Each bullet must be one sentence, under 20 words.
4. Do not use any other formatting — just the 3 bullets."""
        return [UserMessage(prompt)]

    @mcp.prompt(
        name="review_document",
        description="Reviews a document for clarity issues, missing information, and "
                    "inconsistencies. Provides specific, actionable improvement suggestions.",
    )
    def review_document(
        doc_id:   str = Field(description="Document ID to review."),
        focus:    str = Field(description="Review focus: 'clarity', 'completeness', or 'consistency'",
                              default="completeness"),
    ) -> list:
        prompt = f"""You are a professional document reviewer. Review the document '{doc_id}' with focus on: {focus}

1. Read the document using read_doc_contents.
2. Evaluate it against the focus area:
   - clarity: Is the language clear? Are there ambiguous statements?
   - completeness: What important information is missing?
   - consistency: Are there contradictions or mismatched numbers?
3. Output your review in this exact format:

SCORE: X/10
ISSUES FOUND:
- [issue 1]
- [issue 2]
RECOMMENDATIONS:
- [specific fix 1]
- [specific fix 2]"""
        return [UserMessage(prompt)]

    mcp.run()
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  CLIENT + DEMO
# ══════════════════════════════════════════════════════════════════════════════

def prompt_messages_to_claude(prompt_result) -> list[dict]:
    """Convert MCP PromptMessage list to Claude messages format."""
    messages = []
    for msg in prompt_result.messages:
        text = getattr(msg.content, "text", str(msg.content))
        messages.append({"role": msg.role, "content": text})
    return messages


def mcp_tool_to_claude(tool) -> dict:
    return {
        "name":         tool.name,
        "description":  tool.description or "",
        "input_schema": tool.inputSchema,
    }


async def run():
    claude        = Anthropic(api_key=API_KEY)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[__file__, "--server"],
    )

    print("=== MCP Prompts Demo ===")
    print(f"Model: {MODEL}\n")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            print("Choose a mode:")
            print("  [1] List prompts   -- see all prompts and their arguments")
            print("  [2] Preview prompt -- see raw messages a prompt generates (no Claude call)")
            print("  [3] Execute prompt -- run a prompt end-to-end: get messages -> send to Claude")
            print()
            mode = input("Mode (1/2/3, default 3): ").strip() or "3"

            # ── Mode 1: List all prompts ──────────────────────────────────────
            if mode == "1":
                result = await session.list_prompts()
                print(f"\n{len(result.prompts)} prompt(s) on this server:\n")
                for p in result.prompts:
                    print(f"  Name        : {p.name}")
                    print(f"  Description : {p.description}")
                    if p.arguments:
                        print("  Arguments   :")
                        for arg in p.arguments:
                            req = "required" if arg.required else "optional"
                            print(f"    - {arg.name} ({req}): {arg.description}")
                    print()

            # ── Mode 2: Preview messages without calling Claude ───────────────
            elif mode == "2":
                result = await session.list_prompts()
                print("\nAvailable prompts:")
                for i, p in enumerate(result.prompts, 1):
                    print(f"  [{i}] {p.name}")

                choice = input("\nSelect prompt number: ").strip()
                try:
                    prompt = result.prompts[int(choice) - 1]
                except (ValueError, IndexError):
                    print("Invalid choice.")
                    return

                # Collect arguments interactively
                args = {}
                for arg in (prompt.arguments or []):
                    val = input(f"  {arg.name} [{arg.description}]: ").strip()
                    if val:
                        args[arg.name] = val
                    elif arg.required:
                        print(f"  '{arg.name}' is required.")
                        return

                prompt_result = await session.get_prompt(prompt.name, arguments=args)

                print(f"\n--- Messages generated by '{prompt.name}' ---")
                for i, msg in enumerate(prompt_result.messages, 1):
                    text = getattr(msg.content, "text", str(msg.content))
                    print(f"\n  Message {i}  role={msg.role}")
                    print("  " + "-" * 50)
                    print(text)

            # ── Mode 3: Execute a prompt end-to-end with Claude ───────────────
            elif mode == "3":
                prompts_result = await session.list_prompts()
                tools_result   = await session.list_tools()
                claude_tools   = [mcp_tool_to_claude(t) for t in tools_result.tools]

                print("\nDocuments available: project_plan.md, budget.txt, team_notes.md")
                print("\nAvailable prompts:")
                for i, p in enumerate(prompts_result.prompts, 1):
                    print(f"  [{i}] {p.name}  —  {p.description[:60]}...")
                print()

                choice = input("Select prompt number: ").strip()
                try:
                    prompt = prompts_result.prompts[int(choice) - 1]
                except (ValueError, IndexError):
                    print("Invalid choice.")
                    return

                # Collect required arguments
                args = {}
                for arg in (prompt.arguments or []):
                    default = "" if arg.required else " (Enter to skip)"
                    val = input(f"  {arg.name}{default}: ").strip()
                    if val:
                        args[arg.name] = val
                    elif not arg.required:
                        continue
                    else:
                        print(f"'{arg.name}' is required.")
                        return

                # Fetch the expert-crafted prompt messages from the MCP server
                prompt_result   = await session.get_prompt(prompt.name, arguments=args)
                claude_messages = prompt_messages_to_claude(prompt_result)

                print(f"\nRunning prompt '{prompt.name}' with Claude...\n")
                print("=" * 60)

                # Send the prompt messages to Claude with tools attached —
                # Claude will use read_doc_contents and edit_document as instructed
                while True:
                    response = claude.messages.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        tools=claude_tools,
                        messages=claude_messages,
                    )

                    # Collect and display all blocks
                    for block in response.content:
                        if block.type == "text" and block.text.strip():
                            print(block.text)
                        elif block.type == "tool_use":
                            print(f"\n  [tool] {block.name}({block.input})")

                    if response.stop_reason == "end_turn":
                        break

                    # Handle tool calls — route through the MCP session
                    if response.stop_reason == "tool_use":
                        claude_messages.append({"role": "assistant", "content": response.content})

                        tool_results = []
                        for block in response.content:
                            if block.type != "tool_use":
                                continue
                            mcp_result  = await session.call_tool(block.name, block.input)
                            result_text = "\n".join(
                                getattr(c, "text", "") for c in mcp_result.content
                            )
                            print(f"  [result] {result_text[:100]}{'...' if len(result_text)>100 else ''}")
                            tool_results.append({
                                "type":        "tool_result",
                                "tool_use_id": block.id,
                                "content":     result_text,
                            })
                        claude_messages.append({"role": "user", "content": tool_results})

                print("=" * 60)

            else:
                print("Invalid mode.")


if __name__ == "__main__":
    asyncio.run(run())
