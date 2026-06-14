# MCP Resources expose data from your server — think of them as GET endpoints in HTTP.
# A Tool performs an action (edit, delete, compute); a Resource fetches information
# (list documents, read a file, query a database row) — the key rule is: if it changes
# state, it is a Tool; if it only returns data, it is a Resource.
# There are two flavours: Direct Resources have a fixed URI like "docs://documents" and
# always return the same kind of thing; Templated Resources have a URI pattern with
# curly-brace placeholders like "docs://documents/{doc_id}" where the SDK automatically
# parses the placeholder value out of the URI and passes it as a function argument.
# On the client side you discover resources with list_resources() and list_resource_templates(),
# then fetch data with read_resource(uri) — the result is always a list of content items
# whose .text field holds the response, already serialised by the SDK (lists become JSON,
# strings stay plain text, etc.) so you never manually call json.dumps().
# A practical use case from the course: typing "@filename" in a chat interface triggers
# list_resources to power autocomplete, then read_resource to inject that document's full
# text into the Claude prompt automatically — no explicit tool call by Claude needed.

# Full example: this file is both an MCP server (run with --server) and a demo client.
# The server exposes three resources — a direct resource listing all document IDs, a stats
# resource showing word counts, and a templated resource reading any document by ID — plus
# one edit tool to illustrate the resource-vs-tool distinction.  The client mode connects,
# prints every discovered resource and template, reads them with concrete URIs, then runs
# an @mention chat where typing "@budget.txt What is the engineering allocation?" injects
# the full document into Claude's context before sending, without Claude needing to call any tool.


import asyncio
import json
import os
import re
import sys
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

API_KEY    = os.getenv("ANTHROPIC_API_KEY")
MODEL      = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = max(int(os.getenv("MAX_TOKENS", "1024")), 1024)


# ══════════════════════════════════════════════════════════════════════════════
#  MCP SERVER — run with:  python 37_mcp_resources.py --server
# ══════════════════════════════════════════════════════════════════════════════

if "--server" in sys.argv:
    from mcp.server.fastmcp import FastMCP
    from pydantic import Field

    mcp = FastMCP("ResourcesDemo", log_level="ERROR")

    docs: dict[str, str] = {
        "project_plan.md": (
            "Phase 1 (Q1): Core infrastructure and internal testing across 4 backend engineers. "
            "Phase 2 (Q2): Beta rollout to 50 selected enterprise customers. "
            "Phase 3 (Q3): GA release with 99.9% SLA. Estimated 10,000 users at launch."
        ),
        "budget.txt": (
            "Total budget: $480,000. "
            "Engineering: $280,000 (58%). Marketing: $120,000 (25%). Operations: $80,000 (17%). "
            "Q1 actual spend: $95,000. Q2 forecast: $115,000. "
            "Engineering allocation by team: Backend $160k, Frontend $80k, DevOps $40k."
        ),
        "team_notes.md": (
            "Engineering lead: Sarah Chen (sarah@company.com). "
            "Backend: Alice, Bob, Carol, Dave. Frontend: Eve, Frank, Grace. QA: Henry, Irene. "
            "Current sprint velocity: 42 points. Next demo: Friday 3pm. "
            "Blockers: vendor API rate limits (ticket #892), staging environment flaky (ticket #901)."
        ),
        "release_notes.txt": (
            "v1.2.0 (2025-06-01): Fixed login timeout bug. Dashboard load time reduced by 40%%. "
            "v1.1.0 (2025-04-15): Added CSV export. Fixed mobile layout. "
            "v1.0.0 (2025-02-01): Initial release. Core features complete."
        ),
    }

    # ── Direct resource: list all document IDs ────────────────────────────────
    # URI: docs://documents   MIME: application/json   No parameters
    @mcp.resource(
        "docs://documents",
        name="list_documents",
        description="List all available document IDs as a JSON array.",
        mime_type="application/json",
    )
    def list_docs() -> list[str]:
        return list(docs.keys())

    # ── Direct resource: word-count stats for every document ──────────────────
    @mcp.resource(
        "docs://stats",
        name="document_stats",
        description="Word and character counts for every document.",
        mime_type="application/json",
    )
    def doc_stats() -> dict:
        return {
            doc_id: {"words": len(text.split()), "chars": len(text)}
            for doc_id, text in docs.items()
        }

    # ── Templated resource: fetch a specific document ─────────────────────────
    # URI pattern: docs://documents/{doc_id}
    # The SDK extracts {doc_id} from the URI and passes it as a kwarg automatically.
    @mcp.resource(
        "docs://documents/{doc_id}",
        name="fetch_document",
        description="Fetch the full text of a document by its ID.",
        mime_type="text/plain",
    )
    def fetch_doc(doc_id: str) -> str:
        if doc_id not in docs:
            raise ValueError(f"Document '{doc_id}' not found. Use docs://documents to list valid IDs.")
        return docs[doc_id]

    # ── Tool (for contrast): editing IS an action, so it stays a Tool ─────────
    @mcp.tool(
        name="edit_document",
        description="Replace text inside a document. Use this for WRITE operations; use resources for READ.",
    )
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
        return f"'{doc_id}' updated."

    mcp.run()
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  MCP CLIENT + DEMO
# ══════════════════════════════════════════════════════════════════════════════

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def read_resource_text(contents) -> str:
    """Extract text from a list of ResourceContent items."""
    return "\n".join(getattr(item, "text", "") for item in contents)


async def run_demo():
    claude  = Anthropic(api_key=API_KEY)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[__file__, "--server"],
    )

    print("=== MCP Resources Demo ===")
    print(f"Model: {MODEL}")
    print("\nConnecting to MCP server...")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            print("Connected!\n")

            # ── Mode selection ────────────────────────────────────────────────
            print("Choose a mode:")
            print("  [1] Discover resources  -- list all resources and templates the server exposes")
            print("  [2] Read resources      -- fetch each resource and print the content")
            print("  [3] @mention chat       -- type @doc_id in your message to inject a doc into Claude")
            print()
            mode = input("Mode (1/2/3, default 3): ").strip() or "3"

            # ─────────────────────────────────────────────────────────────────
            if mode == "1":
                # ListResources (direct) + ListResourceTemplates (parameterised)
                print("\n--- Direct Resources ---")
                resources = await session.list_resources()
                for res in resources.resources:
                    print(f"  URI       : {res.uri}")
                    print(f"  Name      : {res.name}")
                    print(f"  MIME type : {res.mimeType}")
                    print()

                print("--- Templated Resources ---")
                templates = await session.list_resource_templates()
                for t in templates.resourceTemplates:
                    print(f"  URI template : {t.uriTemplate}")
                    print(f"  Name         : {t.name}")
                    print(f"  Description  : {t.description}")
                    print()

                print("--- Tools (for contrast) ---")
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"  Tool: {tool.name}  <-- action that changes state, NOT a resource")

            # ─────────────────────────────────────────────────────────────────
            elif mode == "2":
                # Read direct resources
                print("\n--- Reading docs://documents ---")
                result = await session.read_resource("docs://documents")
                text   = read_resource_text(result.contents)
                doc_ids = json.loads(text)
                print("Document IDs:", doc_ids)

                print("\n--- Reading docs://stats ---")
                result = await session.read_resource("docs://stats")
                stats  = json.loads(read_resource_text(result.contents))
                for doc_id, s in stats.items():
                    print(f"  {doc_id:<25} {s['words']:>3} words  {s['chars']:>4} chars")

                # Read each document via the templated resource
                print("\n--- Reading each doc via docs://documents/{doc_id} ---")
                for doc_id in doc_ids:
                    uri    = f"docs://documents/{doc_id}"
                    result = await session.read_resource(uri)
                    text   = read_resource_text(result.contents)
                    print(f"\n  [{doc_id}]")
                    print(f"  {text[:120]}{'...' if len(text) > 120 else ''}")

            # ─────────────────────────────────────────────────────────────────
            elif mode == "3":
                # @mention chat — fetch doc list once for autocomplete hint
                resources  = await session.list_resources()
                doc_ids    = json.loads(
                    read_resource_text(
                        (await session.read_resource("docs://documents")).contents
                    )
                )

                print(f"\nAvailable documents for @mention: {', '.join(doc_ids)}")
                print("Prefix a doc name with @ to inject it into your question.")
                print("Example: @budget.txt What is the engineering allocation?")
                print("Type 'exit' to quit.\n")

                messages: list = []

                while True:
                    user_input = input("You : ").strip()
                    if user_input.lower() in ("exit", "quit"):
                        print("Goodbye!")
                        break
                    if not user_input:
                        continue

                    # ── Parse @mentions and inject document content ────────
                    mentions   = re.findall(r"@([\w.\-]+)", user_input)
                    clean_text = re.sub(r"@[\w.\-]+\s*", "", user_input).strip()
                    injected   = []

                    for doc_id in mentions:
                        if doc_id not in doc_ids:
                            print(f"  [warn] '@{doc_id}' not found — skipped. "
                                  f"Available: {', '.join(doc_ids)}")
                            continue
                        uri    = f"docs://documents/{doc_id}"
                        result = await session.read_resource(uri)
                        text   = read_resource_text(result.contents)
                        injected.append(f"[Document: {doc_id}]\n{text}")
                        print(f"  [resource] Fetched '{doc_id}' via {uri}")

                    # Build the full message Claude sees
                    if injected:
                        full_message = (
                            "\n\n".join(injected)
                            + "\n\n"
                            + (clean_text or "Summarise the above document(s).")
                        )
                    else:
                        full_message = user_input

                    messages.append({"role": "user", "content": full_message})

                    response = claude.messages.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        messages=messages,
                    )

                    reply = response.content[-1].text
                    messages.append({"role": "assistant", "content": reply})
                    print(f"\nClaude: {reply}\n")

            else:
                print("Invalid mode.")


if __name__ == "__main__":
    asyncio.run(run_demo())
