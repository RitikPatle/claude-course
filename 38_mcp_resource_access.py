# This lesson is about the CLIENT side of resource access — specifically, wrapping the raw
# session.read_resource() call inside a reusable MCPClient class so the rest of your
# application never has to know about MCP protocol details.
# Two things make this different from calling the session directly: first, the URI must be
# wrapped in AnyUrl (a Pydantic type) to satisfy the SDK's type system — passing a plain
# string raises a validation error; second, you must inspect the returned content object
# with isinstance(resource, types.TextResourceContents) and then check resource.mimeType
# to decide how to parse it — application/json gets json.loads(), text/plain is returned
# as-is, because the SDK gives you the raw serialised bytes and you choose the interpretation.
# Keeping all of this inside MCPClient.read_resource() means every caller gets back a clean
# Python object (a list, a dict, or a string) and never touches MCP types directly — which
# is the "separation of concerns" the course emphasises: the client handles the protocol,
# the application handles the logic.

# Full example: this file is an MCP server (--server flag) plus a standalone MCPClient
# class with list_resources(), list_resource_templates(), and read_resource() methods that
# all handle MIME-type parsing internally.  The demo connects, shows every resource, reads
# each one to prove the JSON/text branching works correctly, then runs an @mention chat
# where typing "@budget.txt ..." auto-injects the document via MCPClient.read_resource()
# before sending to Claude — showing the full pipeline with clean class-based design.


import asyncio
import json
import os
import re
import sys
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp import types
from pydantic import AnyUrl          # ← required for proper URI type handling

load_dotenv()

API_KEY    = os.getenv("ANTHROPIC_API_KEY")
MODEL      = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = max(int(os.getenv("MAX_TOKENS", "1024")), 1024)


# ══════════════════════════════════════════════════════════════════════════════
#  MCP SERVER  —  python 38_mcp_resource_access.py --server
# ══════════════════════════════════════════════════════════════════════════════

if "--server" in sys.argv:
    from mcp.server.fastmcp import FastMCP
    from pydantic import Field

    mcp = FastMCP("ResourceAccessDemo", log_level="ERROR")

    docs: dict[str, str] = {
        "project_plan.md": (
            "Phase 1 (Q1): Core infrastructure and internal testing. "
            "Phase 2 (Q2): Beta rollout to 50 enterprise customers. "
            "Phase 3 (Q3): GA release with 99.9% SLA and estimated 10,000 users."
        ),
        "budget.txt": (
            "Total budget: $480,000. "
            "Engineering: $280,000 (58%). Marketing: $120,000 (25%). "
            "Operations: $80,000 (17%). Q1 spend: $95,000."
        ),
        "team_notes.md": (
            "Lead: Sarah Chen. Backend: 4 engineers. Frontend: 3 engineers. "
            "QA: 2 testers. Next sync: Monday 10am. "
            "Blocker: vendor API rate limits (ticket #892)."
        ),
        "release_notes.txt": (
            "v1.2.0: Fixed login timeout. Dashboard 40% faster. "
            "v1.1.0: CSV export. Mobile layout fixes. "
            "v1.0.0: Initial release."
        ),
    }

    # Direct resource — returns JSON (list)
    @mcp.resource("docs://documents", mime_type="application/json")
    def list_docs() -> list[str]:
        return list(docs.keys())

    # Direct resource — returns JSON (dict with stats)
    @mcp.resource("docs://stats", mime_type="application/json")
    def doc_stats() -> dict:
        return {k: {"words": len(v.split()), "chars": len(v)} for k, v in docs.items()}

    # Templated resource — returns plain text
    @mcp.resource("docs://documents/{doc_id}", mime_type="text/plain")
    def fetch_doc(doc_id: str) -> str:
        if doc_id not in docs:
            raise ValueError(f"Document '{doc_id}' not found.")
        return docs[doc_id]

    mcp.run()
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  MCPClient CLASS
#  Wraps all MCP session calls so application code never touches protocol types.
# ══════════════════════════════════════════════════════════════════════════════

class MCPClient:
    """
    A reusable client that wraps an active MCP ClientSession.
    All methods return plain Python objects — callers never see MCP types.
    """

    def __init__(self, session: ClientSession):
        self._session = session

    # ── Discover what the server exposes ─────────────────────────────────────

    async def list_resources(self) -> list[dict]:
        """Return direct (static) resources as a list of dicts."""
        result = await self._session.list_resources()
        return [
            {"uri": str(r.uri), "name": r.name, "mime_type": r.mimeType}
            for r in result.resources
        ]

    async def list_resource_templates(self) -> list[dict]:
        """Return templated resources as a list of dicts."""
        result = await self._session.list_resource_templates()
        return [
            {"uri_template": t.uriTemplate, "name": t.name, "mime_type": t.mimeType}
            for t in result.resourceTemplates
        ]

    # ── Fetch a resource by URI ───────────────────────────────────────────────

    async def read_resource(self, uri: str) -> Any:
        """
        Fetch a resource and return a parsed Python object.

        - Wraps the URI in AnyUrl (required by the SDK)
        - Checks isinstance to confirm we got text content
        - Branches on mimeType: JSON -> parse with json.loads, plain text -> return as-is
        """
        result   = await self._session.read_resource(AnyUrl(uri))  # ← AnyUrl wrapper
        resource = result.contents[0]                               # first (usually only) item

        if isinstance(resource, types.TextResourceContents):        # ← isinstance check
            if resource.mimeType == "application/json":
                return json.loads(resource.text)   # list or dict — already a Python object
            return resource.text                   # plain string

        # Non-text content (binary blobs etc.) — return raw for the caller to handle
        return resource


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO  —  connects, exercises all MCPClient methods, then @mention chat
# ══════════════════════════════════════════════════════════════════════════════

async def run():
    claude        = Anthropic(api_key=API_KEY)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[__file__, "--server"],
    )

    print("=== MCP Resource Access Demo ===")
    print(f"Model: {MODEL}\n")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            client = MCPClient(session)   # ← wrap session in our clean class

            print("Choose a mode:")
            print("  [1] Inspect   -- show all resources/templates and their parsed types")
            print("  [2] Read all  -- fetch every resource and print returned Python type")
            print("  [3] @mention  -- chat with @doc_id injection via MCPClient.read_resource")
            print()
            mode = input("Mode (1/2/3, default 3): ").strip() or "3"

            # ── Mode 1: Discover ──────────────────────────────────────────────
            if mode == "1":
                print("\n--- Direct Resources ---")
                for r in await client.list_resources():
                    print(f"  uri       : {r['uri']}")
                    print(f"  name      : {r['name']}")
                    print(f"  mime_type : {r['mime_type']}")
                    print()

                print("--- Templated Resources ---")
                for t in await client.list_resource_templates():
                    print(f"  uri_template : {t['uri_template']}")
                    print(f"  name         : {t['name']}")
                    print(f"  mime_type    : {t['mime_type']}")
                    print()

            # ── Mode 2: Read and show parsed types ────────────────────────────
            elif mode == "2":
                tests = [
                    # (uri, description)
                    ("docs://documents",            "direct JSON — expect list"),
                    ("docs://stats",                "direct JSON — expect dict"),
                    ("docs://documents/budget.txt", "templated text/plain — expect str"),
                    ("docs://documents/team_notes.md", "templated text/plain — expect str"),
                ]

                for uri, desc in tests:
                    print(f"\n--- {uri}  ({desc}) ---")
                    data = await client.read_resource(uri)
                    print(f"  Python type : {type(data).__name__}")

                    if isinstance(data, list):
                        print(f"  Value       : {data}")
                    elif isinstance(data, dict):
                        # Pretty-print stats dict
                        for doc_id, stats in data.items():
                            print(f"  {doc_id:<25} {stats}")
                    else:
                        print(f"  Value       : {str(data)[:120]}")

                print("\n--- MIME branching summary ---")
                print("  application/json  -> json.loads()  -> list or dict")
                print("  text/plain        -> raw string    -> str")

            # ── Mode 3: @mention chat ─────────────────────────────────────────
            elif mode == "3":
                # Fetch doc list once for the autocomplete hint
                doc_ids: list[str] = await client.read_resource("docs://documents")

                print(f"\nAvailable: {', '.join(doc_ids)}")
                print("Use @doc_id in your message to inject a document into the prompt.")
                print("Example:  @budget.txt How much has been spent so far?")
                print("Type 'exit' to quit.\n")

                messages: list = []

                while True:
                    user_input = input("You : ").strip()
                    if user_input.lower() in ("exit", "quit"):
                        print("Goodbye!")
                        break
                    if not user_input:
                        continue

                    # Parse @mentions
                    mentions   = re.findall(r"@([\w.\-]+)", user_input)
                    clean_text = re.sub(r"@[\w.\-]+\s*", "", user_input).strip()
                    injected   = []

                    for doc_id in mentions:
                        if doc_id not in doc_ids:
                            print(f"  [warn] '@{doc_id}' not found. "
                                  f"Available: {', '.join(doc_ids)}")
                            continue

                        # ← The key call: MCPClient.read_resource does AnyUrl + MIME parsing
                        text = await client.read_resource(f"docs://documents/{doc_id}")
                        injected.append(f"[Document: {doc_id}]\n{text}")
                        print(f"  [MCP] read_resource('docs://documents/{doc_id}') "
                              f"-> {len(text)} chars injected")

                    full_message = (
                        ("\n\n".join(injected) + "\n\n") if injected else ""
                    ) + (clean_text or "Summarise the document(s) above.")

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
    asyncio.run(run())
