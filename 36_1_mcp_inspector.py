# The MCP Inspector is a browser-based debugging tool that ships with the Python MCP SDK —
# run "mcp dev this_file.py" in your terminal and it starts a local web server on port 6277
# where you can click "Connect", browse your server's tools, fill in parameters, hit
# "Run Tool", and see the result instantly, all without wiring up Claude or any client code.
# This is the MCP equivalent of Postman: instead of building a full application just to
# verify that your read_document tool returns the right text, you test each tool in isolation,
# chain calls (edit a document, then read it back to confirm the change took effect), and
# debug error messages while your server code is still fresh in front of you.
# The inspector talks to your server over the same stdio transport that a real MCP client
# would use — so if it works in the inspector, it will work in your app; there is no gap
# between the two environments.
# Note: the inspector's UI changes frequently since it is under active development, but the
# core workflow (Connect -> Tools -> List Tools -> select -> fill params -> Run Tool) stays
# the same, and the tool schema and result format that Claude sees are exactly what you see.

# Full example: this file is a self-contained FastMCP server with four tools — listing,
# reading, editing, and summarising in-memory documents. Run it with the inspector using
# "mcp dev 36_mcp_inspector.py" (no --server flag needed — just the file), then open the
# browser URL it prints, click Connect, go to Tools, and test each tool interactively; you
# can edit a document and immediately read it back to verify the change, all in one browser tab.

# ─────────────────────────────────────────────
# HOW TO RUN THE INSPECTOR:
#
#   .venv\Scripts\mcp.exe dev 36_mcp_inspector.py
#
# Then open the URL shown in the terminal (usually http://localhost:5173).
# Click "Connect" in the top-left, navigate to "Tools", click "List Tools".
# Select any tool, fill in the parameters, and click "Run Tool".
#
# The server also still works as a plain stdio MCP server for 35_mcp.py —
# just run it in --server mode by importing it or pointing 35_mcp.py at it.
# ─────────────────────────────────────────────

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("DocumentInspectorMCP", log_level="ERROR")

# ── In-memory document store ──────────────────────────────────────────────────

docs: dict[str, str] = {
    "project_plan.md": (
        "Phase 1 (Q1): Core infrastructure and internal testing. "
        "Phase 2 (Q2): Beta rollout to enterprise customers. "
        "Phase 3 (Q3): GA release with full SLA guarantees."
    ),
    "budget.txt": (
        "Total budget: $480,000. "
        "Engineering: $280,000 (58%). "
        "Marketing: $120,000 (25%). "
        "Operations: $80,000 (17%). "
        "Q1 spend: $95,000."
    ),
    "team_notes.md": (
        "Engineering lead: Sarah Chen. "
        "Backend: 4 engineers. Frontend: 3 engineers. QA: 2 testers. "
        "Next sync: Monday 10am. Blockers: vendor API rate limits."
    ),
    "release_notes.txt": (
        "v1.2.0: Fixed login timeout. Dashboard 40% faster. "
        "v1.1.0: CSV export. Mobile layout fixes. "
        "v1.0.0: Initial release."
    ),
}


# ── Tool 1: list_documents ────────────────────────────────────────────────────
# Inspector test: click "List Tools", select this tool, click "Run Tool" (no params).
# Expected result: four document IDs, one per line.

@mcp.tool(
    name="list_documents",
    description="Return a list of all available document IDs.",
)
def list_documents() -> str:
    return "\n".join(docs.keys())


# ── Tool 2: read_doc_contents ─────────────────────────────────────────────────
# Inspector test: enter doc_id = "budget.txt", click "Run Tool".
# Expected result: the full budget text.
# Error test: enter doc_id = "missing.txt" — you should see the ValueError message.

@mcp.tool(
    name="read_doc_contents",
    description="Read the full text content of a document.",
)
def read_document(
    doc_id: str = Field(description="Document ID to read, e.g. 'budget.txt'"),
) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document '{doc_id}' not found. Run list_documents to see valid IDs.")
    return docs[doc_id]


# ── Tool 3: edit_document ─────────────────────────────────────────────────────
# Inspector test: doc_id="budget.txt", old_str="$480,000", new_str="$520,000".
# Then run read_doc_contents on budget.txt — you should see the updated number.
# This chained test (edit → read back) verifies state changes work correctly.

@mcp.tool(
    name="edit_document",
    description="Replace a specific string inside a document with new text.",
)
def edit_document(
    doc_id:  str = Field(description="ID of the document to edit."),
    old_str: str = Field(description="Exact text to replace (case-sensitive, whitespace-sensitive)."),
    new_str: str = Field(description="Replacement text."),
) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document '{doc_id}' not found.")
    if old_str not in docs[doc_id]:
        raise ValueError(f"String not found in '{doc_id}'. Check for exact match including punctuation.")
    docs[doc_id] = docs[doc_id].replace(old_str, new_str)
    return f"OK — '{doc_id}' updated. Run read_doc_contents to verify the change."


# ── Tool 4: get_document_stats ────────────────────────────────────────────────
# Inspector test: run with no params.
# Expected result: word count and character count for every document.

@mcp.tool(
    name="get_document_stats",
    description="Return word count and character count for every document in the store.",
)
def get_document_stats() -> str:
    lines = ["Document stats:"]
    for doc_id, content in docs.items():
        words = len(content.split())
        chars = len(content)
        lines.append(f"  {doc_id:<25} {words:>4} words  {chars:>5} chars")
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────
# mcp.run() is called when mcp dev or mcp run starts this file.
# It is also safe to import this file from 35_mcp.py as a server module.

if __name__ == "__main__":
    mcp.run()
