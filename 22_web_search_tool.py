# The web search tool is Claude's most hands-off built-in tool — unlike the text editor where
# Claude decides what to do but YOU write the code that does it, the web search tool needs
# no implementation at all; Claude handles the entire search process end-to-end automatically.
# All you provide is a tiny schema with three fields: type ("web_search_20250305"), name
# ("web_search"), and max_uses (how many searches Claude is allowed per conversation turn).
# The max_uses cap matters because Claude may decide to run follow-up searches after reading
# initial results — without a cap, one question could trigger many expensive API calls.
# You can also add allowed_domains to restrict Claude to trusted sources only (e.g. "nih.gov"
# for medical questions), which is much more reliable than hoping Claude picks good sources itself.
# The response contains several block types: TextBlock (Claude's commentary), ServerToolUseBlock
# (the exact query Claude sent), WebSearchToolResultBlock (raw results), and citation blocks
# (quoted text + source URL) — printing all of them gives full transparency into how Claude
# found and used the information to build its answer.

# Full example: this program sets up both an unrestricted web search schema and a domain-restricted
# one, runs a multi-turn chat loop where Claude decides when a web search is needed, then prints
# every block in the response — search queries, result titles/URLs, and inline citations — so you
# can see exactly which sources Claude used and how they were woven into the final answer.


import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

client = Anthropic(api_key=API_KEY)


# ─────────────────────────────────────────────
# Web search schemas
# No implementation needed — Claude does the actual searching automatically.
# ─────────────────────────────────────────────

# General web search — Claude can search any website
web_search_schema = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}

# Domain-restricted search — only returns results from trusted sources
web_search_nih_schema = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
    "allowed_domains": ["nih.gov", "pubmed.ncbi.nlm.nih.gov"],
}


# ─────────────────────────────────────────────
# Response renderer
# Prints every block type so you can see the full search + citation trail.
# ─────────────────────────────────────────────

def render_response(response) -> str:
    final_text_parts = []

    for block in response.content:
        block_type = getattr(block, "type", type(block).__name__)

        if block_type == "text":
            # Claude's commentary or final answer
            final_text_parts.append(block.text)

        elif block_type == "server_tool_use":
            # The exact query Claude sent to the search engine
            query = block.input.get("query", "") if hasattr(block, "input") else ""
            print(f"\n  [search query] {query!r}")

        elif block_type == "web_search_tool_result":
            # Container holding individual search results
            print(f"\n  [search results]")
            if hasattr(block, "content"):
                for result in block.content:
                    r_type = getattr(result, "type", "")
                    if r_type == "web_search_result":
                        title = getattr(result, "title", "No title")
                        url   = getattr(result, "url",   "No URL")
                        print(f"    • {title}")
                        print(f"      {url}")

        elif block_type == "tool_result":
            # Generic tool result fallback
            print(f"\n  [tool_result block received]")

        # Citation blocks appear embedded in text blocks in some SDK versions;
        # in others they are top-level — handle both gracefully.
        citations = getattr(block, "citations", []) or []
        for cite in citations:
            cited_text = getattr(cite, "cited_text", "")
            url        = getattr(cite, "url", "")
            title      = getattr(cite, "title", "")
            if url:
                print(f"\n  [citation] \"{cited_text[:80]}\"")
                print(f"             {title} — {url}")

    return "\n".join(final_text_parts)


# ─────────────────────────────────────────────
# Chat — passes the chosen schema to the API
# ─────────────────────────────────────────────

def chat(messages: list, schema: dict) -> object:
    return client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=[schema],
        messages=messages,
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

try:
    print("=== Web Search Tool Demo ===")
    print("Claude will automatically decide when a web search is needed.")
    print()
    print("Modes:")
    print("  [1] General search  — any website")
    print("  [2] NIH only        — restricted to nih.gov / pubmed")
    print()

    mode = input("Choose mode (1 or 2, default 1): ").strip() or "1"
    schema = web_search_nih_schema if mode == "2" else web_search_schema

    print(f"\nUsing schema: {schema['type']}")
    if "allowed_domains" in schema:
        print(f"Restricted to: {schema['allowed_domains']}")
    print()
    print("Try: 'What are the latest developments in quantum computing?'")
    print("  or 'What are the health benefits of magnesium?' (good for NIH mode)")
    print("  or 'What happened in the news today?'")
    print("\nType 'exit' to quit.\n")

    messages: list = []

    while True:
        user_input = input("You : ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        response = chat(messages, schema)

        reply = render_response(response)
        messages.append({"role": "assistant", "content": response.content})

        print(f"\nClaude : {reply}\n")

except Exception as e:
    print(f"\nError: {e}")
    print("\nNote: Web search requires the tool to be enabled in your Anthropic console.")
    print("Enable it at: https://console.anthropic.com/settings/privacy")
