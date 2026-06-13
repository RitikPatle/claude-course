# Normally Claude preprocesses every single request from scratch — tokenising, embedding, and
# analysing context — then throws all that work away the moment it sends a response. Prompt
# caching tells Claude to save that preprocessing work for up to one hour so that the next
# request containing the same content can skip the expensive processing and reuse the saved
# result instead, making follow-up requests both faster and cheaper.
# Caching is NOT automatic — you opt in by adding "cache_control": {"type": "ephemeral"} to
# a specific block; everything before that block gets cached, and nothing after it does.
# The content must be byte-for-byte identical on follow-up requests (even adding "please"
# invalidates the cache), must total at least 1024 tokens to qualify, and you can place up
# to four breakpoints per request across tools, system prompt, and messages in that order.
# The shorthand string format for system prompts and text blocks does not support cache_control,
# so you must use the longhand list-of-blocks format instead.
# When tools are cached you clone the list and modify only the copy so you don't accidentally
# mutate shared state; the breakpoint goes on the last tool in the list.
# After each request the API tells you exactly what happened: cache_creation_input_tokens means
# this was the first request and Claude wrote to the cache; cache_read_input_tokens means the
# cache was hit and Claude skipped reprocessing that content.

# Full example: this program builds a long system prompt and a set of tool schemas that together
# exceed 1024 tokens, makes three consecutive requests with caching enabled, and prints the
# usage stats after each one — so you can watch cache_creation on the first call turn into
# cache_read on the second, then see what happens when the system prompt changes on the third.


import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

API_KEY    = os.getenv("ANTHROPIC_API_KEY")
# Prompt caching requires a model that supports it — claude-sonnet-4-6 and similar work;
# claude-haiku-4-5 does NOT support prompt caching. Use THINKING_MODEL as the caching model.
MODEL      = os.getenv("THINKING_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = max(int(os.getenv("MAX_TOKENS", "1024")), 512)

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")

# Beta header enables prompt caching on supported models
client = Anthropic(
    api_key=API_KEY,
    default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
)


# ─────────────────────────────────────────────
# Long system prompt — must exceed 1024 tokens to be cache-eligible.
# We repeat a detailed coding-assistant prompt to ensure it qualifies.
# ─────────────────────────────────────────────

BASE_SYSTEM = """
You are an expert software engineering assistant specialising in Python, system design, and API development.

Your responsibilities:
- Write clean, well-documented Python code that follows PEP-8 conventions
- Explain complex technical concepts in clear, accessible language
- Review code for bugs, security vulnerabilities, and performance issues
- Suggest architectural improvements and design patterns
- Help debug errors by analysing stack traces and identifying root causes
- Recommend appropriate libraries and frameworks for specific use cases
- Write comprehensive unit tests using pytest
- Explain the trade-offs between different technical approaches

Coding standards you enforce:
- All functions must have type annotations and docstrings
- Variable names must be descriptive (no single-letter names except loop indices)
- Functions should do one thing only (single responsibility principle)
- Error handling must be explicit — never use bare except clauses
- Database queries must use parameterised statements to prevent SQL injection
- Sensitive data must never be logged or printed
- All external API calls must include timeout parameters
- Configuration must come from environment variables, never hardcoded

When reviewing code you always:
1. Identify security vulnerabilities first
2. Flag performance bottlenecks
3. Point out maintainability issues
4. Suggest refactoring opportunities
5. Provide specific, actionable improvements with code examples

You respond with structured output: a brief summary, numbered findings, and a corrected code block.
""".strip()

# Repeat to well exceed 1024 tokens (minimum required for caching)
SYSTEM_PROMPT = (BASE_SYSTEM + "\n\n") * 8


# ─────────────────────────────────────────────
# Tool schemas — also candidates for caching
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "run_code_review",
        "description": "Reviews Python code for bugs, security issues, and style violations. Returns structured findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python code to review."},
                "focus": {"type": "string", "description": "Optional focus area: 'security', 'performance', or 'style'."},
            },
            "required": ["code"],
        },
    },
    {
        "name": "search_documentation",
        "description": "Searches official Python and library documentation for a given topic or function name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic or function to look up."},
                "library": {"type": "string", "description": "Optional specific library name, e.g. 'requests', 'pandas'."},
            },
            "required": ["query"],
        },
    },
]


# ─────────────────────────────────────────────
# Cache-aware chat function
# Adds cache_control to system prompt and last tool schema.
# Uses longhand block format (required for cache_control).
# ─────────────────────────────────────────────

def chat_with_cache(
    messages: list,
    system: str | None = None,
    tools: list | None = None,
    use_cache: bool = True,
) -> object:
    params = {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "messages":   messages,
    }

    # System prompt — must use longhand list format to attach cache_control
    if system:
        if use_cache:
            params["system"] = [
                {
                    "type":          "text",
                    "text":          system,
                    "cache_control": {"type": "ephemeral"},   # ← cache everything up to here
                }
            ]
        else:
            params["system"] = system   # shorthand string — no caching

    # Tools — clone the list, add cache_control only to the last tool
    if tools:
        if use_cache:
            tools_clone       = tools.copy()
            last_tool         = tools_clone[-1].copy()           # ← clone before modifying
            last_tool["cache_control"] = {"type": "ephemeral"}  # ← breakpoint on last tool
            tools_clone[-1]   = last_tool
            params["tools"]   = tools_clone
        else:
            params["tools"] = tools

    return client.messages.create(**params)


# ─────────────────────────────────────────────
# Usage stats printer — shows exactly what happened in the cache
# ─────────────────────────────────────────────

def print_cache_stats(response, label: str = "") -> None:
    u = response.usage
    created  = getattr(u, "cache_creation_input_tokens", 0) or 0
    read     = getattr(u, "cache_read_input_tokens", 0) or 0
    normal   = getattr(u, "input_tokens", 0) or 0
    output   = getattr(u, "output_tokens", 0) or 0

    if created > 0 and read > 0:
        status = "CACHE HIT + new write (system cached, new turn added)"
    elif created > 0:
        status = "CACHE WRITE (first time, nothing was cached yet)"
    elif read > 0:
        status = "CACHE HIT   (100% reused, nothing new to cache)"
    else:
        status = "NO CACHE    (caching not active or content too short)"

    print(f"\n  [{label}] {status}")
    print(f"    input_tokens              : {normal:>6}")
    print(f"    cache_creation_input_tokens: {created:>6}  <- paid to write to cache")
    print(f"    cache_read_input_tokens    : {read:>6}  <- cheaper than normal processing")
    print(f"    output_tokens             : {output:>6}")


# ─────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────

print("=== Prompt Caching Demo ===")
print(f"Model  : {MODEL}")
print(f"System prompt length: ~{len(SYSTEM_PROMPT.split())} words")
print()

print("Choose a mode:")
print("  [1] Automatic demo -- 3 requests showing cache write -> hit -> invalidation")
print("  [2] Interactive chat -- caching active, watch stats after each message")
print()
mode = input("Mode (1/2, default 1): ").strip() or "1"

QUESTION = "Write a Python function that reads a CSV file safely. Keep it short."

if mode == "1":
    messages = [{"role": "user", "content": QUESTION}]

    # ── Request 1: first time — writes to cache ───────────────────────────────
    print("\n" + "=" * 60)
    print("REQUEST 1 -- same system prompt, first time  ->  writes to cache")
    print("=" * 60)
    r1 = chat_with_cache(messages, system=SYSTEM_PROMPT, tools=TOOLS, use_cache=True)
    print_cache_stats(r1, "Request 1")
    text_blocks = [b for b in r1.content if b.type == "text"]
    print(f"\n  Claude: {text_blocks[-1].text[:200] if text_blocks else '(tool call)'}...")

    # ── Request 2: identical content — reads from cache ───────────────────────
    print("\n" + "=" * 60)
    print("REQUEST 2 -- identical system prompt  ->  cache hit, faster + cheaper")
    print("=" * 60)
    r2 = chat_with_cache(messages, system=SYSTEM_PROMPT, tools=TOOLS, use_cache=True)
    print_cache_stats(r2, "Request 2")
    text_blocks = [b for b in r2.content if b.type == "text"]
    print(f"\n  Claude: {text_blocks[-1].text[:200] if text_blocks else '(tool call)'}...")

    # ── Request 3: system prompt changed — cache invalidated ──────────────────
    print("\n" + "=" * 60)
    print("REQUEST 3 -- system prompt changed by ONE word  ->  cache invalidated")
    print("=" * 60)
    import time
    # Adding a timestamp makes this variant unique each run so it's always a fresh cache miss
    modified_system = SYSTEM_PROMPT + f"\n# Session {int(time.time())} — Always respond in bullet points."
    r3 = chat_with_cache(messages, system=modified_system, tools=TOOLS, use_cache=True)
    print_cache_stats(r3, "Request 3")
    text_blocks = [b for b in r3.content if b.type == "text"]
    print(f"\n  Claude: {text_blocks[-1].text[:200] if text_blocks else '(tool call)'}...")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("  Request 1 -> first run: cache_creation | later runs: cache_read (system cached)")
    print("  Request 2 -> same system prompt = cache_read (system reused, turn written)")
    print("  Request 3 -> ALWAYS cache_creation (unique timestamp = never cached)")
    print("=" * 60)

elif mode == "2":
    print("\nInteractive chat with caching. Stats shown after every message.")
    print("Type 'exit' to quit.\n")

    messages: list = []
    turn = 0

    while True:
        user_input = input("You : ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        turn += 1
        messages.append({"role": "user", "content": user_input})
        response = chat_with_cache(messages, system=SYSTEM_PROMPT, tools=TOOLS, use_cache=True)
        print_cache_stats(response, f"Turn {turn}")

        text_blocks = [b for b in response.content if b.type == "text"]
        reply = text_blocks[-1].text if text_blocks else "(tool call - no text output)"
        messages.append({"role": "assistant", "content": reply})
        print(f"\nClaude : {reply}\n")

else:
    print("Invalid mode.")
