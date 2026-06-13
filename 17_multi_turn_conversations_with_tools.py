# Sometimes answering one user question takes more than one tool call — for example "What day is 103 days from
# today?" needs get_current_datetime first, then add_days_to_date with that result; Claude cannot do both in one
# shot because the second tool's input depends on the first tool's output.
# To handle this automatically, you wrap the entire tool-calling sequence in a while-loop: keep calling the API,
# appending Claude's response, running any requested tools, and sending back results until Claude returns
# stop_reason "end_turn" instead of "tool_use" — that means it has enough information to answer.
# The helper functions need a small upgrade: add_user_message and add_assistant_message should accept a plain
# string, a list of blocks, OR a full Message object — that way the same helpers work for normal chat turns
# and tool turns without needing separate code paths.
# The chat() function should also return the full Message object (not just the text) so the conversation loop
# can inspect stop_reason and iterate through blocks; use a separate text_from_message() helper when you
# actually need to display readable text to the user at the end.

# Full example: this program upgrades the chat helpers to accept Message objects, runs a multi-step conversation
# loop that automatically handles as many sequential tool calls as Claude needs, and demonstrates the pattern
# with a question like "What day is 103 days from today?" — showing each tool request and result in the terminal
# so you can follow every round-trip before Claude delivers the final human-readable answer.


import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import Message, ToolParam

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

client = Anthropic(api_key=API_KEY)


# --- Tool Functions ---

def get_current_datetime(date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)


def add_days_to_date(date_string: str, days: int) -> str:
    if not date_string:
        raise ValueError("date_string cannot be empty")
    base = datetime.strptime(date_string, "%Y-%m-%d")
    result = base + timedelta(days=days)
    return result.strftime("%Y-%m-%d (%A)")


# --- Tool Schemas ---

get_current_datetime_schema: ToolParam = {
    "name": "get_current_datetime",
    "description": (
        "Returns the current date and time formatted according to a Python strftime format string. "
        "Use this whenever the user asks about the current time, today's date, or the current day of the week."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": "A Python strftime format string. Default: '%Y-%m-%d %H:%M:%S'.",
                "default": "%Y-%m-%d %H:%M:%S",
            }
        },
        "required": [],
    },
}

add_days_to_date_schema: ToolParam = {
    "name": "add_days_to_date",
    "description": (
        "Adds a number of days to a given date and returns the resulting date with its day name. "
        "Use when the user asks what date it will be in N days. Returns a string like '2024-07-22 (Monday)'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "date_string": {
                "type": "string",
                "description": "Starting date in YYYY-MM-DD format.",
            },
            "days": {
                "type": "integer",
                "description": "Days to add (negative goes back in time).",
            },
        },
        "required": ["date_string", "days"],
    },
}

TOOLS: list[ToolParam] = [get_current_datetime_schema, add_days_to_date_schema]


# --- Upgraded Helper Functions ---
# These now accept a plain string, a list of blocks, OR a full Message object.
# That single change is what lets the same helpers work for normal turns AND tool turns.

def add_user_message(messages: list, message) -> None:
    content = message.content if isinstance(message, Message) else message
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages: list, message) -> None:
    content = message.content if isinstance(message, Message) else message
    messages.append({"role": "assistant", "content": content})


# --- Upgraded chat() — returns the full Message object, not just text ---
# Accepts optional tools, system prompt, temperature, and stop_sequences.

def chat(
    messages: list,
    system: str | None = None,
    temperature: float = 1.0,
    stop_sequences: list[str] = [],
    tools: list[ToolParam] | None = None,
) -> Message:
    params = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }
    if tools:
        params["tools"] = tools
    if system:
        params["system"] = system
    return client.messages.create(**params)


# --- Text extractor: pulls readable text out of a Message for display ---

def text_from_message(message: Message) -> str:
    return "\n".join(block.text for block in message.content if block.type == "text")


# --- Tool dispatcher ---

def run_tool(name: str, inputs: dict) -> tuple[str, bool]:
    try:
        if name == "get_current_datetime":
            return get_current_datetime(**inputs), False
        if name == "add_days_to_date":
            return add_days_to_date(**inputs), False
        return f"Unknown tool: {name}", True
    except Exception as e:
        return str(e), True


# --- Tool result builder ---

def build_tool_results(message: Message) -> list[dict]:
    results = []
    for block in message.content:
        if block.type != "tool_use":
            continue
        print(f"    tool : {block.name}({block.input})")
        output, is_error = run_tool(block.name, block.input)
        print(f"    result: {output!r}")
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
            "is_error": is_error,
        })
    return results


# --- Multi-turn conversation loop ---
# Keeps going until Claude says stop_reason == "end_turn".
# Each iteration: get response → if tool_use, run tools and send results → repeat.

def run_conversation(messages: list) -> Message:
    step = 0
    while True:
        step += 1
        print(f"\n  [round {step}] calling Claude...")
        response = chat(messages, tools=TOOLS)
        print(f"  [round {step}] stop_reason={response.stop_reason!r}")

        add_assistant_message(messages, response)

        if response.stop_reason != "tool_use":
            # Claude is done — final answer is inside response
            break

        tool_results = build_tool_results(response)
        add_user_message(messages, tool_results)

    return response


try:
    print("=== Multi-Turn Conversations with Tools Demo ===")
    print("Claude will automatically chain tool calls when one answer depends on another.")
    print()
    print("Try: 'What day of the week is 103 days from today?'  (requires 2 tool calls)")
    print("  or 'What time is it right now?'                   (single tool call)")
    print("  or 'What date was it 200 days ago?'               (requires 2 tool calls)")
    print("\nType 'exit' to quit.\n")

    messages: list = []

    while True:
        user_input = input("You : ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        add_user_message(messages, user_input)
        final_response = run_conversation(messages)
        reply = text_from_message(final_response)
        add_assistant_message(messages, reply)

        print(f"\nClaude : {reply}\n")

except Exception as e:
    print(f"\nError: {e}")
