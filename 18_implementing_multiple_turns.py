# This file refines the multi-turn loop from the previous lesson with three concrete improvements.
# First: wrap every tool output in json.dumps() before sending it back — tool outputs can be strings,
# numbers, dicts, or lists, and json.dumps() serializes all of them safely into a single string that
# the API always accepts; passing a raw Python object can cause type errors on complex tool outputs.
# Second: print Claude's intermediate text responses INSIDE the loop, not just at the end — Claude
# often says things like "Let me get today's date first..." between tool calls, and showing those
# messages in real time makes the conversation feel alive rather than just frozen until the final answer.
# Third: split the tool-handling into two clearly named functions — run_tools(message) filters all the
# tool_use blocks from a response and loops over them, while run_tool(name, input) executes exactly
# one function; this separation makes it trivial to add new tools because you only touch run_tool().

# Full example: this program runs a multi-turn tool conversation, prints Claude's commentary at every
# intermediate round, serializes all tool outputs with json.dumps(), and uses the clean run_tools /
# run_tool split so you can clearly see what each layer is responsible for — making the full loop easy
# to extend with any new tool by adding just one elif to run_tool().


import json
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


# --- Helpers ---

def add_user_message(messages: list, message) -> None:
    content = message.content if isinstance(message, Message) else message
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages: list, message) -> None:
    content = message.content if isinstance(message, Message) else message
    messages.append({"role": "assistant", "content": content})


def chat(messages: list, tools: list[ToolParam] | None = None) -> Message:
    params = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
    }
    if tools:
        params["tools"] = tools
    return client.messages.create(**params)


def text_from_message(message: Message) -> str:
    return "\n".join(block.text for block in message.content if block.type == "text")


# --- Layer 1: run one tool by name, return its output ---
# To add a new tool: just add one elif here. Nothing else changes.

def run_tool(tool_name: str, tool_input: dict):
    if tool_name == "get_current_datetime":
        return get_current_datetime(**tool_input)
    elif tool_name == "add_days_to_date":
        return add_days_to_date(**tool_input)
    raise ValueError(f"Unknown tool: {tool_name}")


# --- Layer 2: process ALL tool_use blocks in one response ---
# Filters out tool_use blocks, runs each one, wraps result in json.dumps(),
# and returns a list of tool_result blocks ready to send back.

def run_tools(message: Message) -> list[dict]:
    tool_requests = [block for block in message.content if block.type == "tool_use"]
    tool_result_blocks = []

    for tool_request in tool_requests:
        print(f"    [tool] {tool_request.name}({tool_request.input})")
        try:
            tool_output = run_tool(tool_request.name, tool_request.input)
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": json.dumps(tool_output),   # <-- safe serialization for any output type
                "is_error": False,
            })
            print(f"    [result] {json.dumps(tool_output)}")
        except Exception as e:
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": f"Error: {e}",
                "is_error": True,
            })
            print(f"    [error] {e}")

    return tool_result_blocks


# --- Conversation loop ---
# NEW: print intermediate text responses inside the loop so the user sees
# Claude's commentary ("Let me get today's date first...") in real time.
# Returns the final Message object so the caller can extract the reply text.

def run_conversation(messages: list) -> Message:
    while True:
        response = chat(messages, tools=TOOLS)
        add_assistant_message(messages, response)

        # Print whatever text Claude said in this round (may be empty during tool calls)
        intermediate_text = text_from_message(response)
        if intermediate_text:
            print(f"\n  Claude (thinking): {intermediate_text}")

        if response.stop_reason != "tool_use":
            return response

        tool_result_blocks = run_tools(response)
        add_user_message(messages, tool_result_blocks)


try:
    print("=== Implementing Multiple Turns Demo ===")
    print("New in this file vs file 17:")
    print("  • json.dumps() wraps every tool output for safe serialization")
    print("  • Claude's intermediate commentary prints inside the loop")
    print("  • run_tools() / run_tool() clean two-layer split")
    print()
    print("Try: 'What day of the week is 103 days from today?'  (2 tool calls, see intermediate text)")
    print("  or 'What time is it right now?'                   (1 tool call)")
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
