# When Claude uses a tool, its response is no longer a single text block — it becomes a list of blocks that can
# include both a TextBlock (Claude's commentary like "Let me look that up for you") and a ToolUseBlock (the
# structured call containing the tool name, a unique ID, and the input parameters).
# Your code must preserve the entire content list when appending the assistant turn to conversation history;
# stripping it down to just the text causes Claude to lose track of which tool call it made and the API will
# reject the next request with a validation error.
# After executing the tool, you send back a "tool_result" user message keyed by the same tool_use_id so Claude
# can match the result to the call, then Claude continues and issues a final "end_turn" text reply.
# Helper functions like add_user_message() and add_assistant_message() that previously stored only strings must
# be updated to accept either a string or a raw content list so they work for both normal turns and tool turns.
# Inspecting each block's .type attribute ("text" vs "tool_use") is the standard way to route handling;
# never assume there is only one block or that the first block is always text.

# Full example: this program defines a get_current_datetime tool, sends a user question that requires the
# current time, then walks through every block Claude returns — printing the text block's commentary and the
# tool-use block's name/inputs — executes the function, sends the result back, and prints Claude's final
# answer, giving you a transparent, end-to-end view of the complete multi-block message flow.


import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import ToolParam

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
        "Use this tool whenever the user asks about the current time, today's date, or the current day of the week. "
        "The tool reads the system clock at the moment it is called, so the result is always accurate. "
        "Returns a formatted string such as '2024-06-12 14:30:00' by default."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": (
                    "A Python strftime format string controlling the output. "
                    "Examples: '%Y-%m-%d' for date only, '%H:%M:%S' for time only."
                ),
                "default": "%Y-%m-%d %H:%M:%S",
            }
        },
        "required": [],
    },
}

add_days_to_date_schema: ToolParam = {
    "name": "add_days_to_date",
    "description": (
        "Adds a number of days to a given date and returns the resulting date along with its day name. "
        "Use this when the user asks what date it will be in N days, or what day a future date falls on. "
        "Returns a string like '2024-07-22 (Monday)'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "date_string": {
                "type": "string",
                "description": "The starting date in YYYY-MM-DD format.",
            },
            "days": {
                "type": "integer",
                "description": "Number of days to add (negative to go back in time).",
            },
        },
        "required": ["date_string", "days"],
    },
}

TOOLS: list[ToolParam] = [get_current_datetime_schema, add_days_to_date_schema]


# --- Tool Dispatcher ---

def run_tool(name: str, inputs: dict) -> str:
    if name == "get_current_datetime":
        return get_current_datetime(**inputs)
    if name == "add_days_to_date":
        return add_days_to_date(**inputs)
    raise ValueError(f"Unknown tool: {name}")


# --- Updated helper functions that handle both plain strings and multi-block content lists ---

def add_user_message(messages: list, content) -> None:
    """Accept either a plain string or a list of tool_result blocks."""
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages: list, content) -> None:
    """Accept either a plain string or the raw response.content block list."""
    messages.append({"role": "assistant", "content": content})


# --- Block inspector: shows the internal structure of a multi-block response ---

def inspect_response_blocks(response) -> None:
    print(f"\n  [response] stop_reason={response.stop_reason!r}, blocks={len(response.content)}")
    for i, block in enumerate(response.content):
        if block.type == "text":
            print(f"    block[{i}] TEXT     : {block.text!r}")
        elif block.type == "tool_use":
            print(f"    block[{i}] TOOL_USE : id={block.id!r}  name={block.name!r}  input={block.input}")
        else:
            print(f"    block[{i}] UNKNOWN  : {block}")


# --- Chat loop with explicit multi-block handling ---

def chat(messages: list) -> str:
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )

        inspect_response_blocks(response)

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text

        if response.stop_reason == "tool_use":
            # Preserve the entire content list (text + tool_use blocks) in history.
            # Passing only a string here would lose the tool_use block and break the next API call.
            add_assistant_message(messages, response.content)

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n  [executing] {block.name}({block.input})")
                    try:
                        result = run_tool(block.name, block.input)
                    except Exception as e:
                        result = f"Error: {e}"
                    print(f"  [result]    {result!r}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Send tool results back as a user turn; the list structure is required.
            add_user_message(messages, tool_results)


try:
    print("=== Handling Message Blocks Demo ===")
    print("This demo reveals the multi-block structure of tool-enabled responses.")
    print("Try: 'What is the exact time, formatted as HH:MM:SS?'")
    print("  or 'What day of the week will it be 30 days from today?'")
    print("Type 'exit' to quit.\n")

    messages: list = []

    while True:
        user_input = input("You : ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        add_user_message(messages, user_input)
        reply = chat(messages)
        add_assistant_message(messages, reply)

        print(f"\nClaude : {reply}\n")

except Exception as e:
    print(f"\nError: {e}")
