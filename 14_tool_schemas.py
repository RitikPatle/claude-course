# A tool schema is a JSON description that tells Claude the name of your function, what it does, and what arguments it expects.
# Claude reads the schema — not the Python source code — so the quality of your descriptions directly determines how well Claude decides when and how to call the tool.
# Each schema has three parts: name (must match your function), description (3-4 sentences on what it does, when to use it, and what it returns), and input_schema (a JSON Schema object describing every argument).
# Follow the naming convention of pairing each function with a matching schema variable: get_current_datetime and get_current_datetime_schema.
# You can use the ToolParam type from the Anthropic library for type safety, which catches schema mistakes before they reach the API.
# Writing schemas by hand is tedious; a common shortcut is pasting your function into Claude and asking it to generate the schema following Anthropic's tool-use documentation.


import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import ToolParam

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = 1024

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


# --- Tool Schemas (paired with their functions by naming convention) ---
# ToolParam adds type safety: the Anthropic SDK validates the shape before sending.

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
                    "Examples: '%Y-%m-%d' for date only, '%H:%M' for hour and minute, "
                    "'%A %B %d %Y' for a human-readable form like 'Wednesday June 12 2024'."
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
        "Use this tool when the user asks what date it will be in N days, or what day a future or past date falls on. "
        "Pass a negative number of days to go back in time. "
        "Returns a string in the format 'YYYY-MM-DD (DayName)', for example '2024-07-22 (Monday)'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "date_string": {
                "type": "string",
                "description": "The starting date in YYYY-MM-DD format, e.g. '2024-06-12'.",
            },
            "days": {
                "type": "integer",
                "description": "Number of days to add. Use a negative value to subtract days.",
            },
        },
        "required": ["date_string", "days"],
    },
}

# Collect all schemas into one list passed to the API
TOOLS: list[ToolParam] = [
    get_current_datetime_schema,
    add_days_to_date_schema,
]


# --- Tool Dispatcher ---

def run_tool(name: str, inputs: dict) -> str:
    if name == "get_current_datetime":
        return get_current_datetime(**inputs)
    if name == "add_days_to_date":
        return add_days_to_date(**inputs)
    raise ValueError(f"Unknown tool: {name}")


# --- Chat Loop (handles multi-step tool calling) ---

def chat(messages: list) -> str:
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [tool] {block.name}({block.input})")
                    try:
                        result = run_tool(block.name, block.input)
                    except Exception as e:
                        result = f"Error: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})


try:
    print("=== Tool Schemas Demo ===")
    print("Schemas in use: get_current_datetime_schema, add_days_to_date_schema")
    print("Try: 'What is today's date?'  or  'What day is 100 days from now?'")
    print("Type 'exit' to quit.\n")

    messages = []

    while True:
        user_input = input("You : ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})

        print(f"\nClaude : {reply}\n")

except Exception as e:
    print(f"\nError: {e}")
