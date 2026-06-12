# Tool functions are plain Python functions that Claude can call during a conversation when it needs real-time data or the ability to perform actions.
# For example, if a user asks "what day will it be in 10 days?", Claude cannot answer from memory because it has no access to today's date.
# Instead, you define a Python function, describe it to Claude using a JSON schema, and Claude decides on its own when to call it.
# When Claude calls a tool, your code executes the function, returns the result, and Claude uses that result to continue the conversation.
# Best practices for tool functions: use descriptive names, validate inputs, and raise clear errors so Claude can retry with corrected values.
# This pattern is the foundation for building AI assistants that interact with APIs, databases, file systems, calendars, and any external service.


import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from anthropic import Anthropic

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


# --- Tool Schemas (what Claude reads to understand the functions) ---

TOOLS = [
    {
        "name": "get_current_datetime",
        "description": "Returns the current date and time formatted as requested. Use this whenever the user asks about the current time or date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_format": {
                    "type": "string",
                    "description": "Python strftime format string. Default: '%Y-%m-%d %H:%M:%S'",
                }
            },
            "required": [],
        },
    },
    {
        "name": "add_days_to_date",
        "description": "Adds a number of days to a given date and returns the resulting date with the day name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_string": {
                    "type": "string",
                    "description": "The starting date in YYYY-MM-DD format.",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to add (can be negative to go back in time).",
                },
            },
            "required": ["date_string", "days"],
        },
    },
]


def run_tool(name: str, inputs: dict) -> str:
    if name == "get_current_datetime":
        return get_current_datetime(**inputs)
    if name == "add_days_to_date":
        return add_days_to_date(**inputs)
    raise ValueError(f"Unknown tool: {name}")


def chat(messages: list) -> str:
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )

        # Claude finished with a normal text reply
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        # Claude wants to call one or more tools
        if response.stop_reason == "tool_use":
            # Add Claude's response (which contains the tool_use blocks) to history
            messages.append({"role": "assistant", "content": response.content})

            # Execute every tool Claude requested and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [tool call] {block.name}({block.input})")
                    try:
                        result = run_tool(block.name, block.input)
                    except Exception as e:
                        result = f"Error: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Return all tool results to Claude and loop again
            messages.append({"role": "user", "content": tool_results})


try:
    print("=== Tool Functions Demo ===")
    print("Claude has access to: get_current_datetime, add_days_to_date")
    print("Try: 'What time is it?'  or  'What day is it 45 days from today?'")
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
