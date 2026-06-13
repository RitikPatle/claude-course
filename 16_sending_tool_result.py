# After Claude requests a tool, your job is to actually run that function and hand the result back.
# You get the arguments Claude chose from response.content[1].input — it's just a plain Python dictionary.
# Because your function expects keyword arguments (not a dict), you unpack it with **: get_current_datetime(**block.input).
# The result goes inside a "tool_result" block: set tool_use_id to match the ToolUse block's id, content to the
# string output, and is_error to True only when something went wrong — this tells Claude what actually happened.
# Claude can ask for several tools in one response (e.g. two separate math questions at once); each ToolUse block
# gets its own unique id, so you loop over all blocks, run each function, and build one tool_result per call.
# The follow-up request must include the full conversation history AND the tool schemas again — Claude needs the
# schemas to understand the tool references already in the history, even though it probably won't call them again.
# Once Claude gets all the tool results it asked for, it issues a final "end_turn" text reply to the user.

# Full example: this program shows the complete tool-result sending flow step by step — it asks Claude a question
# that needs the current time, prints exactly what Claude requested (tool name + inputs + id), runs the Python
# function, builds the tool_result block, sends it back alongside the full history and tool schema, then prints
# Claude's final natural-language answer so you can clearly see how each piece connects to the next.


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


# --- Tool Dispatcher: unpacks the input dict into keyword arguments ---

def run_tool(name: str, inputs: dict) -> tuple[str, bool]:
    """Returns (result_string, is_error)."""
    try:
        if name == "get_current_datetime":
            return get_current_datetime(**inputs), False
        if name == "add_days_to_date":
            return add_days_to_date(**inputs), False
        return f"Unknown tool: {name}", True
    except Exception as e:
        return str(e), True


# --- Tool result builder: one block per ToolUse block Claude sent ---

def build_tool_results(response) -> list[dict]:
    """
    Loop over every block in the response.
    For each tool_use block: run the function, wrap result in a tool_result block.
    Matching tool_use_id to block.id is what lets Claude pair each result with its request.
    """
    results = []
    for block in response.content:
        if block.type != "tool_use":
            continue

        print(f"\n  [tool requested]")
        print(f"    name  : {block.name}")
        print(f"    id    : {block.id}")
        print(f"    input : {block.input}")

        # ** unpacks the dict so the function receives named keyword arguments
        output, is_error = run_tool(block.name, block.input)

        print(f"    result: {output!r}  (is_error={is_error})")

        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,   # must match the id above exactly
            "content": output,
            "is_error": is_error,
        })

    return results


# --- Main chat loop ---

def chat(messages: list) -> str:
    while True:
        # Always include TOOLS — even on follow-up calls — so Claude understands history references
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )

        print(f"\n  [stop_reason: {response.stop_reason}]")

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text

        if response.stop_reason == "tool_use":
            # Step 1 — save Claude's full response (including tool_use blocks) to history
            messages.append({"role": "assistant", "content": response.content})

            # Step 2 — run every requested tool, build tool_result blocks
            tool_results = build_tool_results(response)

            # Step 3 — send all results back as a single user message
            print(f"\n  [sending {len(tool_results)} tool result(s) back to Claude]")
            messages.append({"role": "user", "content": tool_results})

            # Loop: Claude will now read the results and produce a final reply


try:
    print("=== Sending Tool Results Demo ===")
    print("Watch how tool inputs are extracted, functions are executed,")
    print("and results are matched back to Claude by tool_use_id.")
    print()
    print("Try: 'What is the exact time, formatted as HH:MM:SS?'")
    print("  or 'What day of the week is 45 days from today?'")
    print("  or 'What time is it and what day will it be in 10 days?'  (two tool calls at once!)")
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
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})

        print(f"\nClaude : {reply}\n")

except Exception as e:
    print(f"\nError: {e}")
