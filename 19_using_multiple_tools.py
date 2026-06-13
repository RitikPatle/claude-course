# Files 17 and 18 already taught the multi-turn loop and the run_tools/run_tool pattern.
# This file introduces one new concept: an "action tool" — a tool that doesn't just return
# computed data but actually DOES something, like storing a reminder in a list.
# All the date tools so far were "query tools" (ask a question, get an answer back).
# set_reminder is a "write tool" — it changes state in your application.
# The good news is the tool infrastructure doesn't care: both types plug in exactly the same way —
# add the function, write the schema, add it to the tools list, and add one elif to run_tool().
# This file also shows add_duration_to_datetime, which accepts a human-friendly duration string
# like "177 days" or "3 months" instead of a plain integer, making it easier for users to ask
# natural questions like "Set a reminder 177 days after Jan 1st, 2050."

# Full example: this program registers three tools (get_current_datetime, add_duration_to_datetime,
# set_reminder), demonstrates the simple four-step pattern for adding any new tool, and lets you
# ask something like "Set a reminder for my doctor's appointment 177 days after Jan 1st, 2050" —
# showing Claude chain two tool calls before finally calling the action tool to store the reminder.


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


# ─────────────────────────────────────────────
# Tool Functions
# ─────────────────────────────────────────────

def get_current_datetime(date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now().strftime(date_format)


def add_duration_to_datetime(date_string: str, duration: str) -> str:
    """Accepts a duration string like '177 days', '3 weeks', '2 months'."""
    base = datetime.strptime(date_string, "%Y-%m-%d")
    duration = duration.strip().lower()

    if "day" in duration:
        days = int(duration.split()[0])
        result = base + timedelta(days=days)
    elif "week" in duration:
        weeks = int(duration.split()[0])
        result = base + timedelta(weeks=weeks)
    elif "month" in duration:
        months = int(duration.split()[0])
        # approximate: 1 month = 30 days
        result = base + timedelta(days=months * 30)
    else:
        raise ValueError(f"Unrecognised duration format: {duration!r}. Use '177 days', '3 weeks', or '2 months'.")

    return result.strftime("%Y-%m-%d (%A)")


# In-memory store — a real app would write to a database or calendar API
REMINDERS: list[dict] = []

def set_reminder(title: str, date_string: str, notes: str = "") -> str:
    """Action tool: stores a reminder. Returns a confirmation string."""
    reminder = {"title": title, "date": date_string, "notes": notes}
    REMINDERS.append(reminder)
    return f"Reminder set: '{title}' on {date_string}" + (f" — {notes}" if notes else "")


# ─────────────────────────────────────────────
# Tool Schemas
# ─────────────────────────────────────────────

get_current_datetime_schema: ToolParam = {
    "name": "get_current_datetime",
    "description": "Returns the current date and time. Use when the user asks about today's date or the current time.",
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": "Python strftime format string. Default: '%Y-%m-%d %H:%M:%S'.",
                "default": "%Y-%m-%d %H:%M:%S",
            }
        },
        "required": [],
    },
}

add_duration_to_datetime_schema: ToolParam = {
    "name": "add_duration_to_datetime",
    "description": (
        "Adds a duration to a given date and returns the resulting date with its day name. "
        "Use when the user asks for a date a certain number of days, weeks, or months from a given date. "
        "Returns a string like '2050-06-27 (Thursday)'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "date_string": {
                "type": "string",
                "description": "Starting date in YYYY-MM-DD format, e.g. '2050-01-01'.",
            },
            "duration": {
                "type": "string",
                "description": "Duration string like '177 days', '3 weeks', or '2 months'.",
            },
        },
        "required": ["date_string", "duration"],
    },
}

set_reminder_schema: ToolParam = {
    "name": "set_reminder",
    "description": (
        "Stores a reminder with a title and date. Use this when the user asks to set, create, or schedule a reminder. "
        "Always call add_duration_to_datetime first if the date needs to be calculated. "
        "Returns a confirmation string."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short label for the reminder, e.g. 'Doctor appointment'.",
            },
            "date_string": {
                "type": "string",
                "description": "Date for the reminder in YYYY-MM-DD format.",
            },
            "notes": {
                "type": "string",
                "description": "Optional extra details about the reminder.",
            },
        },
        "required": ["title", "date_string"],
    },
}

# Step 3 of the pattern: collect all schemas into one list
TOOLS: list[ToolParam] = [
    get_current_datetime_schema,
    add_duration_to_datetime_schema,
    set_reminder_schema,
]


# ─────────────────────────────────────────────
# Helpers (same pattern as files 17 & 18)
# ─────────────────────────────────────────────

def add_user_message(messages: list, message) -> None:
    content = message.content if isinstance(message, Message) else message
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages: list, message) -> None:
    content = message.content if isinstance(message, Message) else message
    messages.append({"role": "assistant", "content": content})


def chat(messages: list, tools: list[ToolParam] | None = None) -> Message:
    params = {"model": MODEL, "max_tokens": MAX_TOKENS, "messages": messages}
    if tools:
        params["tools"] = tools
    return client.messages.create(**params)


def text_from_message(message: Message) -> str:
    return "\n".join(block.text for block in message.content if block.type == "text")


# ─────────────────────────────────────────────
# Tool Router
# Step 4 of the pattern: add one elif per new tool — nothing else changes
# ─────────────────────────────────────────────

def run_tool(tool_name: str, tool_input: dict):
    if tool_name == "get_current_datetime":
        return get_current_datetime(**tool_input)
    elif tool_name == "add_duration_to_datetime":
        return add_duration_to_datetime(**tool_input)
    elif tool_name == "set_reminder":
        return set_reminder(**tool_input)
    raise ValueError(f"Unknown tool: {tool_name}")


def run_tools(message: Message) -> list[dict]:
    tool_requests = [b for b in message.content if b.type == "tool_use"]
    results = []
    for req in tool_requests:
        print(f"    [tool] {req.name}({req.input})")
        try:
            output = run_tool(req.name, req.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": req.id,
                "content": json.dumps(output),
                "is_error": False,
            })
            print(f"    [result] {json.dumps(output)}")
        except Exception as e:
            results.append({
                "type": "tool_result",
                "tool_use_id": req.id,
                "content": f"Error: {e}",
                "is_error": True,
            })
            print(f"    [error] {e}")
    return results


# ─────────────────────────────────────────────
# Conversation loop (same pattern as file 18)
# ─────────────────────────────────────────────

def run_conversation(messages: list) -> Message:
    while True:
        response = chat(messages, tools=TOOLS)
        add_assistant_message(messages, response)

        intermediate = text_from_message(response)
        if intermediate:
            print(f"\n  Claude: {intermediate}")

        if response.stop_reason != "tool_use":
            return response

        tool_result_blocks = run_tools(response)
        add_user_message(messages, tool_result_blocks)


try:
    print("=== Using Multiple Tools Demo ===")
    print("Tools available: get_current_datetime, add_duration_to_datetime, set_reminder")
    print()
    print("Try: 'Set a reminder for my doctor appointment 177 days after Jan 1st, 2050'")
    print("  or 'Remind me about the team meeting 3 weeks from today'")
    print("  or 'What day is 50 days from today?'")
    print("\nType 'reminders' to see stored reminders, 'exit' to quit.\n")

    messages: list = []

    while True:
        user_input = input("You : ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input.lower() == "reminders":
            if REMINDERS:
                print("\nStored reminders:")
                for r in REMINDERS:
                    print(f"  • {r['date']}  {r['title']}" + (f"  ({r['notes']})" if r["notes"] else ""))
            else:
                print("  No reminders set yet.")
            print()
            continue
        if not user_input:
            continue

        add_user_message(messages, user_input)
        final_response = run_conversation(messages)
        reply = text_from_message(final_response)
        add_assistant_message(messages, reply)

        print(f"\nClaude : {reply}\n")

except Exception as e:
    print(f"\nError: {e}")
