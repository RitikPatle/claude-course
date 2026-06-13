# The text editor tool is Claude's one built-in tool — you don't write a JSON schema for it
# because Claude already knows the full specification; you only supply a small stub that tells
# the API which schema version to activate, and Claude expands it into the complete tool automatically.
# The schema version you use depends on the model: claude-3-7-sonnet uses text_editor_20250124,
# claude-3-5-sonnet/haiku uses text_editor_20241022, and newer claude-4.x models use text_editor_20250429.
# Even though Claude provides the schema, YOU still need to implement the actual file operations —
# Claude decides what to do (view, create, replace, insert, undo) but your code must carry it out.
# The tool name is always "str_replace_editor" and block.input["command"] tells you which operation
# Claude wants: "view", "create", "str_replace", "insert", or "undo_edit".
# This pattern lets you build your own AI-powered code editor inside any Python application,
# with full control over how Claude interacts with the file system.

# Full example: this program picks the right schema stub for the active model, implements all five
# file operations (view/create/str_replace/insert/undo_edit) with a per-file undo stack, then
# runs a multi-turn conversation loop so you can ask Claude to read, modify, and create files —
# demonstrating that Claude acts as a software engineer right out of the box once the tool is wired up.


import os
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import Message, ToolParam

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
# MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
MAX_TOKENS = 2000 # 100 is too low

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

client = Anthropic(api_key=API_KEY)


# ─────────────────────────────────────────────
# Schema stub — version depends on the model.
# Claude expands this small stub into the full text editor specification automatically.
# Check https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool
# for the complete version list as new models are released.
# ─────────────────────────────────────────────

def get_text_edit_schema(model: str) -> dict:
    if "claude-3-7" in model:
        return {"type": "text_editor_20250124", "name": "str_replace_editor"}
    elif "claude-3-5" in model:
        return {"type": "text_editor_20241022", "name": "str_replace_editor"}
    else:
        # claude-4.x and newer models (haiku-4-5, sonnet-4-x, etc.)
        return {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}

SCHEMA = get_text_edit_schema(MODEL)


# ─────────────────────────────────────────────
# Undo stack — keeps one snapshot per file so undo_edit can restore the previous state.
# ─────────────────────────────────────────────

_undo_stack: dict[str, str] = {}   # path → previous file contents

# Base directory — all paths are resolved relative to here so Claude can't
# accidentally write to the drive root when it sends paths like "/hello.py".
BASE_DIR = Path(__file__).parent.resolve()


def _resolve_path(path: str) -> Path:
    """Strip leading slashes/dots and resolve relative to BASE_DIR."""
    cleaned = path.lstrip("/\\").lstrip("./").lstrip("/\\")
    return BASE_DIR / cleaned if cleaned else BASE_DIR


def _save_undo_snapshot(path: str) -> None:
    p = _resolve_path(path)
    if p.exists():
        _undo_stack[str(p)] = p.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# File operation handlers
# Each function matches one value of block.input["command"].
# ─────────────────────────────────────────────

def handle_view(path: str, view_range: list[int] | None = None) -> str:
    p = _resolve_path(path)

    if p.is_dir():
        items = sorted(p.iterdir())
        lines = [f"{'[DIR] ' if i.is_dir() else '[FILE]'} {i.name}" for i in items]
        return f"Directory: {p}\n" + "\n".join(lines) if lines else f"Empty directory: {p}"

    if not p.exists():
        return f"Error: file not found: {p}"

    content = p.read_text(encoding="utf-8")
    all_lines = content.splitlines()

    if view_range:
        start, end = view_range[0] - 1, view_range[1]
        selected = all_lines[start:end]
        numbered = [f"{start + i + 1}: {line}" for i, line in enumerate(selected)]
        return "\n".join(numbered)

    numbered = [f"{i + 1}: {line}" for i, line in enumerate(all_lines)]
    return "\n".join(numbered)


def handle_create(path: str, file_text: str) -> str:
    p = _resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _save_undo_snapshot(path)
    p.write_text(file_text, encoding="utf-8")
    return f"Created: {p}"


def handle_str_replace(path: str, old_str: str, new_str: str) -> str:
    p = _resolve_path(path)
    if not p.exists():
        return f"Error: file not found: {p}"

    content = p.read_text(encoding="utf-8")
    if old_str not in content:
        return f"Error: old_str not found in {p}"

    _save_undo_snapshot(path)
    p.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
    return f"Replaced text in: {p}"


def handle_insert(path: str, insert_line: int, new_str: str) -> str:
    p = _resolve_path(path)
    if not p.exists():
        return f"Error: file not found: {p}"

    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    _save_undo_snapshot(path)
    insert_text = new_str if new_str.endswith("\n") else new_str + "\n"
    lines.insert(insert_line, insert_text)
    p.write_text("".join(lines), encoding="utf-8")
    return f"Inserted at line {insert_line} in: {p}"


def handle_undo_edit(path: str) -> str:
    p = _resolve_path(path)
    key = str(p)
    if key not in _undo_stack:
        return f"Error: no undo snapshot for: {p}"
    p.write_text(_undo_stack.pop(key), encoding="utf-8")
    return f"Undo successful: {p}"


# ─────────────────────────────────────────────
# Tool dispatcher — routes block.input["command"] to the right handler
# ─────────────────────────────────────────────

def run_text_editor_tool(tool_input: dict) -> str:
    command = tool_input.get("command")

    if command == "view":
        return handle_view(tool_input["path"], tool_input.get("view_range"))
    elif command == "create":
        return handle_create(tool_input["path"], tool_input.get("file_text", ""))
    elif command == "str_replace":
        return handle_str_replace(tool_input["path"], tool_input["old_str"], tool_input["new_str"])
    elif command == "insert":
        return handle_insert(tool_input["path"], tool_input["insert_line"], tool_input["new_str"])
    elif command == "undo_edit":
        return handle_undo_edit(tool_input["path"])
    else:
        return f"Error: unknown command '{command}'"


# ─────────────────────────────────────────────
# Conversation helpers
# ─────────────────────────────────────────────

def add_user_message(messages: list, content) -> None:
    c = content.content if isinstance(content, Message) else content
    messages.append({"role": "user", "content": c})


def add_assistant_message(messages: list, content) -> None:
    c = content.content if isinstance(content, Message) else content
    messages.append({"role": "assistant", "content": c})


def text_from_message(message: Message) -> str:
    return "\n".join(b.text for b in message.content if b.type == "text")


def chat(messages: list) -> Message:
    return client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=[SCHEMA],
        messages=messages,
    )


# ─────────────────────────────────────────────
# Multi-turn loop — same pattern as files 17-19
# ─────────────────────────────────────────────

def run_conversation(messages: list) -> Message:
    while True:
        response = chat(messages)
        add_assistant_message(messages, response)

        intermediate = text_from_message(response)
        if intermediate:
            print(f"\n  Claude: {intermediate}")

        if response.stop_reason != "tool_use":
            return response

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\n  [editor] command={block.input.get('command')}  path={block.input.get('path')}")
            result = run_text_editor_tool(block.input)
            print(f"  [result] {result[:120]}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        add_user_message(messages, tool_results)


try:
    print("=== Text Editor Tool Demo ===")
    print(f"Model  : {MODEL}")
    print(f"Schema : {SCHEMA['type']}")
    print()
    print("Claude can now read, create, and edit files in this directory.")
    print()
    print("Try:")
    print("  'Open ./main.py and summarise its contents'")
    print("  'Create a ./hello.py file that prints Hello World'")
    print("  'Open ./hello.py and add a goodbye() function'")
    print("  'List the files in the current directory'")
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
        final = run_conversation(messages)
        reply = text_from_message(final)
        add_assistant_message(messages, reply)

        print(f"\nClaude : {reply}\n")

except Exception as e:
    print(f"\nError: {e}")
