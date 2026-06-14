# Terminal-based MCP Inspector — a Python replacement for "mcp dev" when Node.js is
# not installed. Connects to any MCP server via stdio, lists all its tools with their
# parameter schemas, lets you pick a tool interactively, prompts for each required
# parameter, calls the tool, and prints the result — the same workflow as the browser
# inspector but entirely in the terminal with no npm/npx dependency.

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ─────────────────────────────────────────────
# Pretty print helpers
# ─────────────────────────────────────────────

def divider(char="=", width=60):
    print(char * width)

def print_tool(tool, index: int):
    print(f"\n  [{index}] {tool.name}")
    if tool.description:
        print(f"      {tool.description}")
    schema = tool.inputSchema or {}
    props  = schema.get("properties", {})
    req    = schema.get("required", [])
    if props:
        print("      Parameters:")
        for pname, pdef in props.items():
            marker = " (required)" if pname in req else " (optional)"
            desc   = pdef.get("description", "no description")
            ptype  = pdef.get("type", "any")
            print(f"        - {pname}: {ptype}{marker} — {desc}")


def collect_inputs(tool) -> dict:
    """Interactively prompt the user for each tool parameter."""
    schema = tool.inputSchema or {}
    props  = schema.get("properties", {})
    req    = schema.get("required", [])

    if not props:
        print("  (This tool takes no parameters)")
        return {}

    print("\n  Enter parameter values (press Enter to skip optional ones):")
    inputs = {}
    for pname, pdef in props.items():
        is_req = pname in req
        ptype  = pdef.get("type", "string")
        desc   = pdef.get("description", "")
        marker = " [required]" if is_req else " [optional, Enter to skip]"
        prompt = f"    {pname}{marker}: "

        while True:
            value = input(prompt).strip()
            if not value:
                if is_req:
                    print(f"    '{pname}' is required — please enter a value.")
                    continue
                break                    # skip optional
            # Cast to correct type
            if ptype == "integer":
                try:
                    value = int(value)
                except ValueError:
                    print(f"    Must be an integer. Try again.")
                    continue
            elif ptype == "number":
                try:
                    value = float(value)
                except ValueError:
                    print(f"    Must be a number. Try again.")
                    continue
            elif ptype == "boolean":
                value = value.lower() in ("true", "1", "yes", "y")
            inputs[pname] = value
            break

    return inputs


# ─────────────────────────────────────────────
# Main inspector loop
# ─────────────────────────────────────────────

async def run_inspector(server_file: str):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_file],
    )

    print()
    divider()
    print("  MCP Terminal Inspector")
    print(f"  Server: {server_file}")
    divider()
    print("Connecting to MCP server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected!\n")

            # ── List all tools once ─────────────────────────────────────────
            tools_result = await session.list_tools()
            tools        = tools_result.tools

            while True:
                divider("-", 60)
                print(f"Available tools ({len(tools)}):")
                for i, t in enumerate(tools, 1):
                    print_tool(t, i)

                print(f"\n  [0] Refresh tool list")
                print(f"  [q] Quit\n")

                choice = input("Select a tool number: ").strip().lower()

                if choice == "q":
                    print("Goodbye!")
                    break

                if choice == "0":
                    tools_result = await session.list_tools()
                    tools        = tools_result.tools
                    print("Tool list refreshed.")
                    continue

                try:
                    idx  = int(choice) - 1
                    tool = tools[idx]
                except (ValueError, IndexError):
                    print("Invalid choice — enter a number from the list.")
                    continue

                # ── Collect inputs and call the tool ───────────────────────
                divider("-", 60)
                print(f"Tool: {tool.name}")
                if tool.description:
                    print(f"Description: {tool.description}")

                inputs = collect_inputs(tool)

                print(f"\nRunning '{tool.name}'...")
                divider("-", 60)

                try:
                    result = await session.call_tool(tool.name, inputs)

                    # Print each content block in the result
                    if not result.content:
                        print("(Tool returned no output)")
                    for item in result.content:
                        text = getattr(item, "text", None)
                        if text is not None:
                            print(text)
                        else:
                            print(repr(item))

                    if result.isError:
                        print("\n[Tool reported an error]")

                except Exception as e:
                    print(f"ERROR calling tool: {e}")

                print()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to our MCP server from file 36
        server_file = str(Path(__file__).parent / "36_mcp_inspector.py")
        print(f"No server file specified — using default: {server_file}")
        print("Usage: python 36b_mcp_terminal_inspector.py <mcp_server.py>")
    else:
        server_file = sys.argv[1]

    if not Path(server_file).exists():
        print(f"Error: server file not found: {server_file}")
        sys.exit(1)

    asyncio.run(run_inspector(server_file))
