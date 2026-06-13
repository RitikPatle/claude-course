# When you combine tool use with streaming, Claude sends back a new event type called InputJsonEvent
# that lets you watch tool arguments being built in real time — each event carries partial_json
# (the latest chunk) and snapshot (everything accumulated so far).
# By default the API buffers chunks and validates each complete top-level key-value pair before
# sending anything — so you see delays followed by bursts, not a smooth character-by-character flow.
# Fine-grained tool calling disables that validation layer: you get chunks as fast as Claude
# generates them, but the tradeoff is that the JSON may be temporarily invalid, so your code must
# wrap every json.loads(snapshot) in a try/except json.JSONDecodeError.
# Enable it by passing betas=["fine-grained-tool-streaming-2025-05-14"] to the API call.
# Use fine-grained when you want real-time progress bars or need to start acting on partial results
# immediately; use the default when you need guaranteed-valid JSON and can tolerate the buffering.

# Full example: this program defines a save_article tool with a multi-field schema (abstract + meta),
# streams the tool argument generation in both default and fine-grained modes, prints every
# partial_json chunk so you can see the buffering vs instant-delivery difference side by side,
# and gracefully handles any invalid JSON that fine-grained mode may produce mid-stream.


import json
import os
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


# ─────────────────────────────────────────────
# Tool Function & Schema
# Uses a rich multi-field schema so the buffering behaviour is clearly visible.
# ─────────────────────────────────────────────

def save_article(abstract: str, meta: dict) -> str:
    print(f"\n  [save_article called]")
    print(f"    abstract : {abstract[:80]}...")
    print(f"    meta     : {meta}")
    return "Article saved successfully."


save_article_schema: ToolParam = {
    "name": "save_article",
    "description": (
        "Saves a summarised article with an abstract and metadata. "
        "Always call this tool when the user asks you to summarise or save an article."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "abstract": {
                "type": "string",
                "description": "A 2-3 sentence summary of the article.",
            },
            "meta": {
                "type": "object",
                "description": "Metadata about the article.",
                "properties": {
                    "word_count": {
                        "type": "integer",
                        "description": "Approximate word count of the original article.",
                    },
                    "review": {
                        "type": "string",
                        "description": "One sentence critical review of the article.",
                    },
                },
                "required": ["word_count", "review"],
            },
        },
        "required": ["abstract", "meta"],
    },
}


# ─────────────────────────────────────────────
# Default streaming (with API-side JSON validation)
# Chunks arrive in bursts — the API buffers until each top-level key-value pair is complete.
# ─────────────────────────────────────────────

def stream_with_tools_default(messages: list) -> None:
    print("\n--- DEFAULT STREAMING (API validates before sending) ---")
    print("You will see bursts, not continuous character output.\n")

    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=[save_article_schema],
        messages=messages,
    ) as stream:
        for event in stream:
            # Regular text delta
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                print(event.delta.text, end="", flush=True)

            # Tool argument chunk
            if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                chunk = event.delta.partial_json
                print(chunk, end="", flush=True)

        # Get the final assembled message to check for tool use
        final = stream.get_final_message()

    print("\n")
    _execute_tools_if_needed(final)


# ─────────────────────────────────────────────
# Fine-grained streaming (API-side validation disabled)
# Chunks arrive immediately — but JSON may be incomplete/invalid mid-stream.
# ─────────────────────────────────────────────

def stream_with_tools_fine_grained(messages: list) -> None:
    print("\n--- FINE-GRAINED STREAMING (no buffering, chunks arrive instantly) ---")
    print("You will see smooth character output. Invalid JSON handled with try/except.\n")

    accumulated_json = ""

    try:
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=[save_article_schema],
            messages=messages,
            betas=["fine-grained-tool-streaming-2025-05-14"],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                    print(event.delta.text, end="", flush=True)

                if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                    chunk = event.delta.partial_json
                    accumulated_json += chunk
                    print(chunk, end="", flush=True)

                    # Fine-grained mode: JSON may be invalid mid-stream — always guard with try/except
                    try:
                        current_args = json.loads(accumulated_json)
                        # If we get here, the JSON is valid at this moment
                    except json.JSONDecodeError:
                        pass  # Expected during streaming — keep accumulating

            final = stream.get_final_message()

        print("\n")
        _execute_tools_if_needed(final)

    except Exception as e:
        # Fine-grained beta may not be enabled on all accounts/models
        print(f"\n  [fine-grained not available: {e}]")
        print("  Falling back to default streaming.\n")
        stream_with_tools_default(messages)


# ─────────────────────────────────────────────
# Tool executor — runs any tool_use blocks in the final message
# ─────────────────────────────────────────────

def _execute_tools_if_needed(final_message) -> None:
    for block in final_message.content:
        if block.type == "tool_use":
            if block.name == "save_article":
                result = save_article(**block.input)
                print(f"\n  [tool result] {result}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

SAMPLE_ARTICLE = (
    "Summarise and save this article: "
    "Researchers at MIT have developed a new neural network architecture called QuanNet that achieves "
    "state-of-the-art performance on natural language benchmarks while using 40% fewer parameters than "
    "existing models. The paper introduces a novel quantisation technique applied during training rather "
    "than post-training, preserving accuracy while dramatically reducing model size. Early experiments "
    "show QuanNet running inference 3x faster on consumer hardware with no measurable drop in quality. "
    "The team plans to open-source the weights and training code in Q3 2025."
)

try:
    print("=== Fine-Grained Tool Calling Demo ===")
    print()
    print("This demo streams a save_article tool call in two modes so you can compare:")
    print("  1. Default   — API buffers & validates before sending (bursts of text)")
    print("  2. Fine-grained — chunks arrive immediately (may produce invalid JSON mid-stream)")
    print()

    choice = input("Mode? [1] Default  [2] Fine-grained  [3] Both : ").strip()

    messages = [{"role": "user", "content": SAMPLE_ARTICLE}]

    if choice == "1":
        stream_with_tools_default(messages)
    elif choice == "2":
        stream_with_tools_fine_grained(messages)
    else:
        stream_with_tools_default(messages)
        messages = [{"role": "user", "content": SAMPLE_ARTICLE}]
        stream_with_tools_fine_grained(messages)

except Exception as e:
    print(f"\nError: {e}")
