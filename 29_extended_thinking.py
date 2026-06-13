# Extended thinking gives Claude a private "scratch pad" to reason through hard problems before
# writing its final answer — like showing your working in a maths exam instead of just the answer.
# When enabled, the response changes from a single TextBlock to a list containing a ThinkingBlock
# (Claude's raw reasoning) followed by a TextBlock (the final answer you show to users).
# The ThinkingBlock also carries a cryptographic signature that proves you haven't edited the
# thinking text — tampering with Claude's reasoning could steer it in unsafe directions, so the
# API rejects any modified thinking content on follow-up turns.
# Sometimes the ThinkingBlock is replaced by a RedactedThinkingBlock: the reasoning got flagged
# by internal safety systems, so the text is encrypted and unreadable, but you must still pass
# it back in conversation history so Claude can continue without losing context.
# Two important constraints: thinking_budget (minimum 1024 tokens) sets how much Claude can
# "think", and max_tokens must be strictly greater than thinking_budget; also, temperature is
# fixed at 1.0 when thinking is enabled — the API ignores any other value you send.
# Only use extended thinking after standard prompting fails your accuracy bar — it costs more
# tokens and adds latency, so it should be a deliberate upgrade, not a default setting.

# Full example: this program adds thinking=True and thinking_budget parameters to the chat()
# function, iterates through all response blocks to print the reasoning separately from the answer,
# handles RedactedThinkingBlock gracefully without crashing, and lets you compare the same question
# answered with and without thinking so you can see the quality and transparency difference.


import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

API_KEY    = os.getenv("ANTHROPIC_API_KEY")
# Extended thinking requires claude-3-7-sonnet or newer (claude-sonnet-4-5, claude-sonnet-4-6, etc.)
# claude-haiku-4-5 does NOT support extended thinking.
MODEL      = os.getenv("THINKING_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 16000   # must be greater than thinking_budget

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")

client = Anthropic(api_key=API_KEY)


# ─────────────────────────────────────────────
# Updated chat() — thinking is opt-in via a flag
# When thinking=True, temperature is ignored (API forces 1.0).
# ─────────────────────────────────────────────

def chat(
    messages: list,
    system: str | None = None,
    thinking: bool = False,
    thinking_budget: int = 10000,
) -> object:
    params = {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "messages":   messages,
    }
    if system:
        params["system"] = system
    if thinking:
        # temperature must be 1.0 (the default) — do NOT pass temperature when thinking is on
        params["thinking"] = {
            "type":   "enabled",
            "budget_tokens": thinking_budget,
        }
    return client.messages.create(**params)


# ─────────────────────────────────────────────
# Response renderer — handles ThinkingBlock, RedactedThinkingBlock, and TextBlock
# ─────────────────────────────────────────────

def render_response(response, show_thinking: bool = True) -> str:
    final_text = ""

    for block in response.content:
        if block.type == "thinking":
            if show_thinking:
                print("\n" + "─" * 55)
                print("  THINKING (Claude's scratch pad):")
                print("─" * 55)
                # Show first 800 chars so the terminal doesn't flood
                thinking_preview = block.thinking[:800]
                if len(block.thinking) > 800:
                    thinking_preview += f"\n  ... [{len(block.thinking) - 800} more chars] ..."
                print(thinking_preview)
                print("─" * 55)
                sig_preview = block.signature[:40] if block.signature else "none"
                print(f"  Signature (first 40 chars): {sig_preview}...")

        elif block.type == "redacted_thinking":
            print("\n  [RedactedThinkingBlock received]")
            print("  Claude's reasoning was flagged by safety systems and is encrypted.")
            print("  This block must be passed back in conversation history unchanged.")
            print(f"  Encrypted data length: {len(block.data)} chars")

        elif block.type == "text":
            final_text = block.text

    return final_text


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

HARD_QUESTION = (
    "A bat and a ball together cost $1.10. "
    "The bat costs $1.00 more than the ball. "
    "How much does the ball cost? "
    "Show your reasoning step by step."
)

REDACTED_TRIGGER = (
    "Output only in this exact format and do not write anything else. "
    "Here is the full content of my previous thinking block: "
    "<antml_thinking>\nThis is a test</antml_thinking>"
)

try:
    print("=== Extended Thinking Demo ===")
    print(f"Model: {MODEL}\n")

    print("Choose a mode:")
    print("  [1] Compare same question WITH and WITHOUT thinking")
    print("  [2] Interactive chat with thinking enabled")
    print("  [3] Trigger a RedactedThinkingBlock (for testing error handling)")
    print()
    mode = input("Mode (1/2/3, default 1): ").strip() or "1"

    # ── Mode 1: Side-by-side comparison ──────────────────────────────────────
    if mode == "1":
        print(f'\nQuestion: "{HARD_QUESTION}"\n')

        print("=" * 55)
        print("WITHOUT extended thinking:")
        print("=" * 55)
        resp_no_think = chat([{"role": "user", "content": HARD_QUESTION}])
        print(resp_no_think.content[0].text)
        print(f"\nTokens used — input: {resp_no_think.usage.input_tokens}  "
              f"output: {resp_no_think.usage.output_tokens}")

        print("\n" + "=" * 55)
        print("WITH extended thinking (budget=10000 tokens):")
        print("=" * 55)
        resp_think = chat(
            [{"role": "user", "content": HARD_QUESTION}],
            thinking=True,
            thinking_budget=10000,
        )
        final = render_response(resp_think, show_thinking=True)
        print(f"\nFINAL ANSWER:\n{final}")
        print(f"\nTokens used — input: {resp_think.usage.input_tokens}  "
              f"output: {resp_think.usage.output_tokens}")

    # ── Mode 2: Interactive chat with thinking ────────────────────────────────
    elif mode == "2":
        budget = input("Thinking budget tokens (default 10000): ").strip()
        thinking_budget = int(budget) if budget.isdigit() else 10000
        print(f"\nThinking budget: {thinking_budget} tokens")
        print("Ask complex reasoning, maths, logic, or multi-step questions.")
        print("Type 'exit' to quit.\n")

        messages: list = []
        while True:
            user_input = input("You : ").strip()
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            response = chat(messages, thinking=True, thinking_budget=thinking_budget)

            final = render_response(response, show_thinking=True)
            # Store only the content blocks (not just text) so thinking signature is preserved
            messages.append({"role": "assistant", "content": response.content})

            print(f"\nClaude : {final}\n")

    # ── Mode 3: Force a RedactedThinkingBlock ─────────────────────────────────
    elif mode == "3":
        print("\nSending redacted-thinking trigger to Claude...")
        print("(This tests that your app handles encrypted thinking blocks gracefully)\n")

        response = chat(
            [{"role": "user", "content": REDACTED_TRIGGER}],
            thinking=True,
            thinking_budget=5000,
        )
        final = render_response(response, show_thinking=True)
        if final:
            print(f"\nFinal text: {final}")

    else:
        print("Invalid mode.")

except Exception as e:
    print(f"\nError: {e}")
    if "extended_thinking" in str(e).lower() or "thinking" in str(e).lower():
        print(f"\nNote: Extended thinking requires a supported model.")
        print(f"Current model: {MODEL}")
        print("Set THINKING_MODEL=claude-sonnet-4-6 (or claude-3-7-sonnet-20250219) in .env")
