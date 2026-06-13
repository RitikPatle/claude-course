# The Files API lets you upload a file once and reference it by a file_id in any future
# message, eliminating the need to base64-encode data in every request — particularly
# useful when the same file is used across multiple turns or when files are too large
# to embed comfortably inline.
# Code Execution is a server-side tool: include {"type": "code_execution_20250522",
# "name": "code_execution"} in your tools list and Claude can write Python and run it
# in an isolated Docker container on Anthropic's servers — no internet access, no setup
# required from your side, and Claude can execute code multiple times per response,
# iterating until it reaches a complete answer.
# The two features pair naturally because the container has no network: the Files API is
# how you get data IN (via a "container_upload" block containing the file_id) and how
# you retrieve generated files OUT (download the file_id from CodeExecutionOutputBlock
# items inside code_execution_tool_result blocks).
# Each response can contain multiple interleaved block types — server_tool_use (the Python
# Claude wrote), code_execution_tool_result (stdout / stderr / generated file IDs from
# running it), and text (Claude's interpretation) — so you parse all three separately.

# Full example: this program generates a 200-row streaming-service churn dataset as a CSV,
# uploads it via the Files API, asks Claude to identify the major churn drivers and produce
# a summary plot, then walks through every block in the response — printing the code Claude
# wrote, the execution output, and saving any generated image files to generated_materials/.


import csv
import io
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

API_KEY    = os.getenv("ANTHROPIC_API_KEY")
MODEL      = os.getenv("THINKING_MODEL", "claude-sonnet-4-6")  # code execution needs a capable model
MAX_TOKENS = max(int(os.getenv("MAX_TOKENS", "1024")), 4096)   # needs headroom for code + analysis

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")

client = Anthropic(api_key=API_KEY)

GENERATED_DIR = Path(__file__).parent / "generated_materials"
GENERATED_DIR.mkdir(exist_ok=True)
CSV_PATH      = GENERATED_DIR / "streaming_churn.csv"

FILES_BETA = ["files-api-2025-04-14"]


# ─────────────────────────────────────────────
# Sample dataset — streaming service with churn flag
# ─────────────────────────────────────────────

def create_churn_csv(path: Path, n: int = 200) -> None:
    random.seed(42)
    rows = []

    for i in range(n):
        tier              = random.choice(["basic", "standard", "premium"])
        age               = random.randint(18, 65)
        hours_per_week    = round(random.uniform(1, 30), 1)
        months_subscribed = random.randint(1, 36)
        support_tickets   = random.randint(0, 8)
        num_profiles      = random.randint(1, 5)

        # Churn probability with realistic drivers
        churn_prob = 0.08
        if tier == "basic" and hours_per_week < 5:
            churn_prob += 0.40
        if support_tickets > 3:
            churn_prob += 0.30
        if months_subscribed < 3:
            churn_prob += 0.25
        if hours_per_week > 20:
            churn_prob -= 0.15
        if tier == "premium":
            churn_prob -= 0.10

        rows.append({
            "user_id":           f"U{i+1:04d}",
            "tier":              tier,
            "age":               age,
            "hours_per_week":    hours_per_week,
            "months_subscribed": months_subscribed,
            "support_tickets":   support_tickets,
            "num_profiles":      num_profiles,
            "churned":           int(random.random() < max(0, min(1, churn_prob))),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    churn_count = sum(r["churned"] for r in rows)
    print(f"  CSV created: {path.name}  ({n} rows, {churn_count} churned = {churn_count/n:.0%})")


# ─────────────────────────────────────────────
# Files API helpers
# ─────────────────────────────────────────────

def upload_file(path: Path) -> str:
    """Upload a local file and return its file_id."""
    with open(path, "rb") as f:
        response = client.beta.files.upload(
            file=(path.name, f, "text/csv"),
            betas=FILES_BETA,
        )
    print(f"  Uploaded '{path.name}' -> file_id: {response.id}")
    return response.id


def delete_file(file_id: str) -> None:
    client.beta.files.delete(file_id, betas=FILES_BETA)
    print(f"  Deleted file: {file_id}")


def download_file(file_id: str, output_path: Path) -> None:
    """Download a generated file from the container and save it locally."""
    response = client.beta.files.download(file_id, betas=FILES_BETA)
    output_path.write_bytes(response.read())
    print(f"  Downloaded '{file_id}' -> {output_path.name}  ({output_path.stat().st_size:,} bytes)")


# ─────────────────────────────────────────────
# Response parser — handles all code execution block types
# ─────────────────────────────────────────────

def parse_and_print_response(response) -> list[str]:
    """
    Walk through every block in a code-execution response.
    Returns a list of file_ids for any files Claude generated.
    """
    generated_file_ids: list[str] = []
    exec_count = 0

    for block in response.content:

        if block.type == "text":
            print("\n--- Claude's Analysis ---")
            print(block.text)

        elif block.type == "server_tool_use":
            # The Python code Claude decided to run
            exec_count += 1
            code = block.input.get("code", "") if isinstance(block.input, dict) else ""
            print(f"\n--- Code Execution #{exec_count} (tool_id: {block.id[:20]}...) ---")
            print(code)

        elif block.type == "code_execution_tool_result":
            result = block.content  # CodeExecutionResultBlock
            print(f"\n--- Execution Result (exit={result.return_code}) ---")
            if result.stdout:
                print("STDOUT:\n" + result.stdout[:600])
            if result.stderr:
                print("STDERR:\n" + result.stderr[:200])

            # Check for generated files (plots, reports, etc.)
            for item in result.content:
                fid = getattr(item, "file_id", None)
                if fid:
                    print(f"  Generated file_id: {fid}")
                    generated_file_ids.append(fid)

        else:
            print(f"\n--- Unhandled block type: {block.type} ---")

    return generated_file_ids


# ─────────────────────────────────────────────
# Main chat call with code execution + uploaded file
# ─────────────────────────────────────────────

def analyze_with_code_execution(file_id: str, prompt: str) -> list[str]:
    """Send a message that includes an uploaded file for Claude to analyze with code."""
    print(f"\nSending to Claude ({MODEL})...\n")

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=[{"type": "code_execution_20250522", "name": "code_execution"}],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "container_upload", "file_id": file_id},  # inject file into container
            ],
        }],
    )

    return parse_and_print_response(response)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

print("=== Files API + Code Execution Demo ===")
print(f"Model: {MODEL}")
print()

print("Choose a mode:")
print("  [1] Churn analysis  -- upload CSV, Claude analyses it and generates a plot")
print("  [2] Quick demo      -- no CSV upload, just test code execution directly")
print()
mode = input("Mode (1/2, default 1): ").strip() or "1"


if mode == "1":
    # ── Step 1: Create the dataset ────────────────────────────────────────────
    print("\nStep 1: Creating churn dataset...")
    create_churn_csv(CSV_PATH)

    # ── Step 2: Upload via Files API ──────────────────────────────────────────
    print("\nStep 2: Uploading via Files API...")
    file_id = upload_file(CSV_PATH)

    # ── Step 3: Ask Claude to analyse + plot ──────────────────────────────────
    print("\nStep 3: Sending to Claude for code-based analysis...")
    prompt = (
        "You have access to streaming_churn.csv containing streaming service user data "
        "with columns: user_id, tier (basic/standard/premium), age, hours_per_week, "
        "months_subscribed, support_tickets, num_profiles, churned (0/1).\n\n"
        "Run a detailed churn analysis:\n"
        "1. Load the CSV and print basic statistics\n"
        "2. Calculate churn rate by tier\n"
        "3. Identify the 3 strongest predictors of churn\n"
        "4. Create a bar chart showing churn rate by subscription tier and save it as churn_plot.png\n"
        "Summarise your findings in plain text after running the code."
    )
    generated_ids = analyze_with_code_execution(file_id, prompt)

    # ── Step 4: Download any generated files ──────────────────────────────────
    if generated_ids:
        print(f"\nStep 4: Downloading {len(generated_ids)} generated file(s)...")
        for i, fid in enumerate(generated_ids):
            out_path = GENERATED_DIR / f"churn_analysis_{i+1}.png"
            try:
                download_file(fid, out_path)
            except Exception as e:
                print(f"  Could not download {fid}: {e}")
    else:
        print("\nNo generated files found in response.")

    # ── Step 5: Clean up uploaded file ────────────────────────────────────────
    print("\nStep 5: Cleaning up uploaded file from Files API...")
    delete_file(file_id)

    print("\nDone! Check generated_materials/ for the churn analysis plot.")


elif mode == "2":
    # Quick test — no file upload, just verify code execution works
    print("\nRunning quick code execution test (no file upload)...\n")

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[{"type": "code_execution_20250522", "name": "code_execution"}],
        messages=[{
            "role": "user",
            "content": (
                "Write and execute Python code that:\n"
                "1. Creates a list of squares [1,4,9,16,25]\n"
                "2. Calculates the mean and standard deviation\n"
                "3. Prints a summary"
            ),
        }],
    )

    print("Response blocks:\n")
    for block in response.content:
        if block.type == "server_tool_use":
            print("--- Code Claude wrote ---")
            print(block.input.get("code", ""))
        elif block.type == "code_execution_tool_result":
            print("\n--- Execution output ---")
            print(f"stdout: {block.content.stdout}")
            print(f"exit code: {block.content.return_code}")
        elif block.type == "text":
            print("\n--- Claude's interpretation ---")
            print(block.text)

else:
    print("Invalid mode.")
