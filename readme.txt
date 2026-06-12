Claude API Course — Python
==========================

A hands-on Python course that teaches how to use the Anthropic Claude API,
progressing from a single request through to evaluations.

------------------------------
Prerequisites
------------------------------
- Python 3.10 or higher (match/case syntax used in some files)
- An Anthropic API key — get one at https://console.anthropic.com

------------------------------
Project Setup
------------------------------

1. Create and activate the virtual environment:

   macOS / Linux:
     python3 -m venv .venv
     source .venv/bin/activate

   Windows (PowerShell):
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1

   Windows (Command Prompt):
     python -m venv .venv
     .\.venv\Scripts\activate.bat

2. Install dependencies:
     pip install -r requirements.txt

3. Configure environment variables:
   Copy the example below into a file named .env in the project root:

     ANTHROPIC_API_KEY=your_api_key_here
     MODEL=claude-haiku-4-5
     MAX_TOKENS=1000

------------------------------
.env File Reference
------------------------------
ANTHROPIC_API_KEY  (required) Your Anthropic API key
MODEL              (optional) Claude model to use. Default: claude-haiku-4-5
MAX_TOKENS         (optional) Max tokens per response. Default: 1000

------------------------------
Course Files
------------------------------
Run each file with:  python <filename>

1_making_a_request.py
  Single question -> single answer. Demonstrates the most basic API call.
  Claude has no memory between runs.

2_multi_turn_conversations.py
  Chat loop that maintains message history so Claude remembers context.
  Type 'exit' to quit.

3_1_system_prompts.py
  Adds a system prompt that turns Claude into a Flutter engineering mentor.
  Shows how to specialise Claude's behaviour without changing the model.

3_2_dynamic_system_prompt.py
  Interactive chat where you choose an assistant role at startup.
  Supports runtime commands to view, change, or clear state:
    /set      Change the system prompt mid-conversation
    /system   Print the current system prompt
    /clear    Clear conversation history
    /history  Show how many messages are stored
    /exit     Exit the program

4_temperature.py
  Demonstrates the temperature parameter (0.0 - 1.0).
  Choose a preset at startup; lower = more deterministic, higher = more creative.

5_response_streaming.py
  Streams Claude's response token-by-token as it is generated,
  so text appears on screen immediately rather than after a full wait.
  Type 'exit' to quit.

6_structured_data.py
  Uses prefill and stop sequences to force Claude to return raw JSON
  with no markdown wrapping, ready for direct json.loads() parsing.

7_running the eval.py
  Evaluation pipeline: creates a dataset, runs each test case through Claude,
  scores results, and writes four JSON reports to generated_materials/:
    dataset.json       The input test cases
    eval_results.json  Claude's output per test case
    scores.json        Score summary (average, highest, lowest)
    report.json        Full evaluation report with recommendations

------------------------------
Deactivate the virtual environment
------------------------------
  deactivate
