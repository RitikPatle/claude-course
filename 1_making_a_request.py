# This is the most basic way to use Claude. The program loads your API key and configuration from the .env file, creates an Anthropic client.
# It sends a single user message to Claude, and prints the response.
# Think of it as a simple question-and-answer interaction.
# Every time you run the program, Claude receives only the current question and has no knowledge of any previous questions you may have asked.
# This is useful for simple tasks such as generating text, answering questions, summarizing content, or testing that your API connection is working correctly.
# The main limitation is that Claude does not remember anything between requests, so every request starts as a completely new conversation.


import os

from dotenv import load_dotenv
from anthropic import Anthropic

# Load variables from .env
load_dotenv()

# Read configuration
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "100"))

# Validate configuration
if not API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY not found in .env file"
    )

# Create client
client = Anthropic(
    api_key=API_KEY
)

try:
    print("To ask a question please write down the text and press enter the answer will be of one scentence only.\n")
    userQuestion = input("Que : ")
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": f"{userQuestion} Answer in one sentence."
            }
        ]
    )

    # Extract response text
    answer = response.content[0].text

    print("\n===== Claude Response =====\n")
    print(answer)

except Exception as e:
    print("\n===== Error =====\n")
    print(f"Failed to call Anthropic API: {e}")