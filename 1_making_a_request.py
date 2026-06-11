import os

from dotenv import load_dotenv
from anthropic import Anthropic

# Load variables from .env
load_dotenv()

# Read configuration
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-haiku-4-0")
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
                "content": f"${userQuestion} Answer in one sentence."
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