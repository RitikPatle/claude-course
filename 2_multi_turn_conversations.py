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


def add_user_message(messages, text):
    user_message = {
        "role": "user",
        "content": text
    }
    messages.append(user_message)


def add_assistant_message(messages, text):
    assistant_message = {
        "role": "assistant",
        "content": text
    }
    messages.append(assistant_message)


def chat(messages):
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages
    )

    return response.content[0].text


try:
    messages = []

    print("Multi Turn Claude Chat Started")
    print("Type 'exit' to quit.\n")

    while True:

        userQuestion = input("Que : ")

        if userQuestion.lower() == "exit":
            print("\nGoodbye!")
            break

        # Add user message
        add_user_message(
            messages,
            userQuestion
        )

        # Get Claude response
        answer = chat(messages)

        # Store Claude response
        add_assistant_message(
            messages,
            answer
        )

        print("\n===== Claude Response =====\n")
        print(answer)
        print()

except Exception as e:
    print("\n===== Error =====\n")
    print(f"Failed to call Anthropic API: {e}")