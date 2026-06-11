# The system prompt program builds on the multi-turn conversation approach by adding instructions that define how Claude should behave.
# A system prompt is a special message that is not written by the user but by the developer.
# It tells Claude what role it should play, what tone it should use, and what rules it should follow when responding.
# For example, you can make Claude act like a math tutor, a customer support representative, a fitness coach, or a senior software engineer.
# The user's questions remain the same, but the system prompt changes the style and behavior of the responses.
# This is one of the most important features for building real-world AI applications because it allows developers to create specialized assistants without changing the underlying model.
# Most production AI systems rely heavily on system prompts to ensure consistent, focused, and appropriate responses.


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


def chat(messages, system=None):

    params = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": messages
    }

    if system:
        params["system"] = system

    response = client.messages.create(**params)

    return response.content[0].text


try:

    messages = []

    system_prompt = """
    You are a patient math tutor.

    Do not directly answer student questions.

    Guide students step by step.

    Ask questions that help them discover the answer themselves.
    """

    print("Math Tutor Claude Started")
    print("Type 'exit' to quit.\n")

    while True:

        userQuestion = input("Que : ")

        if userQuestion.lower() == "exit":
            print("\nGoodbye!")
            break

        add_user_message(
            messages,
            userQuestion
        )

        answer = chat(
            messages,
            system=system_prompt
        )

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