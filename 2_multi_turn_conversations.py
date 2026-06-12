# The multi-turn conversation program solves the memory problem by maintaining a list of messages inside your application.
# Whenever the user asks a question, that question is added to the message history.
# After Claude responds, the response is also added to the history. When the next question is asked, the entire conversation history is sent back to Claude along with the new message.
# Because Claude receives all previous messages each time, it can understand follow-up questions and maintain context throughout the conversation.
# This creates the experience of a real chatbot even though Claude itself does not store any conversation history.
# In practice, most AI chat applications use this approach because it allows natural conversations where users can refer to previous answers and continue discussions without repeating information.


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
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


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