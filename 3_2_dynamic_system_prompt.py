# Dynamic System Prompt Chatbot
#
# Commands:
# /set      -> Change system prompt
# /system   -> View current system prompt
# /clear    -> Clear conversation history
# /history  -> Show number of stored messages
# /exit     -> Exit application

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

    print("===================================")
    print("      Claude Dynamic Chat")
    print("===================================\n")

    print("Choose Assistant Type\n")
    print("1. General Assistant")
    print("2. Math Tutor")
    print("3. Java Mentor")
    print("4. Python Mentor")
    print("5. Fitness Coach")
    print("6. Custom\n")

    choice = input("Choice : ")

    if choice == "1":
        system_prompt = """
        You are a helpful AI assistant.
        Provide clear and concise answers.
        """

    elif choice == "2":
        system_prompt = """
        You are a patient math tutor.

        Do not directly answer student questions.

        Guide students step by step.

        Ask questions that help them discover
        the answer themselves.
        """

    elif choice == "3":
        system_prompt = """
        You are a senior Java developer.

        Explain concepts simply.

        Provide examples whenever possible.

        Focus on teaching.
        """

    elif choice == "4":
        system_prompt = """
        You are a senior Python developer.

        Explain concepts clearly.

        Provide examples whenever possible.

        Focus on teaching.
        """

    elif choice == "5":
        system_prompt = """
        You are a fitness coach.

        Give practical fitness advice.

        Keep answers concise and actionable.
        """

    elif choice == "6":
        print("\nEnter Custom System Prompt:\n")
        system_prompt = input("> ")

    else:
        system_prompt = """
        You are a helpful AI assistant.
        """

    print("\n===================================")
    print("Chat Started")
    print("===================================\n")

    print("Commands:")
    print("/set      -> Change system prompt")
    print("/system   -> View system prompt")
    print("/clear    -> Clear conversation")
    print("/history  -> Show message count")
    print("/exit     -> Exit\n")

    while True:

        userQuestion = input("Que : ")

        if userQuestion.lower() == "/exit":
            print("\nGoodbye!")
            break

        elif userQuestion.lower() == "/system":

            print("\n===== Current System Prompt =====\n")
            print(system_prompt)
            print()
            continue

        elif userQuestion.lower() == "/clear":

            messages.clear()

            print("\nConversation history cleared.\n")
            continue

        elif userQuestion.lower() == "/history":

            print(
                f"\nStored Messages : {len(messages)}\n"
            )
            continue

        elif userQuestion.lower() == "/set":

            print(
                "\nEnter New System Prompt:\n"
            )

            system_prompt = input("> ")

            print(
                "\nSystem Prompt Updated.\n"
            )

            continue

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
    print(
        f"Failed to call Anthropic API: {e}"
    )