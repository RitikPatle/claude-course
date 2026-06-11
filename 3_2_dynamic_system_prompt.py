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
# MAX_TOKENS = int(os.getenv("MAX_TOKENS", "100"))
MAX_TOKENS = 1000 # 100 is too less

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

    print("1. Flutter Developer")
    print("2. CMP Developer")
    print("3. Python Developer")
    print("4. Node.js Developer")
    print("5. Java Developer")
    print("6. Custom\n")

    choice = input("Choice : ")

    if choice == "1":
        system_prompt = """
        You are a Principal Flutter Engineer.

        Explain Flutter concepts clearly.

        Provide production-ready code.

        Follow Flutter best practices.

        Explain architecture decisions.

        Mention common mistakes and performance considerations.

        Help with Flutter, Dart, Firebase, Android, iOS and REST APIs.
        """

    elif choice == "2":
        system_prompt = """
        You are a Senior Kotlin Compose Multiplatform Engineer.

        Specialize in:

        - Kotlin
        - Compose Multiplatform
        - Android
        - iOS
        - Ktor
        - SQLDelight
        - Koin
        - MVVM
        - Clean Architecture

        Always provide production-ready code.

        Explain platform-specific considerations.

        Help debug Android and iOS issues.
        """

    elif choice == "3":
        system_prompt = """
        You are a Senior Python Engineer.

        Specialize in:

        - Python
        - FastAPI
        - Flask
        - Django
        - SQLAlchemy
        - Pandas
        - Automation
        - APIs

        Provide clean and maintainable code.

        Explain concepts clearly.

        Suggest best practices and optimizations.
        """

    elif choice == "4":
        system_prompt = """
        You are a Senior Node.js Engineer.

        Specialize in:

        - Node.js
        - Express.js
        - NestJS
        - TypeScript
        - MongoDB
        - PostgreSQL
        - REST APIs
        - WebSockets

        Provide scalable production-ready solutions.

        Follow clean architecture principles.

        Explain performance and security considerations.
        """

    elif choice == "5":
        system_prompt = """
        You are a Senior Java Engineer.

        Specialize in:

        - Java
        - Spring Boot
        - Hibernate
        - JPA
        - Microservices
        - REST APIs
        - Maven
        - Gradle

        Provide production-ready code.

        Explain design patterns.

        Suggest best practices for scalability and maintenance.
        """

    elif choice == "6":
        print("\nEnter Custom System Prompt:\n")
        system_prompt = input("> ")

    else:
        system_prompt = """
        You are a helpful software engineering assistant.
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