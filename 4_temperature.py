# The temperature parameter controls how predictable or creative Claude's responses will be.
# You can think of temperature as a creativity slider that ranges from 0.0 to 1.0.
# A low temperature makes Claude more focused, consistent, and deterministic, meaning it will usually choose the most likely response and produce similar answers when asked the same question multiple times.
# This is ideal for coding, debugging, technical explanations, data processing, and factual questions where accuracy and consistency are important.
# A high temperature makes Claude more creative and varied by allowing it to consider less likely responses, which can lead to more diverse ideas and writing styles.
# This is useful for brainstorming, storytelling, marketing content, and creative writing tasks.
# Temperature does not change Claude's knowledge or intelligence; it only changes how much randomness is introduced when generating responses.
# In practical software development scenarios such as Flutter, Kotlin CMP, Spring Boot, API integration, and debugging, temperatures between 0.0 and 0.3 are generally recommended because they produce more reliable and predictable code examples.
# By adding temperature as a parameter to your chat function, you gain fine-grained control over how Claude responds, allowing the same model to behave either like a precise technical assistant or a creative brainstorming partner depending on your needs.


import os

from dotenv import load_dotenv
from anthropic import Anthropic

# Load variables from .env
load_dotenv()

# Read configuration
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
# MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
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
    messages.append({
        "role": "user",
        "content": text
    })


def add_assistant_message(messages, text):
    messages.append({
        "role": "assistant",
        "content": text
    })


def chat(
        messages,
        system=None,
        temperature=1.0
):

    params = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
        "temperature": temperature
    }

    if system:
        params["system"] = system

    response = client.messages.create(
        **params
    )

    return response.content[0].text


try:

    messages = []

    print("Claude Temperature Demo\n")

    print("Temperature Options")
    print("1. Coding (0.0)")
    print("2. Technical (0.2)")
    print("3. Balanced (0.5)")
    print("4. Creative (0.8)")
    print("5. Very Creative (1.0)\n")

    choice = input("Choice : ")

    if choice == "1":
        temperature = 0.0

    elif choice == "2":
        temperature = 0.2

    elif choice == "3":
        temperature = 0.5

    elif choice == "4":
        temperature = 0.8

    elif choice == "5":
        temperature = 1.0

    else:
        temperature = 0.5

    print(
        f"\nTemperature Selected: {temperature}\n"
    )

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
            messages=messages,
            temperature=temperature
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