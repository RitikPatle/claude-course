# Response streaming improves the user experience by displaying Claude's answer as it is being generated instead of waiting for the entire response to finish.
# In a normal API request, the user sends a question and waits until Claude has generated the complete answer before seeing any output.
# For longer responses, this can make the application feel slow because nothing appears on the screen while Claude is thinking.
# With streaming enabled, Claude sends small chunks of text as they are generated, allowing the user to watch the response appear word by word or sentence by sentence, similar to how ChatGPT displays responses.
# This makes applications feel much more responsive and interactive. Streaming is especially useful for chat applications, coding assistants, AI tutors, and any situation where responses may take several seconds to generate.
# After streaming is complete, you can still access the final assembled message for storage, conversation history, logging, or additional processing.
# In production AI applications, streaming is commonly used because it significantly improves perceived performance without changing the quality of the generated response.


import os

from dotenv import load_dotenv
from anthropic import Anthropic

# Load variables from .env
load_dotenv()

# Read configuration
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
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
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def stream_chat(messages):

    chunks = []

    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages
    ) as stream:

        print("\n===== Claude Response =====\n")

        for text in stream.text_stream:
            print(text, end="", flush=True)
            chunks.append(text)

        print("\n")

    return "".join(chunks)


try:

    messages = []

    print("Claude Streaming Chat Started")
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

        answer = stream_chat(
            messages
        )

        add_assistant_message(
            messages,
            answer
        )

except Exception as e:

    print("\n===== Error =====\n")
    print(
        f"Failed to call Anthropic API: {e}"
    )