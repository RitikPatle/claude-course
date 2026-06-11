# Structured data generation is used when you need Claude to return data in a specific format such as JSON, Python code, CSV, XML, SQL queries, or lists without any extra explanations.
# Normally Claude tries to be helpful by wrapping code in markdown blocks and adding descriptive text before or after the result.
# While this is useful for humans, it can be problematic when your application needs to directly process the output.
# For example, if you're building a tool that generates JSON configurations, API payloads, SQL queries, or Flutter model classes, you want only the structured content and nothing else.
# Structured data generation solves this by guiding Claude to continue from a predefined assistant message and stopping generation when a specific sequence is reached.
# This gives you much cleaner output that can be directly parsed by your application without requiring users to manually remove markdown formatting or explanations.
# In production AI applications, structured data generation is commonly used when integrating Claude with databases, APIs, automation systems, code generators, and configuration management tools.


import os
import json

from dotenv import load_dotenv
from anthropic import Anthropic

# Load variables from .env
load_dotenv()

# Read configuration
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

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


def chat(messages):

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages,
        stop_sequences=["```"]
    )

    return response.content[0].text


try:

    messages = []

    add_user_message(
        messages,
        """
        Generate a JSON object for a user profile.

        Fields:
        - name
        - age
        - city
        """
    )

    # Prefill assistant response
    add_assistant_message(
        messages,
        "```json"
    )

    text = chat(messages)

    print("\n===== Raw JSON =====\n")
    print(text)

    # Convert string to JSON object
    parsed_json = json.loads(
        text.strip()
    )

    print("\n===== Parsed JSON =====\n")
    print(parsed_json)

except Exception as e:

    print("\n===== Error =====\n")
    print(
        f"Failed to call Anthropic API: {e}"
    )