# Providing examples in a prompt is one of the most effective prompt engineering techniques.
# This is called one-shot prompting (one example) or multi-shot prompting (multiple examples).
# Instead of trying to describe the exact output format in words, you show Claude a real input paired with an ideal output.
# This is especially powerful when you need a precise JSON structure, a specific tone, or correct handling of edge cases.
# Without examples, Claude has to guess your intent from instructions alone and may produce inconsistent structure.
# With examples, Claude sees exactly what a correct response looks like and replicates that pattern reliably.
# Examples are most valuable when defining complex output formats, handling ambiguous inputs, or covering corner cases like missing fields.
# Always wrap examples in XML tags like <sample_input> and <ideal_output> so Claude can clearly tell the example apart from the real request.


import os

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = 600

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

client = Anthropic(api_key=API_KEY)

# Raw text about a person — this is the input Claude must parse
PERSON_TEXT = """
Alex is 29 years old and works as a backend engineer at a fintech startup in Berlin.
He studied computer science at TU Munich and speaks German and English fluently.
Outside work he enjoys rock climbing and reading sci-fi novels.
His email is alex.müller@email.com and he joined the platform on March 12, 2023.
"""

# Prompt without examples — Claude must guess the exact JSON shape you want
PROMPT_WITHOUT_EXAMPLES = f"""
Extract the person's profile from the text below and return it as JSON.

<person_text>
{PERSON_TEXT}
</person_text>
"""

# Prompt with a one-shot example — Claude sees exactly what the output should look like
PROMPT_WITH_EXAMPLES = f"""
Extract the person's profile from the text and return it as JSON.

Here is an example input with an ideal response:

<sample_input>
Maria is 34 and a product designer at a healthcare company in Amsterdam.
She speaks Dutch and English. She enjoys yoga and photography.
Her email is maria.j@email.com and she joined on January 5, 2022.
</sample_input>

<ideal_output>
{{
  "name": "Maria",
  "age": 34,
  "job_title": "Product Designer",
  "company_industry": "Healthcare",
  "city": "Amsterdam",
  "languages": ["Dutch", "English"],
  "hobbies": ["Yoga", "Photography"],
  "email": "maria.j@email.com",
  "joined_date": "2022-01-05"
}}
</ideal_output>

This output is ideal because it uses consistent key names, formats the date as ISO 8601,
groups languages and hobbies as arrays, and separates job title from industry.

Now extract the profile from this text:

<person_text>
{PERSON_TEXT}
</person_text>
"""


def ask(prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


try:
    print("=== Providing Examples in Prompts ===\n")
    print("Input text about a person is the same for both options.\n")
    print("1. Without example — Claude guesses the JSON shape")
    print("2. With example    — Claude follows an ideal output pattern\n")

    choice = input("Choose (1/2): ").strip()

    if choice == "1":
        print("\n--- Prompt: No Example ---\n")
        print(ask(PROMPT_WITHOUT_EXAMPLES))
        print("\nNotice: key names, date format, and array fields may vary each run.")

    elif choice == "2":
        print("\n--- Prompt: With One-Shot Example ---\n")
        print(ask(PROMPT_WITH_EXAMPLES))
        print("\nNotice: output consistently follows the shown structure — ISO dates, arrays, clean keys.")

    else:
        print("Invalid choice. Please run again and enter 1 or 2.")

except Exception as e:
    print(f"\nError: {e}")
