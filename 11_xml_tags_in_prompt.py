# XML tags give Claude clear boundaries when a prompt mixes instructions with large amounts of data.
# Without tags, Claude may confuse your instructions with the content you want it to analyze.
# For example, if you paste code and documentation together without any markers, Claude has to guess where one ends and the other begins.
# By wrapping sections in tags like <my_code> or <athlete_information>, you make the structure explicit.
# You can name tags anything descriptive — <sales_records>, <docs>, <user_data> — there are no reserved names.
# XML tags become especially valuable as prompts grow: the bigger the content, the more Claude benefits from clear delimiters.


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


def ask(prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# --- Demo 1: Without XML tags (ambiguous) ---
prompt_without_tags = """
Height: 6'2"
Weight: 180 lbs
Goal: Build muscle
Dietary restrictions: Vegetarian

Generate a one-day meal plan based on the information above.
"""

# --- Demo 2: With XML tags (structured) ---
prompt_with_tags = """
<athlete_information>
Height: 6'2"
Weight: 180 lbs
Goal: Build muscle
Dietary restrictions: Vegetarian
</athlete_information>

Generate a one-day meal plan based on the athlete information above.
"""

try:
    print("=== XML Tags in Prompts ===\n")
    print("Same data, same question — two different prompt structures.\n")

    print("--- Without XML Tags ---")
    print("Claude must guess which lines are data and which is the instruction.\n")
    answer1 = ask(prompt_without_tags)
    print(answer1)

    print("\n--- With XML Tags ---")
    print("Claude knows exactly what <athlete_information> contains.\n")
    answer2 = ask(prompt_with_tags)
    print(answer2)

    # --- Interactive: user provides their own data ---
    print("\n=== Try It Yourself ===")
    print("Enter your own details and Claude will use them inside XML tags.\n")

    name = input("Your name      : ")
    goal = input("Your goal      : ")
    restriction = input("Dietary restriction (or 'none') : ")

    custom_prompt = f"""
<user_information>
Name: {name}
Goal: {goal}
Dietary restriction: {restriction}
</user_information>

Based on the user information above, suggest a simple one-day meal plan.
Keep it brief and practical.
"""

    print("\n--- Claude's Response ---\n")
    print(ask(custom_prompt))

except Exception as e:
    print(f"\nError: {e}")
