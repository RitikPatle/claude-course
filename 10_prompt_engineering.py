# Prompt Engineering: Write a prompt → Test it → Score it → Improve it
# The goal is to show that a better-structured prompt gets consistently higher scores.

import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DATASET = [
    {"topic": "Python Lists"},
    {"topic": "Python Dictionaries"},
    {"topic": "List Comprehension"},
]

NORMAL_PROMPT = "Explain the topic."

GOOD_PROMPT = """You are a Senior Python Mentor teaching beginners.

For every topic, cover:
- What it is (clear definition)
- Why it is useful
- A simple working code example
- One common mistake to avoid"""


def generate_answer(prompt: str, topic: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=prompt,
        messages=[{"role": "user", "content": f"Topic: {topic}"}],
    )
    return response.content[0].text


def grade_answer(topic: str, answer: str) -> dict:
    grading_prompt = f"""You are an evaluation expert. Grade the following Python explanation.

Topic: {topic}
Answer: {answer}

Return ONLY valid JSON in this format:
{{"score": <1-10>, "reason": "<one sentence why>"}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": grading_prompt}],
    )
    text = response.content[0].text.strip()
    # Extract JSON even if surrounded by markdown
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()
    return json.loads(text)


def review_prompt(user_prompt: str, avg_score: float) -> str:
    review_request = f"""You are a prompt engineering expert.

This prompt scored {avg_score:.1f}/10 on a Python teaching dataset:

--- PROMPT ---
{user_prompt}
--------------

In 3-4 lines explain what is weak about it, then provide an improved version.
Format:
ISSUES: <your analysis>
IMPROVED PROMPT:
<the better prompt>"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": review_request}],
    )
    return response.content[0].text


def run_evaluation(prompt_label: str, prompt: str) -> float:
    print(f"\n{'='*55}")
    print(f"  Prompt Type : {prompt_label}")
    print(f"{'='*55}")

    total = 0
    for item in DATASET:
        topic = item["topic"]
        answer = generate_answer(prompt, topic)
        grade = grade_answer(topic, answer)
        score = grade["score"]
        reason = grade["reason"]
        total += score
        print(f"\n  Topic  : {topic}")
        print(f"  Score  : {score}/10")
        print(f"  Reason : {reason}")

    avg = total / len(DATASET)
    print(f"\n  Average Score : {avg:.1f}/10")
    return avg


def main():
    print("\n=== Prompt Engineering Demo ===")
    print("Learn how prompt quality affects Claude's answers.\n")
    print("Choose a prompt to evaluate:")
    print("  1. Normal Prompt  — vague, minimal instruction")
    print("  2. Good Prompt    — structured, role-based")
    print("  3. Custom Prompt  — enter your own\n")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        avg = run_evaluation("Normal", NORMAL_PROMPT)
        print(f"\nThe normal prompt scored {avg:.1f}/10.")
        print("Try the Good Prompt (option 2) to see the difference.\n")

    elif choice == "2":
        avg = run_evaluation("Good", GOOD_PROMPT)
        print(f"\nThe structured prompt scored {avg:.1f}/10.")
        print("The role + structure + checklist made answers much better.\n")

    elif choice == "3":
        print("\nEnter your custom prompt (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        user_prompt = "\n".join(lines).strip()

        if not user_prompt:
            print("No prompt entered. Exiting.")
            return

        avg = run_evaluation("Custom", user_prompt)

        if avg < 7.0:
            print(f"\nScore is {avg:.1f}/10 — Claude will now review your prompt...\n")
            review = review_prompt(user_prompt, avg)
            print("--- Prompt Review ---")
            print(review)
            print("---------------------\n")
        else:
            print(f"\nGood job! Your prompt scored {avg:.1f}/10. Well structured.\n")
    else:
        print("Invalid choice. Please run again and enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
