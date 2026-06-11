# Running an evaluation means taking every test case in a dataset, sending it through a prompt, collecting Claude's responses, and storing the results for analysis.
# The evaluation pipeline typically consists of a function that generates a prompt from a test case, a function that executes and grades a single test case, and a function that processes the entire dataset.
# This allows developers to automatically test prompts against many different inputs instead of manually checking a few examples.
# The collected results can then be scored and compared to identify weaknesses in the prompt and measure improvements over time.
# Running evaluations is a core part of prompt engineering because it provides objective evidence about how well a prompt performs before it is used in production.


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

    user_message = {
        "role": "user",
        "content": text
    }

    messages.append(
        user_message
    )


def add_assistant_message(messages, text):

    assistant_message = {
        "role": "assistant",
        "content": text
    }

    messages.append(
        assistant_message
    )


def chat(
        messages,
        system=None,
        temperature=0.0
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


def ensure_generated_materials_folder():

    os.makedirs(
        "generated_materials",
        exist_ok=True
    )


def create_dataset():

    dataset = [
        {
            "task": "Write a Python function to reverse a string"
        },
        {
            "task": "Create a Python function that validates email addresses"
        },
        {
            "task": "Generate a JSON object representing a user profile"
        }
    ]

    with open(
        "generated_materials/dataset.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=4
        )

    return dataset


def run_prompt(test_case):

    prompt = f"""
Please solve the following task.

Return a complete solution.

Task:

{test_case["task"]}
"""

    messages = []

    add_user_message(
        messages,
        prompt
    )

    output = chat(
        messages=messages,
        temperature=0.0
    )

    return output


def run_test_case(test_case):

    print(
        f"Running: "
        f"{test_case['task']}"
    )

    output = run_prompt(
        test_case
    )

    # Placeholder grading
    score = 10

    return {
        "test_case": test_case,
        "output": output,
        "score": score
    }


def run_eval(dataset):

    results = []

    for test_case in dataset:

        result = run_test_case(
            test_case
        )

        results.append(
            result
        )

    return results


def save_eval_results(results):

    with open(
        "generated_materials/eval_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


def generate_scores(results):

    scores = [
        result["score"]
        for result in results
    ]

    summary = {
        "total_test_cases":
            len(scores),

        "average_score":
            sum(scores) / len(scores),

        "highest_score":
            max(scores),

        "lowest_score":
            min(scores)
    }

    with open(
        "generated_materials/scores.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    return summary


def generate_report(summary):

    report = {

        "evaluation_name":
            "Prompt Evaluation V1",

        "model":
            MODEL,

        "dataset_size":
            summary["total_test_cases"],

        "average_score":
            summary["average_score"],

        "summary":
            (
                "Claude successfully "
                "generated outputs for all "
                "test cases. Placeholder "
                "grading is currently used."
            ),

        "recommendations": [

            (
                "Implement an AI grader"
            ),

            (
                "Increase dataset size"
            ),

            (
                "Compare prompt versions"
            ),

            (
                "Track evaluation history"
            )
        ]
    }

    with open(
        "generated_materials/report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )


try:

    print(
        "\n===== Running Evaluation =====\n"
    )

    ensure_generated_materials_folder()

    dataset = create_dataset()

    results = run_eval(
        dataset
    )

    save_eval_results(
        results
    )

    summary = generate_scores(
        results
    )

    generate_report(
        summary
    )

    print(
        "\n===== Evaluation Complete =====\n"
    )

    print(
        "Generated Files:\n"
    )

    print(
        "generated_materials/dataset.json"
    )

    print(
        "generated_materials/eval_results.json"
    )

    print(
        "generated_materials/scores.json"
    )

    print(
        "generated_materials/report.json"
    )

except Exception as e:

    print(
        "\n===== Error =====\n"
    )

    print(
        f"Failed to run evaluation: {e}"
    )