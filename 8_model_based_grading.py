# Model-based grading is an evaluation technique where an AI model is used to assess the quality of another model's output and assign a score, typically between 1 and 10, along with reasoning, strengths, and weaknesses.
# Unlike code graders that rely on fixed programmatic rules or human graders that require manual review, model graders can evaluate subjective qualities such as instruction following, completeness, helpfulness, relevance, and overall response quality.
# The process involves sending the original task and the generated solution to a grading prompt, asking the model to return structured feedback in JSON format.
# This grading result is then integrated into the evaluation pipeline, allowing each test case to receive a measurable score and explanation.
# By averaging scores across a dataset, developers can objectively compare prompt versions, identify weaknesses, and track improvements over time while automating much of the evaluation process.


import os
import json

from dotenv import load_dotenv
from anthropic import Anthropic
from statistics import mean

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
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(
        messages,
        system=None,
        temperature=0.0,
        stop_sequences=None
):

    params = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
        "temperature": temperature
    }

    if system:
        params["system"] = system

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

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

    add_user_message(messages, prompt)

    output = chat(messages=messages, temperature=0.0)

    return output

def grade_by_model(test_case, output):
    eval_prompt = f"""
You are an expert Python code reviewer. Your task is to evaluate the following AI-generated solution.

Original Task:
<task>
{test_case["task"]}
</task>

Solution to Evaluate:
<solution>
{output}
</solution>

Output Format
Provide your evaluation as a structured JSON object with the following fields, in this specific order:
- "strengths": An array of 1-3 key strengths
- "weaknesses": An array of 1-3 key areas for improvement
- "reasoning": A concise explanation of your overall assessment
- "score": A number between 1-10

Respond with JSON. Keep your response concise and direct.
Example response shape:
{{
    "strengths": string[],
    "weaknesses": string[],
    "reasoning": string,
    "score": number
}}
    """

    messages = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "{")
    eval_text = "{" + chat(messages)
    return json.loads(eval_text)

def run_test_case(test_case):

    print(f"Running: {test_case['task']}")

    output = run_prompt(test_case)
    grade = grade_by_model(test_case, output)
    
    return {
        "test_case": test_case,
        "output": output,
        "score": grade["score"],
        "reasoning": grade["reasoning"],
        "strengths": grade["strengths"],
        "weaknesses": grade["weaknesses"]
    }


def run_eval(dataset):
    """Loads the dataset and calls run_test_case with each case"""
    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    average_score = mean([result["score"] for result in results])
    print(f"Average score: {average_score}")

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

    scores = [result["score"] for result in results]
    total = len(scores)

    summary = {
        "total_test_cases": total,
        "average_score": sum(scores) / total if total > 0 else 0,
        "highest_score": max(scores) if scores else 0,
        "lowest_score": min(scores) if scores else 0,
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
        "evaluation_name": "Prompt Evaluation V1",
        "model": MODEL,
        "dataset_size": summary["total_test_cases"],
        "average_score": summary["average_score"],
        "summary": (
            "Claude successfully generated outputs for all test cases. "
            "Placeholder grading is currently used."
        ),
        "recommendations": [
            "Implement an AI grader",
            "Increase dataset size",
            "Compare prompt versions",
            "Track evaluation history",
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

    generated_files = [
        "generated_materials/dataset.json",
        "generated_materials/eval_results.json",
        "generated_materials/scores.json",
        "generated_materials/report.json",
    ]

    print("Generated Files:\n")
    print("\n".join(generated_files))

except Exception as e:

    print(
        "\n===== Error =====\n"
    )

    print(
        f"Failed to run evaluation: {e}"
    )