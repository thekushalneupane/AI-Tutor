import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def cot_solve(problem):
    prompt = f"""Solve the following problem. Think through it step by step, showing your reasoning clearly before giving the final answer.

Problem: {problem}

Let's think step by step:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

def main():
    print("=== Chain-of-Thought Prompting Demo ===")
    print("Type 'quit' to exit\n")

    while True:
        problem = input("Enter a math/logic problem: ").strip()

        if problem.lower() == "quit":
            print("Goodbye!")
            break

        if not problem:
            continue

        print("\nSolving...\n")
        solution = cot_solve(problem)
        print(f"Solution:\n{solution}\n")

if __name__ == "__main__":
    main()