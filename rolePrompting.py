import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

ROLES = {
    "1": ("Friendly Study Coach", "You are a warm, encouraging study coach. Explain things simply, use relatable analogies, and motivate the student."),
    "2": ("Strict Examiner", "You are a strict, no-nonsense examiner. Be precise, formal, and point out gaps in understanding directly."),
    "3": ("Funny Professor", "You are a witty professor who explains concepts using humor and pop-culture references while still being accurate.")
}

def role_prompt(role_description, question):
    prompt = f"""{role_description}

Student's question: {question}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

def main():
    print("=== Role Prompting Demo ===")
    print("Type 'quit' to exit\n")

    while True:
        print("Choose a role:")
        for key, (name, _) in ROLES.items():
            print(f"  {key}. {name}")

        choice = input("\nEnter role number (or 'quit'): ").strip()

        if choice.lower() == "quit":
            print("Goodbye!")
            break

        if choice not in ROLES:
            print("Invalid choice, try again.\n")
            continue

        role_name, role_description = ROLES[choice]
        question = input(f"\nAsk your question to the {role_name}: ").strip()

        if not question:
            continue

        print(f"\n{role_name} is responding...\n")
        answer = role_prompt(role_description, question)
        print(f"{role_name}:\n{answer}\n")

if __name__ == "__main__":
    main()