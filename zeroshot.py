import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def zero_shot_explain(topic):
    prompt = f"Explain {topic} in simple terms for a college student."

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

def main():
    print("=== Zero-Shot Prompting Demo ===")
    print("Type 'quit' to exit\n")

    while True:
        topic = input("Enter a topic to explain: ").strip()

        if topic.lower() == "quit":
            print("Goodbye!")
            break

        if not topic:
            continue

        print("\nGenerating explanation...\n")
        explanation = zero_shot_explain(topic)
        print(f"Explanation:\n{explanation}\n")

if __name__ == "__main__":
    main()