import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def few_shot_quiz(topic):
    prompt = f"""Generate 5 quiz questions in this exact format. Here are examples:

Topic: Photosynthesis
Q1: What pigment in plants absorbs sunlight for photosynthesis?
A1: Chlorophyll

Topic: Newton's Laws
Q1: What does Newton's First Law describe?
A1: An object's tendency to resist changes in its motion (inertia)

Now generate exactly 5 questions and answers for this topic, following the same Q/A format:

Topic: {topic}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

def main():
    print("=== Few-Shot Prompting Demo (Quiz Generator) ===")
    print("Type 'quit' to exit\n")

    while True:
        topic = input("Enter a topic for quiz generation: ").strip()

        if topic.lower() == "quit":
            print("Goodbye!")
            break

        if not topic:
            continue

        print("\nGenerating quiz...\n")
        quiz = few_shot_quiz(topic)
        print(f"Quiz:\n{quiz}\n")

if __name__ == "__main__":
    main()