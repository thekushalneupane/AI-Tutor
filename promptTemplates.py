import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, **kwargs):
        return self.template.format(**kwargs)


explain_template = PromptTemplate(
    "Explain {topic} to a {level} student in {language}. Keep it under {length} words."
)

quiz_template = PromptTemplate(
    "Create {num_questions} quiz questions about {topic} suitable for a {level} level student."
)

compare_template = PromptTemplate(
    "Compare and contrast {concept_a} and {concept_b} in the context of {subject}."
)


def generate_response(prompt_text):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_text
    )
    return response.text


def main():
    print("=== Prompt Templates Demo ===")
    print("1. Explain a topic")
    print("2. Generate quiz questions")
    print("3. Compare two concepts")
    print("Type 'quit' to exit\n")

    while True:
        choice = input("Choose an option (1/2/3): ").strip()

        if choice.lower() == "quit":
            print("Goodbye!")
            break

        if choice == "1":
            topic = input("Topic: ").strip()
            level = input("Student level (beginner/intermediate/advanced): ").strip()
            language = input("Language: ").strip()
            length = input("Max word length: ").strip()

            prompt = explain_template.format(
                topic=topic, level=level, language=language, length=length
            )

        elif choice == "2":
            topic = input("Topic: ").strip()
            num_questions = input("Number of questions: ").strip()
            level = input("Student level: ").strip()

            prompt = quiz_template.format(
                topic=topic, num_questions=num_questions, level=level
            )

        elif choice == "3":
            concept_a = input("First concept: ").strip()
            concept_b = input("Second concept: ").strip()
            subject = input("Subject area: ").strip()

            prompt = compare_template.format(
                concept_a=concept_a, concept_b=concept_b, subject=subject
            )

        else:
            print("Invalid choice.\n")
            continue

        print("\nGenerating response...\n")
        response = generate_response(prompt)
        print(f"Response:\n{response}\n")

if __name__ == "__main__":
    main()