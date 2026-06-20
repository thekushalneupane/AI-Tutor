import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Step 1: Topic -> Explanation
explain_prompt = PromptTemplate.from_template(
    "Explain {topic} clearly in 3-4 sentences for a college student."
)
explain_chain = explain_prompt | llm

# Step 2: Explanation -> Study Notes
notes_prompt = PromptTemplate.from_template(
    "Convert this explanation into 3-5 concise bullet-point study notes:\n\n{explanation}"
)
notes_chain = notes_prompt | llm

# Step 3: Notes -> Quiz
quiz_prompt = PromptTemplate.from_template(
    "Based on these study notes, generate 3 quiz questions with answers:\n\n{notes}"
)
quiz_chain = quiz_prompt | llm


def run_pipeline(topic):
    # Step 1
    explanation = explain_chain.invoke({"topic": topic}).content
    print(f"--- Explanation ---\n{explanation}\n")

    # Step 2
    notes = notes_chain.invoke({"explanation": explanation}).content
    print(f"--- Study Notes ---\n{notes}\n")

    # Step 3
    quiz = quiz_chain.invoke({"notes": notes}).content
    print(f"--- Quiz ---\n{quiz}\n")

    return explanation, notes, quiz


def main():
    print("=== LangChain Pipeline Demo (Topic -> Explanation -> Notes -> Quiz) ===")
    print("Type 'quit' to exit\n")

    while True:
        topic = input("Enter a topic: ").strip()

        if topic.lower() == "quit":
            print("Goodbye!")
            break

        if not topic:
            continue

        print("\nRunning pipeline...\n")
        run_pipeline(topic)

if __name__ == "__main__":
    main()