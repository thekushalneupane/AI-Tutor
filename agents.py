import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


@tool
def calculator(expression: str) -> str:
    """Evaluates a basic math expression. Use this when the user asks a math question.
    Input should be a valid Python math expression, e.g. '2 + 2' or '15 * 7'."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"The result is {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def study_planner(subjects: str) -> str:
    """Creates a simple study schedule. Use this when the user asks for a study plan
    or wants help organizing study time. Input should be a comma-separated list of subjects,
    e.g. 'Math, Physics, Compiler Design'."""
    subject_list = [s.strip() for s in subjects.split(",")]
    plan = "Here's a suggested daily study plan:\n"
    for i, subject in enumerate(subject_list, 1):
        plan += f"{i}. {subject} - 45 minutes\n"
    plan += "\nTake a 10-minute break between each subject."
    return plan


@tool
def summarizer(text: str) -> str:
    """Summarizes a block of text into key points. Use this when the user provides
    a paragraph or long text and wants it shortened."""
    response = llm.invoke(f"Summarize this in 2-3 bullet points:\n\n{text}")
    return response.content


tools = [calculator, study_planner, summarizer]

agent = create_agent(llm, tools)


def chat(user_input):
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    return result["messages"][-1].content


def main():
    print("=== Agents & Tools Demo ===")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        print("\nAgent is thinking...\n")
        response = chat(user_input)
        print(f"Agent: {response}\n")

if __name__ == "__main__":
    main()