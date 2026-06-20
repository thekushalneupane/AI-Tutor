import os
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# ----- Step 1: Basic chat -----
conversation_history = []

def basic_chat():
    print("\n--- Basic Chat (type 'back' to return to menu) ---")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "back":
            break
        conversation_history.append({"role": "user", "parts": [{"text": user_input}]})
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=conversation_history,
            config={"system_instruction": "You are a helpful academic tutor."}
        )
        reply = response.text
        conversation_history.append({"role": "model", "parts": [{"text": reply}]})
        print(f"\nTutor: {reply}\n")


# ----- Step 2: Zero-shot -----
def zero_shot():
    print("\n--- Zero-Shot Explanation (type 'back' to return) ---")
    while True:
        topic = input("Topic: ").strip()
        if topic.lower() == "back":
            break
        prompt = f"Explain {topic} in simple terms for a college student."
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        print(f"\n{response.text}\n")


# ----- Step 3: Few-shot quiz -----
def few_shot_quiz():
    print("\n--- Few-Shot Quiz Generator (type 'back' to return) ---")
    while True:
        topic = input("Topic: ").strip()
        if topic.lower() == "back":
            break
        prompt = f"""Generate 5 quiz questions in this exact format. Here are examples:

Topic: Photosynthesis
Q1: What pigment in plants absorbs sunlight for photosynthesis?
A1: Chlorophyll

Topic: Newton's Laws
Q1: What does Newton's First Law describe?
A1: An object's tendency to resist changes in its motion (inertia)

Now generate exactly 5 questions and answers for this topic, following the same Q/A format:

Topic: {topic}"""
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        print(f"\n{response.text}\n")


# ----- Step 4: Chain-of-Thought -----
def chain_of_thought():
    print("\n--- Chain-of-Thought Solver (type 'back' to return) ---")
    while True:
        problem = input("Problem: ").strip()
        if problem.lower() == "back":
            break
        prompt = f"""Solve the following problem. Think through it step by step, showing your reasoning clearly before giving the final answer.

Problem: {problem}

Let's think step by step:"""
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        print(f"\n{response.text}\n")


# ----- Step 5: Role prompting -----
ROLES = {
    "1": ("Friendly Study Coach", "You are a warm, encouraging study coach. Explain things simply, use relatable analogies, and motivate the student."),
    "2": ("Strict Examiner", "You are a strict, no-nonsense examiner. Be precise, formal, and point out gaps in understanding directly."),
    "3": ("Funny Professor", "You are a witty professor who explains concepts using humor and pop-culture references while still being accurate.")
}

def role_prompting():
    print("\n--- Role Prompting (type 'back' to return) ---")
    while True:
        print("Roles: 1. Friendly Coach  2. Strict Examiner  3. Funny Professor")
        choice = input("Choose role (or 'back'): ").strip()
        if choice.lower() == "back":
            break
        if choice not in ROLES:
            print("Invalid choice.")
            continue
        role_name, role_description = ROLES[choice]
        question = input("Your question: ").strip()
        prompt = f"{role_description}\n\nStudent's question: {question}"
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        print(f"\n{role_name}:\n{response.text}\n")


# ----- Step 6: Prompt templates -----
def prompt_templates():
    print("\n--- Prompt Templates (type 'back' to return) ---")
    while True:
        choice = input("1. Explain  2. Quiz  3. Compare  (or 'back'): ").strip()
        if choice.lower() == "back":
            break
        if choice == "1":
            topic = input("Topic: ").strip()
            level = input("Level: ").strip()
            prompt = f"Explain {topic} to a {level} student in English. Keep it under 100 words."
        elif choice == "2":
            topic = input("Topic: ").strip()
            prompt = f"Create 5 quiz questions about {topic} suitable for a beginner level student."
        elif choice == "3":
            a = input("Concept A: ").strip()
            b = input("Concept B: ").strip()
            prompt = f"Compare and contrast {a} and {b}."
        else:
            print("Invalid choice.")
            continue
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        print(f"\n{response.text}\n")


# ----- Step 7: LangChain chains -----
def langchain_chains():
    print("\n--- LangChain Pipeline: Topic -> Explanation -> Notes -> Quiz (type 'back' to return) ---")
    explain_prompt = PromptTemplate.from_template("Explain {topic} clearly in 3-4 sentences for a college student.")
    notes_prompt = PromptTemplate.from_template("Convert this explanation into 3-5 concise bullet-point study notes:\n\n{explanation}")
    quiz_prompt = PromptTemplate.from_template("Based on these study notes, generate 3 quiz questions with answers:\n\n{notes}")

    explain_chain = explain_prompt | llm
    notes_chain = notes_prompt | llm
    quiz_chain = quiz_prompt | llm

    while True:
        topic = input("Topic (or 'back'): ").strip()
        if topic.lower() == "back":
            break
        explanation = explain_chain.invoke({"topic": topic}).content
        print(f"\n--- Explanation ---\n{explanation}\n")
        notes = notes_chain.invoke({"explanation": explanation}).content
        print(f"--- Study Notes ---\n{notes}\n")
        quiz = quiz_chain.invoke({"notes": notes}).content
        print(f"--- Quiz ---\n{quiz}\n")


# ----- Step 8: Memory -----
memory_chat_history = []
student_info = {"name": None}

def memory_chat():
    print("\n--- Memory Chat (type 'back' to return, 'memory' to inspect) ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful academic tutor. Remember the student's name and topics discussed."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    chain = prompt | llm

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "back":
            break
        if user_input.lower() == "memory":
            print(f"Remembered name: {student_info['name']}, Messages: {len(memory_chat_history)}")
            continue
        if "my name is" in user_input.lower():
            name = user_input.lower().split("my name is")[-1].strip().split()[0]
            student_info["name"] = name.capitalize()

        response = chain.invoke({"chat_history": memory_chat_history, "input": user_input})
        memory_chat_history.append(HumanMessage(content=user_input))
        memory_chat_history.append(AIMessage(content=response.content))
        print(f"\nTutor: {response.content}\n")


# ----- Step 9: Agents -----
@tool
def calculator(expression: str) -> str:
    """Evaluates a basic math expression."""
    try:
        return f"The result is {eval(expression, {'__builtins__': {}})}"
    except Exception as e:
        return f"Error: {e}"

@tool
def study_planner(subjects: str) -> str:
    """Creates a simple study schedule from a comma-separated list of subjects."""
    subject_list = [s.strip() for s in subjects.split(",")]
    plan = "Suggested study plan:\n"
    for i, s in enumerate(subject_list, 1):
        plan += f"{i}. {s} - 45 minutes\n"
    return plan

@tool
def summarizer(text: str) -> str:
    """Summarizes a block of text into key points."""
    response = llm.invoke(f"Summarize this in 2-3 bullet points:\n\n{text}")
    return response.content

def agents_demo():
    print("\n--- Agents & Tools (type 'back' to return) ---")
    agent = create_agent(llm, [calculator, study_planner, summarizer])
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "back":
            break
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print(f"\nAgent: {result['messages'][-1].content}\n")


# ----- Main Menu -----
def main():
    menu = {
        "1": ("Basic Chat", basic_chat),
        "2": ("Zero-Shot Prompting", zero_shot),
        "3": ("Few-Shot Quiz Generator", few_shot_quiz),
        "4": ("Chain-of-Thought Solver", chain_of_thought),
        "5": ("Role Prompting", role_prompting),
        "6": ("Prompt Templates", prompt_templates),
        "7": ("LangChain Pipeline", langchain_chains),
        "8": ("Memory Chat", memory_chat),
        "9": ("Agents & Tools", agents_demo),
    }

    print("=" * 50)
    print("       AI ACADEMIC TUTOR - MAIN MENU")
    print("=" * 50)

    while True:
        print("\nSelect a feature:")
        for key, (name, _) in menu.items():
            print(f"  {key}. {name}")
        print("  0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "0":
            print("Goodbye!")
            break

        if choice in menu:
            menu[choice][1]()
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()