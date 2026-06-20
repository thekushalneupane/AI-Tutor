import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Our memory store: a simple list of messages, plus extra tracked info
chat_history = []
student_info = {
    "name": None,
    "topics_studied": []
}

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful academic tutor. Use the conversation history to remember the student's name and what they've studied. Refer back to earlier topics when relevant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm


def remember_topic(user_input):
    # Very simple keyword tracking — if the user mentions "studying X" or "learn X"
    lowered = user_input.lower()
    if "my name is" in lowered:
        name = user_input.split("my name is")[-1].strip().split()[0]
        student_info["name"] = name.capitalize()


def chat(user_input):
    remember_topic(user_input)

    response = chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))

    return response.content


def main():
    print("=== Memory System Demo ===")
    print("Type 'quit' to exit, 'memory' to see what's remembered\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "memory":
            print(f"\nRemembered name: {student_info['name']}")
            print(f"Messages in history: {len(chat_history)}\n")
            continue

        if not user_input:
            continue

        response = chat(user_input)
        print(f"\nTutor: {response}\n")

if __name__ == "__main__":
    main()