import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conversation_history = []

def chat(user_message):
    conversation_history.append({"role": "user", "parts": [{"text": user_message}]})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=conversation_history,
        config={"system_instruction": "You are a helpful academic tutor."}
    )

    reply = response.text
    conversation_history.append({"role": "model", "parts": [{"text": reply}]})
    return reply

def main():
    print("=== AI Academic Tutor ===")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if not user_input:
            continue
        response = chat(user_input)
        print(f"\nTutor: {response}\n")

if __name__ == "__main__":
    main()