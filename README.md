# AI Academic Tutor

A console-based AI tutor built with Python and the Gemini API, demonstrating core prompt engineering techniques and LangChain concepts (chains, memory, agents). Built as a lab project covering compiler design, software engineering, and prompting fundamentals.

## Features

- **Conversational tutoring** with persistent context across a session
- **Zero-shot prompting** for free-form topic explanations
- **Few-shot prompting** for structured quiz generation
- **Chain-of-Thought (CoT) prompting** for step-by-step math/logic problem solving
- **Role-based prompting** for persona-driven responses
- **Reusable prompt templates**
- **LangChain chains** for multi-step pipelines (topic → explanation → notes → quiz)
- **Memory system** to recall user info and past topics across the session
- **Agents & tools** that decide which tool (calculator, planner, summarizer) to use

## Tech Stack

- Python 3.14
- [Google Gemini API](https://aistudio.google.com) (`google-genai`) — `gemini-2.5-flash`
- LangChain (`langchain`, `langchain-google-genai`)
- `python-dotenv` for environment variable management

## Project Structure

```
ai-tutor/
├── .env                  # API keys (not committed)
├── .gitignore
├── requirements.txt
├── api.py                # Step 1: Core API integration + conversation history
├── zeroshot.py           # Step 2: Zero-shot prompting
├── fewshot.py            # Step 3: Few-shot prompting (quiz generator)
├── chainOfThought.py          # Step 4: Chain-of-Thought prompting
├── rolePrompting.py         # Step 5: Role prompting
├── promptTemplates.py     # Step 6: Prompt templates
├── chains.py        # Step 7: LangChain chains
├── memory.py        # Step 8: Memory system
├── agents.py        # Step 9: Agents & tools
└── main.py                # Step 10: Final integration
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/thekushalneupane/AI-Tutor.git
   cd AI-Tutor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
   Get a free key from [aistudio.google.com](https://aistudio.google.com).

5. Run any step:
   ```bash
   python api.py
   ```

## Progress

| Step | Topic | Status |
|------|-------|--------|
| 1 | LLM API Integration | ✅ Done |
| 2 | Zero-Shot Prompting | ✅ Done |
| 3 | Few-Shot Prompting | ✅ Done |
| 4 | Chain-of-Thought Prompting | ✅ Done |
| 5 | Role Prompting | ✅ Done |
| 6 | Prompt Templates | ✅ Done |
| 7 | LangChain Chains | ✅ Done |
| 8 | Memory System | ✅ Done |
| 9 | Agents & Tools | ✅ Done |
| 10 | Final Integration | ✅ Done |

## Notes

- Originally built with OpenAI's API, switched to Google Gemini (`gemini-2.5-flash`) due to free-tier availability.
- Migrated from the deprecated `google-generativeai` package to the new `google-genai` SDK.

## Author

Kushal Neupane — BCE, Semester 6
