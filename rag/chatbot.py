import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

with open("data/waheed_profile.json", "r", encoding="utf-8") as f:
    profile_data = json.load(f)

PROFILE_CONTEXT = json.dumps(profile_data, indent=2)


def get_llm():
    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-chat",
        temperature=0.4,
        timeout=30,
    )


def ask(question: str):
    try:
        llm = get_llm()

        prompt = f"""
You are MD Waheed Pasha's official AI portfolio assistant.

You represent Waheed professionally.

ONLY answer questions about:
- Waheed
- education
- skills
- projects
- internships
- certifications
- leadership
- hackathons
- achievements
- career goals
- contact
- portfolio
- experience

If unrelated:
"I can only answer questions about Waheed and his professional profile."

Be natural, professional, recruiter-friendly.

WAHEED PROFILE DATA:
{PROFILE_CONTEXT}

QUESTION:
{question}
"""

        response = llm.invoke(prompt)

        return {
            "result": response.content
        }

    except Exception as e:
        print("CHATBOT ERROR:", e)

        return {
            "result": "AI service temporarily unavailable."
        }


def get_chatbot():
    return ask