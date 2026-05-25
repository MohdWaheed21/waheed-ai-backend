import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rag.vector_store import load_vector_store

load_dotenv()

vectorstore = load_vector_store()


def get_llm():
    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-chat-v3-0324",
        temperature=0.4,
        timeout=30,
    )


def ask(question: str):
    try:
        llm = get_llm()

        docs = vectorstore.similarity_search(question, k=8)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are MD Waheed Pasha's official AI portfolio assistant.

ONLY answer questions about Waheed:
education, skills, projects, internships, certifications,
leadership, contact, achievements, portfolio.

If unrelated:
"I can only answer questions about Waheed."

Be professional and recruiter-friendly.

CONTEXT:
{context}

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