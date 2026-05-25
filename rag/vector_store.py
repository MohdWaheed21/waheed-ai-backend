import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rag.loader import load_documents

VECTOR_DB_PATH = "./chroma_db"


def get_embeddings():
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="text-embedding-3-small",
    )


def create_vector_store():
    docs = load_documents()
    embeddings = get_embeddings()

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )

    print("Chroma vector DB created")


def load_vector_store():
    embeddings = get_embeddings()

    if not os.path.exists(VECTOR_DB_PATH):
        create_vector_store()

    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )