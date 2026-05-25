import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag.loader import load_documents

VECTOR_DB_PATH = "./chroma_db"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )


def create_vector_store():
    docs = load_documents()
    embeddings = get_embeddings()

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )

    print("Vector DB created")


def load_vector_store():
    embeddings = get_embeddings()

    if not os.path.exists(VECTOR_DB_PATH):
        create_vector_store()

    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )