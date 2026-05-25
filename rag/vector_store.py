import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.loader import load_documents

VECTOR_DB_PATH = "faiss_index"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store():
    docs = load_documents()
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    vectorstore.save_local(VECTOR_DB_PATH)

    print("JSON Vector DB created successfully")


def load_vector_store():
    embeddings = get_embeddings()

    if not os.path.exists(VECTOR_DB_PATH):
        create_vector_store()

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )