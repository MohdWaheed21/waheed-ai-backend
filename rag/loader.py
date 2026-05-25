import json
from langchain_core.documents import Document


def load_documents():
    with open("data/waheed_profile.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    def flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                flatten(value, f"{prefix}{key}: ")
        elif isinstance(obj, list):
            for item in obj:
                flatten(item, prefix)
        else:
            docs.append(Document(page_content=f"{prefix}{obj}"))

    flatten(data)

    return docs