from fastapi import APIRouter
from pydantic import BaseModel
from rag.chatbot import get_chatbot

router = APIRouter()

chatbot = get_chatbot()


class ChatRequest(BaseModel):
    question: str


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat")
def chat(request: ChatRequest):
    response = chatbot(request.question)

    return {
        "answer": response["result"]
    }