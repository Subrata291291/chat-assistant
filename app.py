import os
import json
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="ChatGPT Clone", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False
    thread_id: Optional[str] = None
    thread_title: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    provider: str


conversation_memory: Dict[str, List[ChatMessage]] = {}
thread_titles: Dict[str, str] = {}


def sync_thread_conversation(thread_id: str, messages: List[ChatMessage], title: Optional[str] = None) -> List[ChatMessage]:
    if title:
        thread_titles[thread_id] = title

    trimmed = messages[-16:] if len(messages) > 16 else messages
    conversation_memory[thread_id] = trimmed
    return trimmed


def get_reply_from_groq(messages: List[ChatMessage]) -> tuple[str, str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        fallback = (
            "Groq API key not found. "
            "Set GROQ_API_KEY in your environment to use the real model. "
            "The demo is running in fallback mode."
        )
        return fallback, "fallback"

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": m.role, "content": m.content} for m in messages],
        temperature=0.7,
    )
    return completion.choices[0].message.content, "groq"


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    with open("templates/index.html", "r", encoding="utf-8") as handle:
        return HTMLResponse(content=handle.read())


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    last_user_message = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    if not last_user_message:
        raise HTTPException(status_code=400, detail="A user message is required")

    thread_id = request.thread_id or "default"
    conversation = sync_thread_conversation(thread_id, request.messages, request.thread_title)

    reply, provider = get_reply_from_groq(conversation)
    conversation.append(ChatMessage(role="assistant", content=reply))
    conversation_memory[thread_id] = conversation[-16:] if len(conversation) > 16 else conversation

    return ChatResponse(reply=reply, model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), provider=provider)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    thread_id = request.thread_id or "default"
    conversation = sync_thread_conversation(thread_id, request.messages, request.thread_title)

    reply, provider = get_reply_from_groq(conversation)
    conversation.append(ChatMessage(role="assistant", content=reply))
    conversation_memory[thread_id] = conversation[-16:] if len(conversation) > 16 else conversation

    def event_stream():
        words = reply.split()
        for index, word in enumerate(words):
            prefix = " " if index else ""
            chunk = prefix + word
            payload = json.dumps({"delta": chunk, "provider": provider})
            yield f"data: {payload}\n\n"
            time.sleep(0.05)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/threads")
def get_threads() -> List[Dict[str, str]]:
    result = []
    for thread_id, history in conversation_memory.items():
        preview = ""
        for message in reversed(history):
            if message.role == "user" and message.content.strip():
                preview = message.content.strip()
                break
        result.append({"id": thread_id, "title": thread_titles.get(thread_id, preview[:30] or "New chat"), "preview": preview})
    return result


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
