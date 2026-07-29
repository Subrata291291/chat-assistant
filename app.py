import os
import json
import time
import traceback
from typing import List, Dict, Optional

from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

print("=" * 60)
print("Starting Chat Assistant...")
print("GROQ_API_KEY Loaded:", bool(os.getenv("GROQ_API_KEY")))
print("GROQ_MODEL:", os.getenv("GROQ_MODEL"))
print("=" * 60)

app = FastAPI(
    title="ChatGPT Clone",
    version="1.0.0"
)

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


def sync_thread_conversation(
    thread_id: str,
    messages: List[ChatMessage],
    title: Optional[str] = None,
):
    if title:
        thread_titles[thread_id] = title

    conversation = messages[-16:]
    conversation_memory[thread_id] = conversation
    return conversation


def get_reply_from_groq(messages: List[ChatMessage]):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("ERROR: GROQ_API_KEY NOT FOUND")
        return (
            "Groq API key is missing. Please configure GROQ_API_KEY in Render.",
            "fallback",
        )

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    print(f"Using model: {model}")

    try:

        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": m.role,
                    "content": m.content,
                }
                for m in messages
            ],
            temperature=0.7,
        )

        reply = completion.choices[0].message.content

        print("Groq Reply Generated Successfully")

        return reply, "groq"

    except Exception as e:

        print("=" * 60)
        print("GROQ ERROR")
        traceback.print_exc()
        print("=" * 60)

        return (
            f"Groq Error: {str(e)}",
            "error",
        )


@app.get("/", response_class=HTMLResponse)
def root():

    with open(
        "templates/index.html",
        "r",
        encoding="utf-8",
    ) as f:
        return HTMLResponse(f.read())


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    print("/chat called")

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages cannot be empty",
        )

    thread_id = request.thread_id or "default"

    conversation = sync_thread_conversation(
        thread_id,
        request.messages,
        request.thread_title,
    )

    reply, provider = get_reply_from_groq(conversation)

    conversation.append(
        ChatMessage(
            role="assistant",
            content=reply,
        )
    )

    conversation_memory[thread_id] = conversation[-16:]

    return ChatResponse(
        reply=reply,
        model=os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
        ),
        provider=provider,
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    print("/chat/stream called")

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages cannot be empty",
        )

    thread_id = request.thread_id or "default"

    conversation = sync_thread_conversation(
        thread_id,
        request.messages,
        request.thread_title,
    )

    reply, provider = get_reply_from_groq(conversation)

    conversation.append(
        ChatMessage(
            role="assistant",
            content=reply,
        )
    )

    conversation_memory[thread_id] = conversation[-16:]

    def event_stream():

        try:

            words = reply.split()

            for i, word in enumerate(words):

                payload = json.dumps(
                    {
                        "delta": (" " if i else "") + word,
                        "provider": provider,
                    }
                )

                yield f"data: {payload}\n\n"

                time.sleep(0.03)

            yield "event: done\ndata: {}\n\n"

        except Exception:

            traceback.print_exc()

            yield (
                'data: {"delta":"Streaming Error"}\n\n'
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@app.get("/threads")
def get_threads():

    result = []

    for thread_id, history in conversation_memory.items():

        preview = ""

        for msg in reversed(history):

            if msg.role == "user":

                preview = msg.content

                break

        result.append(
            {
                "id": thread_id,
                "title": thread_titles.get(
                    thread_id,
                    preview[:30] or "New chat",
                ),
                "preview": preview,
            }
        )

    return result


@app.get("/health")
def health():

    return {
        "status": "ok",
        "groq_key_loaded": bool(os.getenv("GROQ_API_KEY")),
        "model": os.getenv("GROQ_MODEL"),
    }