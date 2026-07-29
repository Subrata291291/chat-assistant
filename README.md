# ChatGPT-Style AI Assistant (Python)

This project is a fast, Python-first prototype for a ChatGPT-style assistant that can be demonstrated in a one-hour webinar.

## Features
- FastAPI backend
- Simple browser-based chat UI
- Conversation memory using in-memory history
- Streaming token-style response rendering
- Fallback mode when no Groq API key is configured

## Run locally

```bash
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/ in your browser.

## Optional: real model
Create a .env file in the project root and add:

```env
GROQ_API_KEY=<api key>
GROQ_MODEL=openai/gpt-oss-120b
```

Then restart the app to use the real model.
