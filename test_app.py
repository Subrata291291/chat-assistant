from app import get_reply_from_groq


def test_groq_fallback_when_no_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    reply, provider = get_reply_from_groq([])
    assert "Groq API key not found" in reply
    assert provider == "fallback"
