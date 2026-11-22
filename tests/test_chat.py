import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_chat_endpoint(monkeypatch):

    async def mock_generate_response(model, message):
        return "A transformer uses self-attention."

    monkeypatch.setattr(
        "app.api.chat.chat",
        mock_generate_response
    )

    response = client.post(
        "/v1/chat",
        json={
            "model": "ollama_chat/qwen2.5:1.5b",
            "message": "Explain transformers."
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "ollama_chat/qwen2.5:1.5b"
    assert data["response"] == "A transformer uses self-attention."
    assert "request_id" in data
    assert len(data["request_id"]) > 0

def test_chat_invalid_request():

    response = client.post(
        "/v1/chat",
        json={
            "model": "ollama_chat/qwen2.5:1.5b"
        }
    )

    assert response.status_code == 422