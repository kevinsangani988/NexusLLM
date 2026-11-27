import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_chat_endpoint(monkeypatch):

    async def mock_generate_response(model, message, *args, **kwargs):
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


def test_semantic_cache_hit_for_message_request(monkeypatch):
    observed = {}

    async def cache_miss(redis, key):
        return None

    async def semantic_hit(model, messages):
        observed["model"] = model
        observed["messages"] = messages
        return "semantic answer"

    monkeypatch.setattr("app.api.chat.get_cached_response", cache_miss)
    monkeypatch.setattr("app.api.chat.lookup_semantic_cache", semantic_hit)

    response = client.post(
        "/v1/chat",
        json={
            "model": "ollama_chat/qwen2.5:1.5b",
            "message": "A similar prompt",
            "temperature": 0.1,
        },
    )

    assert response.status_code == 200
    assert response.json()["cache_type"] == "semantic"
    assert observed["messages"] == [
        {"role": "user", "content": "A similar prompt"}
    ]


def test_messages_request_is_stored_in_both_caches(monkeypatch):
    observed = {}

    async def cache_miss(redis, key):
        return None

    async def semantic_miss(model, messages):
        return None

    async def mock_chat_completions(request, *, redis, cache_key):
        return {
            "choices": [{"message": {"content": "generated answer"}}],
        }

    async def mock_cache_response(redis, key, response):
        observed["exact_response"] = response

    async def mock_store_semantic(model, messages, response):
        observed["semantic_messages"] = messages
        observed["semantic_response"] = response

    monkeypatch.setattr("app.api.chat.get_cached_response", cache_miss)
    monkeypatch.setattr("app.api.chat.lookup_semantic_cache", semantic_miss)
    monkeypatch.setattr("app.api.chat._chat_completions", mock_chat_completions)
    monkeypatch.setattr("app.api.chat.cache_response", mock_cache_response)
    monkeypatch.setattr("app.api.chat.store_semantic_cache", mock_store_semantic)

    response = client.post(
        "/v1/chat",
        json={
            "model": "ollama_chat/qwen2.5:1.5b",
            "messages": [{"role": "user", "content": "A prompt"}],
            "temperature": 0.1,
        },
    )

    assert response.status_code == 200
    assert response.json()["cache_type"] == "none"
    assert observed["exact_response"] == "generated answer"
    assert observed["semantic_messages"] == [
        {"role": "user", "content": "A prompt"}
    ]
    assert observed["semantic_response"] == "generated answer"


def test_streaming_semantic_hit_reports_cache_type(monkeypatch):
    async def cache_miss(redis, key):
        return None

    async def semantic_hit(model, messages):
        return "semantic answer"

    monkeypatch.setattr("app.api.chat.get_cached_response", cache_miss)
    monkeypatch.setattr("app.api.chat.lookup_semantic_cache", semantic_hit)

    response = client.post(
        "/v1/chat",
        json={
            "model": "ollama_chat/qwen2.5:1.5b",
            "message": "A similar prompt",
            "temperature": 0.1,
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["x-cache-type"] == "semantic"
    assert "semantic answer" in response.text


def test_exact_cache_key_includes_full_message_history(monkeypatch):
    cache_keys = []

    async def cache_miss(redis, key):
        cache_keys.append(key)
        return None

    async def semantic_miss(model, messages):
        return None

    async def mock_chat_completions(request, *, redis, cache_key):
        return {"choices": [{"message": {"content": "generated answer"}}]}

    async def mock_cache_response(redis, key, response):
        return None

    async def mock_store_semantic(model, messages, response):
        return None

    monkeypatch.setattr("app.api.chat.get_cached_response", cache_miss)
    monkeypatch.setattr("app.api.chat.lookup_semantic_cache", semantic_miss)
    monkeypatch.setattr("app.api.chat._chat_completions", mock_chat_completions)
    monkeypatch.setattr("app.api.chat.cache_response", mock_cache_response)
    monkeypatch.setattr("app.api.chat.store_semantic_cache", mock_store_semantic)

    for history in (
        [{"role": "user", "content": "Earlier context A"}],
        [{"role": "user", "content": "Earlier context B"}],
    ):
        response = client.post(
            "/v1/chat",
            json={
                "model": "ollama_chat/qwen2.5:1.5b",
                "messages": history + [{"role": "user", "content": "Same question"}],
                "temperature": 0.1,
            },
        )
        assert response.status_code == 200

    assert len(cache_keys) == 2
    assert cache_keys[0] != cache_keys[1]