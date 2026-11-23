import pytest
from unittest.mock import AsyncMock
from app.services.fallback import complete_with_fallback
from types import SimpleNamespace
from unittest.mock import AsyncMock
from app.services.cache import CACHE_TTL_SECONDS, cache_response

from app.services import llm_service


@pytest.mark.asyncio
async def test_generate_response(monkeypatch):

    mock_response = type(
        "MockResponse",
        (),
        {
            "choices": [
                type(
                    "MockChoice",
                    (),
                    {
                        "message": type(
                            "MockMessage",
                            (),
                            {"content": "Hello from the mock model"}
                        )()
                    }
                )()
            ]
        }
    )()

    mock_acompletion = AsyncMock(return_value=mock_response)

    monkeypatch.setattr(
        llm_service,
        "acompletion",
        mock_acompletion
    )

    result = await llm_service.chat_completions(
        model="ollama_chat/qwen2.5:1.5b",
        message="Hello"
    )

    assert result == "Hello from the mock model"
    mock_acompletion.assert_awaited_once()


@pytest.mark.asyncio
async def test_completion_uses_fallback_after_primary_failure():
    completion = AsyncMock(
        side_effect=[
            ConnectionError("primary unavailable"),
            {"choices": [{"message": {"content": "fallback response"}}]},
        ]
    )

    result = await complete_with_fallback(
        completion,
        model="ollama_chat/qwen2.5:1.5b",
        fallback_model="openrouter/test-model",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )

    assert result.fallback_used is True
    assert result.attempted_model == "openrouter/test-model"
    assert result.response["choices"][0]["message"]["content"] == "fallback response"
    assert completion.await_count == 2


@pytest.mark.asyncio
async def test_stream_response_skips_usage_only_chunks(monkeypatch):
    usage_chunk = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(total_tokens=12),
        model_dump=lambda: {"choices": [], "usage": {"total_tokens": 12}},
    )
    content_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content="Hello"))
        ],
        usage=None,
        model_dump=lambda: {
            "choices": [{"delta": {"content": "Hello"}}],
        },
    )

    async def mock_chunks():
        yield usage_chunk
        yield content_chunk

    async def mock_completion(**kwargs):
        return mock_chunks()

    monkeypatch.setattr(
        llm_service,
        "acompletion",
        mock_completion,
    )

    output = [
        item
        async for item in llm_service.stream_response(
            model="openrouter/test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )
    ]

    assert any("Hello" in item for item in output)
    assert output[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_cache_response_uses_full_ttl():
    redis = SimpleNamespace(set=AsyncMock())

    await cache_response(redis, "nexus:chat_cache:test", "cached answer")

    redis.set.assert_awaited_once_with(
        "nexus:chat_cache:test",
        "cached answer",
        ex=CACHE_TTL_SECONDS,
    )