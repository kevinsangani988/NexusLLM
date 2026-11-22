import pytest
from unittest.mock import AsyncMock
from app.services.fallback import complete_with_fallback

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