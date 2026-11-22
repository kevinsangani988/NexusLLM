import pytest
from unittest.mock import AsyncMock

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