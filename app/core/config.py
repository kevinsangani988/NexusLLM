from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NexusLLM Gateway"
    ollama_base_url: str = "http://localhost:11434"
    default_timeout_seconds: float = 60.0

    # Logical model aliases mapped to LiteLLM provider strings
    model_registry: dict[str, str] = {
        "local-small": "ollama_chat/qwen2.5:1.5b",
        "openrouter": "openrouter/nvidia/nemotron-3.5-super-vl:free",
    }
    fallback_registry: dict[str, str] = {
        "ollama_chat/qwen2.5:1.5b": "openrouter/thinkingmachines/inkling-small",
        "openrouter/thinkingmachines/inkling-small": "ollama_chat/qwen2.5:1.5b",
        "openrouter/nvidia/nemotron-3.5-super-vl:free": "openrouter/thinkingmachines/inkling-small",
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
settings = Settings()