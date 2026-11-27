import os
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Tag
from redisvl.utils.vectorize import HFTextVectorizer
from redis.exceptions import RedisError
from app.logger import logger

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

vectorizer = HFTextVectorizer("sentence-transformers/all-MiniLM-L6-v2")

semantic_cache = SemanticCache(
    name="nexus_semantic_cache",
    redis_url=REDIS_URL,
    vectorizer=vectorizer,
    distance_threshold=0.12,
    filterable_fields=[{"name": "model", "type": "tag"}],
    ttl=3600,
)

def _latest_user_prompt(messages: list[dict]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return None

async def lookup_semantic_cache(model: str, messages: list[dict]) -> str | None:
    prompt = _latest_user_prompt(messages)
    if not prompt:
        return None
    try:
        hits = await semantic_cache.acheck(
            prompt=prompt,
            filter_expression=(Tag("model") == model),
        )
        if hits:
            return hits[0]["response"]
    except RedisError:
        logger.exception("Redis error while looking up semantic cache")
        return None
    return None

async def store_semantic_cache(model: str, messages: list[dict], response_text: str) -> None:
    prompt = _latest_user_prompt(messages)
    if not prompt or not response_text:
        return
    try:
        await semantic_cache.astore(
            prompt=prompt,
            response=response_text,
            filters={"model": model},
        )
    except RedisError:
        logger.exception("Redis error while storing semantic cache")