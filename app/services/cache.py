import json
import hashlib
from redis.asyncio import Redis
from redis.exceptions import RedisError

async def build_cache_key(message: str, model: str, temperature: float ) -> str:
    payload = {
        "message": message,
        "model": model,
        "temperature": temperature
    }
    payload_str = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    return f"nexus:chat_cache:{payload_hash}"

async def cache_response(redis: Redis, key: str, response: str, expiration_seconds: int = 3600) -> None:
    try:
        await redis.set(key, response, ex=expiration_seconds)
    except RedisError as e:
        print(f"Redis error while caching response: {e}")

async def get_cached_response(redis: Redis, key: str) -> str | None:
    try:
        cached_response = await redis.get(key)
        if cached_response:
            if isinstance(cached_response, bytes):
                return cached_response.decode("utf-8")
            return cached_response

    except RedisError as e:
        print(f"Redis error while getting cached response: {e}")
    return None
