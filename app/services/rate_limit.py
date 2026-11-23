# app/services/ratelimit.py
import time
from fastapi import Depends, Header, HTTPException, Request
from redis.asyncio import Redis
from redis.exceptions import RedisError

RATE_LIMIT_LUA = """
local current = redis.call("INCR", KEYS[1])
if current == 1 then
    redis.call("EXPIRE", KEYS[1], ARGV[1])
end
return current
"""

TPM_ADD_LUA = """
local current = redis.call("INCRBY", KEYS[1], ARGV[1])
if tonumber(current) == tonumber(ARGV[1]) then
    redis.call("EXPIRE", KEYS[1], ARGV[2])
end
return current
"""

async def get_redis(request: Request) -> Redis:
    return request.app.state.redis

def _bucket_key(prefix: str, api_key: str) -> str:
    minute_bucket = int(time.time() // 60)
    return f"{prefix}:{api_key}:{minute_bucket}"

async def check_rpm_limit(
    redis: Redis,
    api_key: str,
    rpm_limit: int = 30,
    ttl: int = 60,
) -> None:
    key = _bucket_key("nexus:rl:rpm", api_key)
    try:
        count = int(await redis.eval(RATE_LIMIT_LUA, 1, key, ttl))
    except RedisError:
        return  

    if count > rpm_limit:
        seconds_left = 60 - int(time.time() % 60)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {rpm_limit} requests per minute.",
            headers={"Retry-After": str(max(1, seconds_left))},
        )

async def check_tpm_limit(
    redis: Redis,
    api_key: str,
    tpm_limit: int = 100_000,
) -> None:
    key = _bucket_key("nexus:rl:tpm", api_key)
    try:
        current_tokens = await redis.get(key)
    except RedisError:
        return

    if current_tokens is not None and int(current_tokens) >= tpm_limit:
        seconds_left = 60 - int(time.time() % 60)
        raise HTTPException(
            status_code=429,
            detail=f"Token budget exceeded: max {tpm_limit} tokens per minute.",
            headers={"Retry-After": str(max(1, seconds_left))},
        )

async def record_tpm_usage(
    redis: Redis,
    api_key: str,
    tokens: int,
    ttl: int = 60,
) -> None:
    if tokens <= 0:
        return
    key = _bucket_key("nexus:rl:tpm", api_key)
    try:
        await redis.eval(TPM_ADD_LUA, 1, key, tokens, ttl)
    except RedisError:
        pass