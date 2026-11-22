import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from redis.asyncio import Redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_timeout=0.5,        
        socket_connect_timeout=0.5,
    )
    yield
    await app.state.redis.aclose()

async def get_redis(request: Request) -> Redis:
    return request.app.state.redis