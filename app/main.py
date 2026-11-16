from fastapi import FastAPI, Request
from app.api.chat import router as chat_router
from time import time
from app.logger import logger

app = FastAPI()

app.include_router(chat_router)

@app.middleware("http")
async def log_latency(request: Request, call_next):
    start_time = time()
    response = await call_next(request)
    process_time = time() - start_time
    logger.info(f"Request: {request.method} {request.url.path} completed in {process_time:.4f} seconds")
    return response

@app.get("/")
def health_check():
    return {"status": "ok"}
