"""HTTP API for the agent. Run with: uvicorn ragagent.api:app"""

from __future__ import annotations

import logging
from functools import lru_cache

import anthropic
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import RagAgent
from .config import Settings
from .factory import build_agent

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG agent", version="0.1.0")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    answer: str
    stop_reason: str
    retrieved_sources: list[str]
    tool_calls: int


@lru_cache(maxsize=1)
def get_agent() -> RagAgent:
    return build_agent(Settings.from_env())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest, agent: RagAgent = Depends(get_agent)) -> ChatResponse:
    try:
        answer = agent.answer(request.session_id, request.message)
    except anthropic.RateLimitError as error:
        raise HTTPException(429, "The model is rate limited. Try again shortly.") from error
    except anthropic.APIConnectionError as error:
        raise HTTPException(503, "The model could not be reached.") from error
    except anthropic.APIStatusError as error:
        logger.exception("model request failed with status %s", error.status_code)
        raise HTTPException(502, "The model returned an error.") from error
    return ChatResponse(
        answer=answer.text,
        stop_reason=answer.stop_reason,
        retrieved_sources=answer.sources,
        tool_calls=answer.tool_calls,
    )
