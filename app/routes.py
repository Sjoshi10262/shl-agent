"""
FastAPI route definitions for the SHL Assessment Recommendation API.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.rag import run_rag_pipeline
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    return JSONResponse({
        "name": "SHL Assessment Recommendation Agent",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "chat": "POST /chat",
            "docs": "GET /docs"
        }
    })


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        for msg in request.messages:
            if msg.role not in ("user", "assistant"):
                raise HTTPException(status_code=422, detail=f"Invalid role '{msg.role}'.")
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=422, detail="At least one user message is required.")
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        response = run_rag_pipeline(messages)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")