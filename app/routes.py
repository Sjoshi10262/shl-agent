"""
FastAPI route definitions for the SHL Assessment Recommendation API.
"""

from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.rag import run_rag_pipeline
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Conversational SHL assessment recommendation endpoint.
    
    - Accepts full conversation history in messages[]
    - Returns reply, 0-10 grounded recommendations, and end_of_conversation flag
    - Stateless: all context must be in the messages array
    """
    try:
        # Validate message roles
        for msg in request.messages:
            if msg.role not in ("user", "assistant"):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid role '{msg.role}'. Must be 'user' or 'assistant'."
                )
        
        # Check at least one user message exists
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=422,
                detail="At least one user message is required."
            )
        
        # Convert to dicts for the pipeline
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Run RAG pipeline
        response = run_rag_pipeline(messages)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")
