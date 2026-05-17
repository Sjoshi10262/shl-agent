"""
SHL Assessment Recommendation Agent — FastAPI Application

A production-ready conversational AI system for recommending SHL Individual
Test Solutions from the SHL Product Catalog.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router
from app.retriever import retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    logger.info("Initializing SHL Assessment Retriever...")
    try:
        retriever.load()
        logger.info("Retriever loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load retriever: {e}")
    yield
    logger.info("Shutting down SHL Assessment Agent.")


app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    description=(
        "A conversational AI agent that recommends SHL Individual Test Solutions "
        "from the SHL Product Catalog based on hiring manager requirements."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
