"""
Data models for the SHL Assessment Recommendation API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


class AssessmentRecommendation(BaseModel):
    name: str
    url: str
    test_type: str = Field(..., description="Single or multi-value: 'K', 'P,C', 'A,S', etc.")


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[AssessmentRecommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
