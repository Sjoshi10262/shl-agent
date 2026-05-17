"""
Utility functions for the SHL Assessment Recommendation Agent.
"""

import json
import os
import re
from typing import Any


def load_json(path: str) -> Any:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def clean_json_response(raw: str) -> str:
    """Strip markdown fences and whitespace from LLM JSON output."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars, preserving word boundaries."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def count_user_turns(messages: list[dict]) -> int:
    """Count number of user turns in conversation."""
    return sum(1 for m in messages if m.get("role") == "user")


def get_last_user_message(messages: list[dict]) -> str:
    """Get the most recent user message content."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""
