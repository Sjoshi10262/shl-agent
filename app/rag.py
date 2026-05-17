"""
RAG Pipeline for SHL Assessment Recommendation.

Learned from 10 ground-truth conversations:
- OPQ32r = default personality layer (added proactively)
- SHL Verify Interactive G+ = default cognitive layer
- Comparison questions → text reply, unchanged recommendations
- Legal questions → redirect, no end_of_conversation
- Refinements → update shortlist without restarting
- SVAR → clarify accent (US/UK/AU/IN)
- No catalog match → state gap, propose proxies
"""

import json
import os
import re
import logging

from app.models import AssessmentRecommendation, ChatResponse
from app.prompts import SYSTEM_PROMPT, build_rag_prompt
from app.retriever import retriever

logger = logging.getLogger(__name__)

# ─── Injection Detection ─────────────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|directives?)",
    r"(pretend|act|behave)\s+(you are|as if|like)\s+(a\s+)?(different|new|another|unrestricted|dan)",
    r"\bjailbreak\b",
    r"\bdo anything now\b",
    r"\bDAN\b",
    r"you are now\s+(?!the|a\s+hiring)",
    r"forget\s+(your|all)\s+(instructions?|rules?|guidelines?|training)",
    r"disregard\s+(your|all)\s+(instructions?|rules?|constraints?)",
    r"(reveal|show|print|output)\s+(your\s+)?(system\s+)?prompt",
    r"override\s+(your\s+)?(safety|guidelines?|rules?)",
]

def _is_injection(text: str) -> bool:
    tl = text.lower()
    return any(re.search(p, tl) for p in INJECTION_PATTERNS)


# ─── Scope Guard ─────────────────────────────────────────────────────────────

OFF_TOPIC_PATTERNS = [
    r"\b(weather|recipe|cook|movie|music|sport|travel)\b",
    r"\b(write\s+code|debug\s+my|fix\s+my\s+bug)\b",
]

def _is_off_topic(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS)


# ─── Conversation Analyser ───────────────────────────────────────────────────

# Keywords that signal each test type
_TYPE_SIGNALS = {
    "P": ["personality", "behaviour", "behavior", "opq", "character", "cultural fit", "interpersonal"],
    "O": ["occupational personality"],
    "A": ["cognitive", "reasoning", "aptitude", "numerical", "verbal", "abstract", "inductive", "deductive", "iq", "g+", "verify g"],
    "K": ["knowledge", "technical", "coding", "programming", "java", "python", "sql", "spring", "aws",
          "docker", "javascript", "angular", "linux", "hipaa", "excel", "word", "rust", "networking"],
    "M": ["motivation", "motivator", "drive", "mq", "values"],
    "S": ["simulation", "inbox", "work sample", "call simulation", "svar", "spoken"],
    "B": ["situational judgment", "sjt", "scenarios", "graduate scenarios"],
    "C": ["competency", "competencies", "360", "ucf"],
    "D": ["development", "talent audit", "reskill", "re-skill", "l&d", "360 feedback"],
}

_LEVEL_SIGNALS = {
    "executive": ["cxo", "ceo", "cto", "cfo", "executive", "c-suite", "vp", "vice president", "c level", "c-level"],
    "senior": ["senior manager", "director", "head of", "15 years", "10+ years"],
    "manager": ["manager", "management", "supervisor", "team lead", "frontline manager"],
    "graduate": ["graduate", "fresh grad", "recent grad", "final year", "final-year", "no work experience", "trainee scheme"],
    "entry": ["entry", "entry-level", "junior", "frontline", "associate", "intern"],
    "professional": ["professional", "mid", "experienced", "senior engineer", "senior analyst", "senior ic", "ic"],
}


def _analyse_conversation(messages: list[dict]) -> dict:
    """Extract structured context from full conversation history."""
    all_user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()

    # Test type signals
    test_types = []
    for code, keywords in _TYPE_SIGNALS.items():
        if any(kw in all_user_text for kw in keywords):
            test_types.append(code)

    # Seniority
    job_level = None
    for level, keywords in _LEVEL_SIGNALS.items():
        if any(kw in all_user_text for kw in keywords):
            job_level = level
            break

    # Contact centre / SVAR signal
    is_contact_centre = any(w in all_user_text for w in
        ["contact cent", "contact center", "call cent", "call center", "inbound call", "customer service"])

    # Safety signal
    is_safety_role = any(w in all_user_text for w in
        ["safety", "chemical", "plant operator", "industrial", "manufacturing", "dependab"])

    # Graduate signal
    is_graduate = job_level == "graduate" or any(w in all_user_text for w in
        ["graduate", "trainee scheme", "final year", "no work experience"])

    # Sales/development signal
    is_sales = any(w in all_user_text for w in
        ["sales", "selling", "commercial", "revenue", "quota"])

    # Has enough info to retrieve?
    has_role = bool(re.search(
        r"\b(developer|engineer|analyst|manager|sales|customer|finance|hr|data|java|python|"
        r"javascript|accounting|marketing|operations|project|product|designer|consultant|"
        r"banker|nurse|healthcare|admin|assistant|operator|agent|trainee|executive|director|"
        r"specialist|scientist|devops|architect|lead|rust|go|kotlin|swift|ruby|php|perl)\b",
        all_user_text
    ))

    has_enough_info = (
        has_role
        or len(test_types) > 0
        or job_level is not None
        or is_contact_centre
        or is_safety_role
        or len(all_user_text.split()) > 12
    )

    # Build rich retrieval query from all user turns
    query = " ".join(m["content"] for m in messages if m["role"] == "user")

    return {
        "query": query,
        "has_enough_info": has_enough_info,
        "test_types": test_types,
        "job_level": job_level,
        "is_contact_centre": is_contact_centre,
        "is_safety_role": is_safety_role,
        "is_graduate": is_graduate,
        "is_sales": is_sales,
    }


# ─── Retrieval with Domain Boosting ──────────────────────────────────────────

def _retrieve(ctx: dict, top_k: int = 20) -> list[dict]:
    """Retrieve with domain-aware boosting so defaults always appear."""
    catalog = retriever.get_all()

    # Always retrieve broadly first
    results = retriever.retrieve(ctx["query"], top_k=top_k)
    result_names = {r["name"] for r in results}

    # Guarantee defaults are in the pool for professional/senior roles
    defaults_needed = []
    if ctx["job_level"] in ("professional", "manager", "senior", "executive", "graduate", None):
        for name in ["Occupational Personality Questionnaire OPQ32r", "SHL Verify Interactive G+"]:
            if name not in result_names:
                for a in catalog:
                    if a["name"] == name:
                        defaults_needed.append(a)
                        break

    # Domain boosting — ensure relevant items surface
    if ctx["is_contact_centre"]:
        for name in ["SVAR Spoken English (US) (New)", "Contact Center Call Simulation (New)",
                     "Entry Level Customer Serv - Retail & Contact Center",
                     "Customer Service Phone Simulation"]:
            if name not in result_names:
                for a in catalog:
                    if a["name"] == name:
                        results.append(a)
                        result_names.add(name)
                        break

    if ctx["is_safety_role"]:
        for name in ["Dependability and Safety Instrument (DSI)",
                     "Manufac. & Indust. - Safety & Dependability 8.0",
                     "Workplace Health and Safety (New)"]:
            if name not in result_names:
                for a in catalog:
                    if a["name"] == name:
                        results.append(a)
                        result_names.add(name)
                        break

    if ctx["is_graduate"]:
        for name in ["Graduate Scenarios", "SHL Verify Interactive G+"]:
            if name not in result_names:
                for a in catalog:
                    if a["name"] == name:
                        results.append(a)
                        result_names.add(name)
                        break

    if ctx["is_sales"]:
        for name in ["OPQ MQ Sales Report", "Sales Transformation 2.0 - Individual Contributor",
                     "Global Skills Assessment"]:
            if name not in result_names:
                for a in catalog:
                    if a["name"] == name:
                        results.append(a)
                        result_names.add(name)
                        break

    # Prepend guaranteed defaults
    results = defaults_needed + results

    return results[:top_k]


# ─── LLM Calls ───────────────────────────────────────────────────────────────

def _call_llm(system: str, user_prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        return _call_gemini(system, user_prompt)
    elif provider == "openai":
        return _call_openai(system, user_prompt)
    elif provider == "groq":
        return _call_groq(system, user_prompt)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _call_gemini(system: str, user_prompt: str) -> str:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )
    )
    return model.generate_content(user_prompt).text


def _call_openai(system: str, user_prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        temperature=0.1,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def _call_groq(system: str, user_prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        temperature=0.1,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


# ─── Response Parser + Grounder ──────────────────────────────────────────────

def _parse_and_ground(raw: str, catalog: list[dict]) -> ChatResponse:
    """Parse LLM JSON and ground every recommendation against the real catalog."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e} | raw[:300]={raw[:300]}")
        return ChatResponse(
            reply="I encountered an issue processing that. Could you rephrase your question?",
            recommendations=[],
            end_of_conversation=False,
        )

    reply = data.get("reply", "")
    end_of_conversation = bool(data.get("end_of_conversation", False))
    raw_recs = data.get("recommendations") or []

    # Build lookup indices
    by_url = {a["url"]: a for a in catalog}
    by_name = {a["name"].lower(): a for a in catalog}

    grounded = []
    for rec in raw_recs:
        name = rec.get("name", "")
        url  = rec.get("url", "")

        # Match by URL first (most precise)
        entry = by_url.get(url)

        # Match by exact name
        if not entry:
            entry = by_name.get(name.lower())

        # Fuzzy substring match
        if not entry:
            name_lower = name.lower()
            for key, a in by_name.items():
                if name_lower in key or key in name_lower:
                    entry = a
                    break

        if entry:
            grounded.append(AssessmentRecommendation(
                name=entry["name"],
                url=entry["url"],
                test_type=entry.get("test_type", rec.get("test_type", "K")),
            ))
        else:
            logger.warning(f"Could not ground: name={name!r} url={url!r}")

    return ChatResponse(
        reply=reply,
        recommendations=grounded[:10],
        end_of_conversation=end_of_conversation,
    )


# ─── Main Entry Point ────────────────────────────────────────────────────────

def run_rag_pipeline(messages: list[dict]) -> ChatResponse:
    """Full RAG pipeline: guard → analyse → retrieve → LLM → ground → return."""

    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    # 1. Injection guard
    if _is_injection(last_user):
        return ChatResponse(
            reply="I'm an SHL Assessment Consultant and can only help with assessment selection.",
            recommendations=[],
            end_of_conversation=False,
        )

    # 2. Scope guard
    if _is_off_topic(last_user):
        return ChatResponse(
            reply="I'm specialised in SHL assessment recommendations. What role are you looking to assess?",
            recommendations=[],
            end_of_conversation=False,
        )

    # 3. Analyse conversation
    ctx = _analyse_conversation(messages)

    # 4. Retrieve
    if ctx["has_enough_info"]:
        retrieved = _retrieve(ctx, top_k=20)
    else:
        # Still retrieve some context to help LLM ask informed questions
        retrieved = retriever.retrieve(ctx["query"], top_k=8)

    # 5. Build prompt and call LLM
    user_prompt = build_rag_prompt(messages, retrieved)
    try:
        raw = _call_llm(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ChatResponse(
            reply="I'm experiencing a temporary issue. Please try again in a moment.",
            recommendations=[],
            end_of_conversation=False,
        )

    # 6. Parse and ground
    catalog = retriever.get_all()
    return _parse_and_ground(raw, catalog)

# ─── Backwards-compat aliases (used by test_api.py) ──────────────────────────
_is_injection_attempt = _is_injection
_extract_query_context = _analyse_conversation
