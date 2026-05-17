# SHL Assessment Recommendation Agent

A production-ready conversational AI system for recommending SHL Individual Test Solutions. Built with FastAPI, FAISS vector search, and a configurable LLM backend (Gemini/OpenAI/Groq).

---

## Architecture

```
User Query
    ↓
FastAPI /chat
    ↓
Injection & Scope Guard
    ↓
Conversation Analyzer (extract role, seniority, test type needs)
    ↓
FAISS Vector Retriever (sentence-transformers/all-MiniLM-L6-v2)
    ↓
LLM Grounded Response (Gemini / OpenAI / Groq)
    ↓
JSON Schema Validator + Catalog Grounder
    ↓
Structured ChatResponse
```

---

## Project Structure

```
shl-agent/
├── app/
│   ├── main.py        # FastAPI app, lifespan startup
│   ├── routes.py      # /health and /chat endpoints
│   ├── rag.py         # RAG pipeline, injection guard, LLM calls
│   ├── retriever.py   # FAISS vector index builder and searcher
│   ├── prompts.py     # System prompt + RAG prompt builder
│   ├── models.py      # Pydantic request/response models
│   └── utils.py       # Helper utilities
├── data/
│   └── shl_catalog.json   # SHL Individual Test Solutions catalog
├── vectorstore/           # Auto-generated FAISS index cache
├── tests/
│   └── test_api.py        # pytest test suite
├── scraper.py             # SHL catalog web scraper
├── requirements.txt
├── Dockerfile
├── render.yaml            # Render.com deployment config
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone and Set Up

```bash
git clone <your-repo>
cd shl-agent
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API key
```

**Choose your LLM:**

| Provider | Speed | Cost | Key |
|----------|-------|------|-----|
| Gemini | Fast | Free tier | `GEMINI_API_KEY` |
| Groq | Fastest | Free tier | `GROQ_API_KEY` |
| OpenAI | High quality | Paid | `OPENAI_API_KEY` |

### 3. Run the Server

```bash
python -m uvicorn app.main:app --reload --port 8000
```

The first startup builds the FAISS vector index (~5 seconds). Subsequent startups use the cache.

---

## API Reference

### GET /health

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

### POST /chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to hire a Java developer"}
    ]
  }'
```

Response:
```json
{
  "reply": "Here are some assessments for a Java developer role. Do you also need cognitive or personality assessments?",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    },
    {
      "name": "Verify G+ Cognitive Ability",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-g-plus-cognitive-ability/",
      "test_type": "A"
    }
  ],
  "end_of_conversation": false
}
```

---

## Example Conversations

### Vague Query → Clarification
```json
Request: {"messages": [{"role": "user", "content": "I need an assessment"}]}

Response: {
  "reply": "I'd be happy to help! Could you tell me: (1) What role are you hiring for? (2) Is this an entry-level or experienced position?",
  "recommendations": [],
  "end_of_conversation": false
}
```

### Specific Role
```json
Request: {"messages": [{"role": "user", "content": "Hiring a senior sales manager"}]}

Response: {
  "reply": "For a senior sales manager, I recommend these assessments...",
  "recommendations": [
    {"name": "OPQ32 Personality Questionnaire", "url": "...", "test_type": "P"},
    {"name": "Sales Achievement Predictor (SAP)", "url": "...", "test_type": "P"},
    {"name": "Verify Numerical Reasoning", "url": "...", "test_type": "A"}
  ],
  "end_of_conversation": true
}
```

### Multi-Turn with Refinement
```json
Messages: [
  {"role": "user", "content": "Hiring a Python data scientist"},
  {"role": "assistant", "content": "Here are Python and cognitive tests..."},
  {"role": "user", "content": "Actually add personality tests too"}
]
```

### Comparison Query
```json
Messages: [{"role": "user", "content": "What's the difference between OPQ and the Motivation Questionnaire?"}]
```

---

## Test Type Codes

| Code | Meaning |
|------|---------|
| A | Ability & Aptitude |
| B | Biodata & Situational Judgment |
| C | Competencies |
| D | Development & 360 |
| E | Assessment Exercises |
| K | Knowledge & Skills |
| M | Motivation & Preferences |
| O | Occupational Personality |
| P | Personality & Behavior |
| S | Simulations |

---

## Running the Scraper

To refresh the catalog from the live SHL website:

```bash
python scraper.py
```

This overwrites `data/shl_catalog.json`. Re-run the server to rebuild the vector index.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Deployment

### Render.com (Recommended — Free Tier)

1. Push to GitHub
2. Connect repo to Render
3. Use `render.yaml` (already configured)
4. Add `GEMINI_API_KEY` in Render environment variables
5. Deploy

### Docker

```bash
docker build -t shl-agent .
docker run -p 8000:8000 \
  -e LLM_PROVIDER=gemini \
  -e GEMINI_API_KEY=your_key \
  shl-agent
```

### Railway / Fly.io

Both support Docker-based deployment — use the included `Dockerfile`.

---

## Design Decisions

### Why FAISS over ChromaDB?
FAISS is faster for small-medium catalogs, has no external service dependency, and is deployment-friendly (pure Python/numpy).

### Why stateless?
The API receives full `messages[]` history each request. No session state needed. This scales horizontally and keeps the API simple.

### Hallucination Prevention
All recommendations are grounded: the LLM's output is cross-referenced against the catalog by name and URL before returning. Fake recommendations are silently dropped.

### Injection Defense
Regex patterns catch common jailbreak attempts before they reach the LLM.

---

## Extending the Catalog

Edit `data/shl_catalog.json` following this schema:

```json
{
  "name": "Assessment Name",
  "url": "https://www.shl.com/solutions/products/product-catalog/view/...",
  "description": "What the assessment measures",
  "skills": ["skill1", "skill2"],
  "duration": "20-30 minutes",
  "test_type": "K",
  "job_levels": ["Entry", "Mid-Professional"],
  "languages": ["English"]
}
```

Then restart the server to rebuild the vector index.
