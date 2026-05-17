"""
Retriever module — builds vector index from SHL catalog and retrieves
relevant assessments based on semantic similarity.

Supports two backends:
  1. sentence-transformers (preferred, requires internet/model download)
  2. TF-IDF with sklearn (fallback, works offline)
"""

import json
import os
import pickle
import re
import math
import logging
import numpy as np
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)

CATALOG_PATH = "data/shl_catalog.json"
EMBEDDINGS_PATH = "vectorstore/embeddings.npy"
CATALOG_CACHE_PATH = "vectorstore/catalog_cache.pkl"
TFIDF_PATH = "vectorstore/tfidf.pkl"


def _build_document_text(assessment: dict) -> str:
    """Build a rich text representation of an assessment for embedding."""
    parts = [
        assessment.get("name", ""),
        assessment.get("name", ""),  # double weight name
        assessment.get("description", ""),
        f"Skills: {', '.join(assessment.get('skills', []))}",
        f"Job levels: {', '.join(assessment.get('job_levels', []))}",
        f"Test type: {assessment.get('test_type', '')}",
    ]
    return " | ".join(p for p in parts if p.strip())


# ─── TF-IDF Backend (no internet required) ───────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_tfidf(documents: list[str]):
    """Build TF-IDF matrix from documents."""
    tokenized = [_tokenize(d) for d in documents]
    
    # Build vocabulary
    vocab = {}
    for tokens in tokenized:
        for t in set(tokens):
            vocab[t] = vocab.get(t, 0) + 1
    
    # IDF
    N = len(documents)
    idf = {term: math.log(N / (1 + df)) for term, df in vocab.items()}
    
    # Build TF-IDF vectors
    idx_to_term = list(vocab.keys())
    term_to_idx = {t: i for i, t in enumerate(idx_to_term)}
    
    matrix = np.zeros((N, len(idx_to_term)), dtype=np.float32)
    for doc_i, tokens in enumerate(tokenized):
        tf = Counter(tokens)
        total = len(tokens) or 1
        for term, count in tf.items():
            if term in term_to_idx:
                j = term_to_idx[term]
                matrix[doc_i, j] = (count / total) * idf[term]
    
    # L2 normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms
    
    return {"matrix": matrix, "term_to_idx": term_to_idx, "idf": idf}


def _tfidf_query(query: str, tfidf: dict) -> np.ndarray:
    """Convert query to TF-IDF vector."""
    tokens = _tokenize(query)
    term_to_idx = tfidf["term_to_idx"]
    idf = tfidf["idf"]
    
    vec = np.zeros(len(term_to_idx), dtype=np.float32)
    tf = Counter(tokens)
    total = len(tokens) or 1
    
    for term, count in tf.items():
        if term in term_to_idx:
            j = term_to_idx[term]
            vec[j] = (count / total) * idf.get(term, 0)
    
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ─── Main Retriever Class ─────────────────────────────────────────────────────

class AssessmentRetriever:
    def __init__(self):
        self.catalog: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.tfidf: Optional[dict] = None
        self.model = None
        self.backend: str = "none"
        self._loaded = False

    def load(self):
        """Load catalog and build/restore vector index."""
        if self._loaded:
            return

        # Load catalog
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        os.makedirs("vectorstore", exist_ok=True)
        texts = [_build_document_text(a) for a in self.catalog]

        # Try sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer
            
            # Check cache
            if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(CATALOG_CACHE_PATH):
                with open(CATALOG_CACHE_PATH, "rb") as f:
                    cached_catalog = pickle.load(f)
                if len(cached_catalog) == len(self.catalog):
                    self.embeddings = np.load(EMBEDDINGS_PATH)
                    self.backend = "sentence-transformers (cached)"
                    logger.info(f"[Retriever] Loaded {len(self.catalog)} cached ST embeddings.")
                    self._loaded = True
                    return

            logger.info("[Retriever] Loading sentence-transformers model...")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            np.save(EMBEDDINGS_PATH, self.embeddings)
            with open(CATALOG_CACHE_PATH, "wb") as f:
                pickle.dump(self.catalog, f)
            self.backend = "sentence-transformers"
            logger.info("[Retriever] ST embeddings built and cached.")

        except Exception as e:
            logger.warning(f"[Retriever] sentence-transformers unavailable ({e}), falling back to TF-IDF.")
            
            # TF-IDF fallback
            if os.path.exists(TFIDF_PATH):
                with open(TFIDF_PATH, "rb") as f:
                    self.tfidf = pickle.load(f)
                logger.info("[Retriever] TF-IDF index loaded from cache.")
            else:
                logger.info("[Retriever] Building TF-IDF index...")
                self.tfidf = _build_tfidf(texts)
                with open(TFIDF_PATH, "wb") as f:
                    pickle.dump(self.tfidf, f)
                logger.info("[Retriever] TF-IDF index built and cached.")
            
            self.backend = "tfidf"

        self._loaded = True

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[dict] = None) -> list[dict]:
        """
        Retrieve top-k assessments by semantic/lexical similarity.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional dict with keys like 'test_types', 'job_level'

        Returns:
            List of assessment dicts, sorted by relevance
        """
        if not self._loaded:
            self.load()

        if not query.strip():
            return self.catalog[:top_k]

        # Compute scores
        if self.embeddings is not None and self.model is not None:
            query_vec = self.model.encode([query], normalize_embeddings=True)[0]
            scores = self.embeddings @ query_vec
        elif self.embeddings is not None:
            # Cached ST embeddings — rebuild model lazily
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                query_vec = self.model.encode([query], normalize_embeddings=True)[0]
                scores = self.embeddings @ query_vec
            except Exception:
                scores = np.zeros(len(self.catalog))
        elif self.tfidf is not None:
            query_vec = _tfidf_query(query, self.tfidf)
            scores = self.tfidf["matrix"] @ query_vec
        else:
            scores = np.zeros(len(self.catalog))

        # Apply filters
        if filters:
            for i, assessment in enumerate(self.catalog):
                if "test_types" in filters and filters["test_types"]:
                    if assessment.get("test_type") not in filters["test_types"]:
                        scores[i] = -1.0
                if "job_level" in filters and filters["job_level"]:
                    levels = [l.lower() for l in assessment.get("job_levels", [])]
                    if not any(filters["job_level"].lower() in l for l in levels):
                        scores[i] *= 0.7

        # Sort
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.catalog[idx] for idx in top_indices if scores[idx] >= 0]

    def get_all(self) -> list[dict]:
        """Return full catalog."""
        if not self._loaded:
            self.load()
        return self.catalog


# Singleton instance
retriever = AssessmentRetriever()
