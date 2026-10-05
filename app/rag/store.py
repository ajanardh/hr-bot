"""Local hashed TF-IDF index. Seeded, dependency-free, and small enough for a free tier."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

from app.config import EMBED_DIM, SEED, index_path
from app.rag.ingest import chunk_document, load_corpus

TOKEN = re.compile(r"[a-z0-9]+")
STOP = {
    "a", "an", "the", "of", "to", "for", "and", "or", "in", "on", "my", "i", "is", "are",
    "what", "how", "does", "do", "can", "with", "about", "from", "me", "we", "our", "your",
    "their", "this", "that", "be", "if", "at", "by", "it", "as", "am", "was", "were", "be",
    "please", "would", "could", "should", "into", "over", "under", "than", "then", "also",
}
EXPANSIONS = {
    "pto": ("paid", "time", "off", "vacation", "leave"),
    "vacation": ("pto", "leave"),
    "accrue": ("accrual", "accrued"),
    "accrual": ("accrue", "accrued"),
    "reimburse": ("reimbursement", "reimbursable"),
    "reimbursement": ("reimburse", "reimbursable"),
    "remote": ("hybrid",),
    "hybrid": ("remote",),
    "vpn": ("globalprotect", "network"),
    "mfa": ("multifactor", "authentication", "password"),
    "401k": ("retirement", "match"),
    "stipend": ("home", "office"),
}


def tokenize(text: str) -> list[str]:
    lowered = text.lower().replace("401(k)", "401k").replace("p&c", "people and culture")
    return TOKEN.findall(lowered)


def _bucket(token: str) -> tuple[int, int]:
    digest = hashlib.sha256(f"{SEED}:{token}".encode()).digest()
    index = int.from_bytes(digest[:4], "big") % EMBED_DIM
    sign = 1 if digest[4] % 2 == 0 else -1
    return index, sign


def _embed(tokens: list[str], idf: dict[str, float]) -> list[float]:
    vector = [0.0] * EMBED_DIM
    if not tokens:
        return vector
    counts = Counter(tokens)
    for token, count in counts.items():
        index, sign = _bucket(token)
        weight = (1.0 + math.log(count)) * idf.get(token, 1.0)
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def _expand(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(EXPANSIONS.get(token, ()))
    return expanded


def build_index() -> dict:
    documents = load_corpus()
    chunks: list[dict] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    document_frequency: Counter[str] = Counter()
    tokenized: list[list[str]] = []
    for chunk in chunks:
        tokens = tokenize(chunk["text"])
        tokenized.append(tokens)
        document_frequency.update(set(tokens))
    total = max(len(chunks), 1)
    idf = {
        token: math.log((total + 1) / (freq + 1)) + 1.0
        for token, freq in document_frequency.items()
    }
    stored = []
    for chunk, tokens in zip(chunks, tokenized):
        stored.append(
            {
                **chunk,
                "vector": [round(value, 6) for value in _embed(tokens, idf)],
            }
        )
    payload = {
        "seed": SEED,
        "dim": EMBED_DIM,
        "chunk_count": len(stored),
        "document_count": len(documents),
        "idf": {key: round(value, 6) for key, value in idf.items()},
        "chunks": stored,
    }
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)
    return payload


def load_index() -> dict:
    path = index_path()
    if not path.exists():
        return build_index()
    return json.loads(path.read_text())


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def search(query: str, k: int = 4, document_id: str | None = None) -> list[dict]:
    """Hybrid retrieval: seeded hashed TF-IDF plus lexical overlap, then a light rerank."""
    index = load_index()
    query_tokens = tokenize(query)
    expanded = _expand(query_tokens)
    query_vector = _embed(expanded, index["idf"])
    query_terms = set(query_tokens) or set(expanded)
    ranked: list[tuple[float, dict]] = []
    for chunk in index["chunks"]:
        if document_id and chunk["document_id"] != str(document_id):
            continue
        chunk_terms = set(tokenize(chunk["text"]))
        overlap = len(query_terms & chunk_terms)
        lexical = overlap / max(len(query_terms), 1)
        heading_terms = {token for token in tokenize(f"{chunk['title']} {chunk['section']}") if not token.isdigit() and len(token) > 2}
        query_for_heading = {token for token in query_terms if not token.isdigit() and len(token) > 2}
        heading_boost = 0.12 * len(query_for_heading & heading_terms)
        score = (0.62 * _cosine(query_vector, chunk["vector"])) + (0.38 * lexical) + heading_boost
        if score <= 0:
            continue
        ranked.append((score, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    # Rerank the candidate pool with a second pass that prefers heading matches.
    pool = ranked[: max(k * 3, k)]
    pool.sort(
        key=lambda item: (
            item[0]
            + (
                0.08
                if {token for token in tokenize(query) if not token.isdigit() and len(token) > 2}
                & {token for token in tokenize(item[1]["section"]) if not token.isdigit() and len(token) > 2}
                else 0
            )
        ),
        reverse=True,
    )
    hits = []
    for score, chunk in pool[:k]:
        hits.append(
            {
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "section": chunk["section"],
                "source": chunk["source"],
                "format": chunk["format"],
                "snippet": chunk["snippet"],
                "text": chunk["text"],
                "score": round(score, 4),
                "chunk_id": chunk["id"],
            }
        )
    return hits


def missing_terms(query: str) -> list[str]:
    index = load_index()
    vocabulary = set(index["idf"])
    missing = []
    for token in tokenize(query):
        if token in STOP or len(token) < 3:
            continue
        if token not in vocabulary and token not in EXPANSIONS:
            missing.append(token)
    return missing


def get_section(document_id: str, section: str, limit: int = 2) -> dict:
    index = load_index()
    document_id = str(document_id)
    known = sorted({chunk["document_id"] for chunk in index["chunks"]})
    if document_id not in known:
        return {"error": "unknown_document", "document_id": document_id, "available_documents": known}
    needle = section.lower().strip()
    needle_tokens = [token for token in tokenize(section) if token not in STOP and token != document_id and len(token) > 2]
    scored: list[tuple[int, dict]] = []
    for chunk in index["chunks"]:
        if chunk["document_id"] != document_id:
            continue
        haystack = tokenize(f"{chunk['section']}\n{chunk['text']}")
        overlap = len(set(needle_tokens) & set(haystack))
        if needle and needle in chunk["section"].lower():
            overlap += 5
        if overlap or not needle_tokens:
            scored.append((overlap, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    selected = [chunk for _, chunk in scored[:limit]]
    return {
        "document_id": document_id,
        "section_query": section,
        "citations": [
            {
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "section": chunk["section"],
                "source": chunk["source"],
                "format": chunk["format"],
                "snippet": chunk["snippet"],
                "text": chunk["text"],
                "score": 1.0,
                "chunk_id": chunk["id"],
            }
            for chunk in selected
        ],
    }
