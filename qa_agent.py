#!/usr/bin/env python3
"""
qa_agent.py — CLI Q&A agent over Gutermann's product_overview.md

Pipeline:
  1. Ingestion (runs once at startup)
     - parse markdown
     - chunk it by product (### headings), carrying along the shared
       category intro text (## section) each product lives under
     - embed each chunk, cache the vectors to disk (keyed by a hash of
       the source file so edits invalidate the cache automatically)
  2. Retrieval (per query)
     - embed the query
     - hybrid score = cosine(embedding) + keyword-overlap bonus
       (pure cosine under-weights exact spec terms like "plastic" /
       "long distances" when several chunks are topically similar —
       see README "Discussion" section, query 4)
  3. Generation
     - retrieved chunks -> strict "answer only from context" prompt
     - low-confidence retrieval short-circuits to an explicit
       "not enough info" answer instead of letting the LLM guess
  4. CLI loop — type a question, "exit"/"quit" to stop

Usage:
    python qa_agent.py                       # uses product_overview.md
    python qa_agent.py --file other.md
    python qa_agent.py --rebuild-index        # force re-embedding
    python qa_agent.py --top-k 5
    python qa_agent.py --no-sources           # hide the [Source: ...] line

Configure the LLM via environment variables (see .env.example):
    LLM_PROVIDER = anthropic | openai | ollama   (auto-detected if unset)
    ANTHROPIC_API_KEY / OPENAI_API_KEY as needed
    EMBEDDING_BACKEND = tfidf (default, zero setup) | sbert (better recall,
        needs `pip install sentence-transformers` + a one-time model download)
"""

import argparse
import hashlib
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

CACHE_DIR = Path(".qa_cache")
CACHE_FILE = CACHE_DIR / "index.pkl"

CONFIDENCE_FLOOR = {
    "tfidf": 0.08,
    "sbert": 0.20,
}

NOT_ENOUGH_INFO = "The document doesn't contain enough information to answer that."


# --------------------------------------------------------------------------
# 1. Ingestion: parsing + chunking
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    id: int
    category: str          # the ## section it lives under
    product: str           # the ### heading (product name), or "" for a
                            # category-level chunk with no sub-product
    text: str              # embeddable text: category intro + product body

    def label(self) -> str:
        return f"{self.category} > {self.product}" if self.product else self.category


_IMG_MARKDOWN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_COMMENT = re.compile(r"<!--.*?-->")
_BULLET = re.compile(r"^\s*[•\-]\s*")


def _strip_noise(line: str) -> str:
    """Remove inline image markdown and HTML comments (mineru export
    artifacts like '![img_005](...) <!-- image_id:... -->'), which can
    share a line with nothing else meaningful."""
    line = _IMG_MARKDOWN.sub("", line)
    line = _HTML_COMMENT.sub("", line)
    return line.strip()


def _clean_line(line: str) -> str:
    line = _BULLET.sub("", line.rstrip())
    return line.strip()


def parse_markdown_to_chunks(md_text: str) -> List[Chunk]:
    """
    Splits the document into one chunk per product (### heading).

    Each chunk's text is the *category intro* (any prose/bullets that sit
    directly under a ## heading, before the first ###) followed by the
    product's own bullets. This matters for e.g. "## Permanent Leak
    Detection Monitoring", where shared facts like "no drilling required"
    and "NB-IoT" live in the intro, not under ZONESCAN AI / ZONESCAN HYDRO
    individually — a query about "no drilling" would miss those products
    entirely if we dropped the shared intro.

    A handful of section labels in the source aren't real markdown
    headings (e.g. "Acoustic Leak Detection Microphones" is a bare line,
    not "## ..."). We detect these heuristically: a short, non-bullet,
    unpunctuated line that is immediately followed (skipping blanks/
    images/comments) by a "### " line is treated as a category header.
    """
    raw_lines = md_text.split("\n")

    # Pre-scan to find "soft" category headings.
    def next_meaningful(idx: int) -> Optional[str]:
        for j in range(idx + 1, len(raw_lines)):
            candidate = _strip_noise(raw_lines[j])
            if not candidate:
                continue
            return candidate
        return None

    chunks: List[Chunk] = []
    category = "General"
    intro_lines: List[str] = []
    product: Optional[str] = None
    product_lines: List[str] = []
    chunk_id = 0

    def flush_product():
        nonlocal product, product_lines, chunk_id
        if product is None and not product_lines:
            return
        body = "\n".join(l for l in product_lines if l.strip())
        intro = "\n".join(l for l in intro_lines if l.strip())
        core = (intro + "\n" + body).strip() if intro else body.strip()
        # Prepend product/category name so exact-name queries ("AQUASCAN 610")
        # retrieve correctly — product names live in metadata, not the body
        # bullets, so a pure body-text embedding would otherwise never match them.
        header = f"{product}. Category: {category}." if product else category
        text = f"{header}\n{core}".strip() if core else header
        if text:
            chunks.append(Chunk(id=chunk_id, category=category,
                                 product=product or "", text=text))
            chunk_id += 1
        product = None
        product_lines = []

    for i, raw in enumerate(raw_lines):
        stripped = _strip_noise(raw)

        if not stripped:
            continue

        if stripped.startswith("### "):
            flush_product()
            product = stripped[4:].strip()
            continue

        if stripped.startswith("## "):
            flush_product()
            category = stripped[3:].strip()
            intro_lines = []
            continue

        if stripped.startswith("# "):
            # Top-level document title — not product content, skip.
            continue

        is_soft_heading = (
            len(stripped) < 60
            and not stripped.endswith((".", ":", ")"))
            and not stripped.startswith(("•", "-"))
            and (next_meaningful(i) or "").startswith("### ")
        )
        if is_soft_heading:
            flush_product()
            category = stripped
            intro_lines = []
            continue

        cleaned = _clean_line(stripped)
        if not cleaned:
            continue
        if product is not None:
            product_lines.append(cleaned)
        else:
            intro_lines.append(cleaned)

    flush_product()
    return chunks


# --------------------------------------------------------------------------
# 2. Embeddings
# --------------------------------------------------------------------------

class Embedder:
    """Pluggable embedding backend: 'tfidf' (default, no download, works
    offline) or 'sbert' (sentence-transformers, better semantic recall).
    Both expose the same fit()/transform() shape so the rest of the
    pipeline doesn't care which one is active."""

    def __init__(self, backend: str = "tfidf"):
        self.backend = backend
        if backend == "sbert":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        elif backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        else:
            raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if self.backend == "sbert":
            return np.asarray(self.model.encode(texts, normalize_embeddings=True))
        matrix = self.vectorizer.fit_transform(texts)
        return _l2_normalize(matrix.toarray())

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.backend == "sbert":
            return np.asarray(self.model.encode(texts, normalize_embeddings=True))
        matrix = self.vectorizer.transform(texts)
        return _l2_normalize(matrix.toarray())


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# --------------------------------------------------------------------------
# Vector store + hybrid retrieval
# --------------------------------------------------------------------------

class VectorStore:
    def __init__(self, chunks: List[Chunk], vectors: np.ndarray, embedder: Embedder):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder

    def search(self, query: str, top_k: int = 4, hybrid_alpha: float = 0.75):
        """Returns [(chunk, score)] sorted by descending hybrid score.

        hybrid_alpha weights cosine similarity vs. a plain keyword-overlap
        bonus. Pure cosine similarity over whole-chunk embeddings tends to
        reward chunks that are broadly on-topic but can under-rank the
        one chunk that satisfies *every* specific term in a multi-constraint
        query (e.g. "plastic pipes" AND "long distances") — see README.
        """
        q_vec = self.embedder.transform([query])[0]
        cos_scores = self.vectors @ q_vec

        q_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        kw_scores = []
        for c in self.chunks:
            c_terms = set(re.findall(r"[a-z0-9]+", c.text.lower()))
            overlap = len(q_terms & c_terms) / max(1, len(q_terms))
            kw_scores.append(overlap)
        kw_scores = np.array(kw_scores)

        combined = hybrid_alpha * cos_scores + (1 - hybrid_alpha) * kw_scores
        order = np.argsort(-combined)[:top_k]
        return [(self.chunks[i], float(combined[i])) for i in order]


# --------------------------------------------------------------------------
# Ingestion cache (persist embeddings to a local file)
# --------------------------------------------------------------------------

def build_or_load_index(md_path: Path, backend: str, rebuild: bool) -> VectorStore:
    md_text = md_path.read_text(encoding="utf-8")
    file_hash = hashlib.sha256(md_text.encode("utf-8")).hexdigest()

    if not rebuild and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                cached = pickle.load(f)
            if cached.get("hash") == file_hash and cached.get("backend") == backend:
                embedder = cached["embedder"]
                return VectorStore(cached["chunks"], cached["vectors"], embedder)
        except Exception:
            pass  # fall through and rebuild on any cache read issue

    chunks = parse_markdown_to_chunks(md_text)
    if not chunks:
        raise RuntimeError("No chunks were parsed from the markdown file — check its structure.")

    embedder = Embedder(backend)
    vectors = embedder.fit_transform([c.text for c in chunks])

    CACHE_DIR.mkdir(exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"hash": file_hash, "backend": backend,
                     "chunks": chunks, "vectors": vectors, "embedder": embedder}, f)

    return VectorStore(chunks, vectors, embedder)


# --------------------------------------------------------------------------
# 3. Generation
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a product-knowledge assistant for Gutermann's water leak \
detection equipment catalogue.

Rules (follow strictly):
- Answer ONLY using the CONTEXT provided below. Never use outside/general knowledge \
about these or similar products.
- If the context does not state something explicitly, do not infer or assume it. \
This includes availability, pricing, release/order status, and technical specs that \
are not written down.
- If the context does not contain enough information to answer, respond with exactly: \
"{not_enough_info}"
- Be concise and specific. Name the exact product(s) involved.
- If asked to compare products, only compare using facts present in the context.
""".format(not_enough_info=NOT_ENOUGH_INFO)


def _build_user_prompt(query: str, hits) -> str:
    context_blocks = []
    for chunk, score in hits:
        context_blocks.append(f"[{chunk.label()}]\n{chunk.text}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer using only the CONTEXT above."


def _detect_provider() -> str:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider:
        return provider
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "ollama"  # last resort: assumes a local Ollama server


def call_llm(system: str, user: str, provider: str) -> str:
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
        resp = client.messages.create(
            model=model, max_tokens=500, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                       {"role": "user", "content": user}],
            max_tokens=500, temperature=0,
        )
        return resp.choices[0].message.content.strip()

    if provider == "ollama":
        import requests
        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": f"{system}\n\n{user}", "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def answer_query(store: VectorStore, query: str, top_k: int, provider: str,
                  show_sources: bool) -> str:
    hits = store.search(query, top_k=top_k)
    floor = CONFIDENCE_FLOOR.get(store.embedder.backend, 0.1)
    if not hits or hits[0][1] < floor:
        return NOT_ENOUGH_INFO

    system = SYSTEM_PROMPT
    user = _build_user_prompt(query, hits)
    try:
        answer = call_llm(system, user, provider)
    except Exception as e:
        return (f"[LLM call failed: {e}]\n"
                f"Configure LLM_PROVIDER / API keys — see .env.example.")

    if show_sources:
        sources = ", ".join(sorted({c.label() for c, _ in hits}))
        answer = f"{answer}\n[Source: {sources}]"
    return answer


# --------------------------------------------------------------------------
# 4. CLI loop
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CLI Q&A agent over product_overview.md")
    parser.add_argument("--file", default="product_overview.md", help="Path to the markdown knowledge base")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rebuild-index", action="store_true", help="Ignore cache and re-embed")
    parser.add_argument("--no-sources", action="store_true", help="Hide the [Source: ...] line")
    parser.add_argument("--backend", default=os.environ.get("EMBEDDING_BACKEND", "tfidf"),
                         choices=["tfidf", "sbert"])
    args = parser.parse_args()

    md_path = Path(args.file)
    if not md_path.exists():
        print(f"Error: file not found: {md_path}", file=sys.stderr)
        sys.exit(1)

    provider = _detect_provider()
    print(f"[qa_agent] embedding backend: {args.backend} | LLM provider: {provider}")

    print("[qa_agent] ingesting and indexing...", end=" ", flush=True)
    store = build_or_load_index(md_path, args.backend, args.rebuild_index)
    print(f"done ({len(store.chunks)} chunks).")
    print("Ask a question about the product catalogue. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        answer = answer_query(store, query, args.top_k, provider, not args.no_sources)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
