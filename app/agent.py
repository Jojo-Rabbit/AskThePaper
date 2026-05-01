"""
ResearchAgent — the core agentic loop.

Pipeline per query:
  1. Decompose   — break complex question into sub-questions
  2. Retrieve    — vector search for each sub-question
  3. Rerank      — deduplicate + score chunks
  4. Synthesize  — LLM generates answer with inline citations
  5. Memory      — append turn to session history

Streaming variant yields token chunks via a generator.
"""

from __future__ import annotations
import anthropic
from typing import Any, Generator

from vector_store import VectorStore
from config import Config

_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

# ── Prompts ───────────────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM = """You are a research query planner.
Given a user question, output a JSON array of 1-3 focused sub-questions that together cover the original question.
Output ONLY the JSON array, nothing else. Example:
["What is X?", "How does X relate to Y?"]"""

SYNTHESIZE_SYSTEM = """You are a precise research assistant. You answer questions using ONLY the provided context chunks.

Rules:
- Cite sources inline as [Source: <filename>, p.<page>]
- If the context is insufficient, say so clearly — do not hallucinate
- Be concise but complete
- If multiple chunks support a point, cite all of them"""

# ── Agent ─────────────────────────────────────────────────────────────────────

class ResearchAgent:
    def __init__(self, vector_store: VectorStore):
        self._vs    = vector_store
        self._history: list[dict] = []   # conversation memory

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, question: str) -> dict[str, Any]:
        """Full pipeline, returns dict with answer + citations."""
        sub_questions = self._decompose(question)
        chunks        = self._retrieve_all(sub_questions)
        context       = self._build_context(chunks)
        answer        = self._synthesize(question, context)

        self._history.append({"role": "user",      "content": question})
        self._history.append({"role": "assistant", "content": answer})

        return {
            "answer":         answer,
            "sub_questions":  sub_questions,
            "citations":      self._extract_citations(chunks),
        }

    def stream(self, question: str) -> Generator[dict, None, None]:
        """Streaming pipeline — yields { type, content } dicts."""
        sub_questions = self._decompose(question)
        yield {"type": "sub_questions", "content": sub_questions}

        chunks  = self._retrieve_all(sub_questions)
        context = self._build_context(chunks)
        yield {"type": "citations", "content": self._extract_citations(chunks)}

        full_answer = ""
        with _client.messages.stream(
            model=Config.MODEL,
            max_tokens=Config.MAX_TOKENS,
            system=SYNTHESIZE_SYSTEM,
            messages=[
                *self._history,
                {"role": "user", "content": self._user_prompt(question, context)},
            ],
        ) as stream:
            for text in stream.text_stream:
                full_answer += text
                yield {"type": "token", "content": text}

        self._history.append({"role": "user",      "content": question})
        self._history.append({"role": "assistant", "content": full_answer})

    # ── Steps ──────────────────────────────────────────────────────────────────

    def _decompose(self, question: str) -> list[str]:
        """Step 1 — break complex question into focused sub-questions."""
        import json
        resp = _client.messages.create(
            model=Config.MODEL,
            max_tokens=200,
            system=DECOMPOSE_SYSTEM,
            messages=[{"role": "user", "content": question}],
        )
        try:
            sub_qs = json.loads(resp.content[0].text)
            return sub_qs if isinstance(sub_qs, list) else [question]
        except (json.JSONDecodeError, IndexError):
            return [question]

    def _retrieve_all(self, sub_questions: list[str]) -> list[dict]:
        """Step 2 — retrieve chunks for each sub-question, then deduplicate."""
        seen, chunks = set(), []
        for q in sub_questions:
            for chunk in self._vs.search(q, top_k=Config.TOP_K_RETRIEVAL):
                key = chunk["text"][:80]   # dedup by content prefix
                if key not in seen:
                    seen.add(key)
                    chunks.append(chunk)
        # Step 3 — rerank by score descending, keep top 8
        chunks.sort(key=lambda c: c["score"], reverse=True)
        return chunks[:8]

    def _synthesize(self, question: str, context: str) -> str:
        """Step 4 — LLM synthesis with conversation memory."""
        resp = _client.messages.create(
            model=Config.MODEL,
            max_tokens=Config.MAX_TOKENS,
            system=SYNTHESIZE_SYSTEM,
            messages=[
                *self._history,
                {"role": "user", "content": self._user_prompt(question, context)},
            ],
        )
        return resp.content[0].text

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Chunk {i} | {c['source']} p.{c['page']} | score={c['score']}]\n{c['text']}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _user_prompt(question: str, context: str) -> str:
        return f"Context:\n{context}\n\nQuestion: {question}"

    @staticmethod
    def _extract_citations(chunks: list[dict]) -> list[dict]:
        seen, citations = set(), []
        for c in chunks:
            key = (c["source"], c["page"])
            if key not in seen:
                seen.add(key)
                citations.append({"source": c["source"], "page": c["page"], "score": c["score"]})
        return citations
