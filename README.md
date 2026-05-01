# Ask The Paper

An agentic RAG pipeline that lets you upload PDFs and ask multi-step research questions with cited answers.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│  Query Decomposer   │  LLM breaks complex question into sub-questions
└────────┬────────────┘
         │ sub-questions
         ▼
┌─────────────────────┐
│  Vector Retriever   │  ChromaDB similarity search per sub-question
└────────┬────────────┘
         │ raw chunks
         ▼
┌─────────────────────┐
│  Reranker           │  Dedup + sort by cosine similarity, keep top 8
└────────┬────────────┘
         │ ranked context
         ▼
┌─────────────────────┐
│  LLM Synthesizer    │  Claude generates answer with inline citations
│  + Memory           │  Full conversation history in context
└────────┬────────────┘
         │
         ▼
     Answer + Citations (streaming or batch)
```

**PDF Ingestion pipeline:**
```
PDF file → PyPDFLoader → RecursiveTextSplitter (600 tok / 100 overlap)
         → SentenceTransformer embeddings → ChromaDB upsert
         → S3 archive (optional)
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Jojo-Rabbit/AskThePaper
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY (get one at console.anthropic.com)

# 3. Run
cd app
python main.py
# Server at http://localhost:5000
```

## API Reference

### Upload PDFs
```bash
curl -X POST http://localhost:5000/upload \
  -F "files=@paper1.pdf" \
  -F "files=@paper2.pdf"

# Response:
# {
#   "results": [
#     { "file": "paper1.pdf", "chunks_indexed": 42, "doc_id": "a1b2c3d4", "status": "indexed" }
#   ]
# }
```

### Ask a question (batch)
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main contributions of this paper?", "session_id": "my-session"}'

# Response:
# {
#   "session_id": "my-session",
#   "answer": "The paper proposes... [Source: paper1.pdf, p.3]",
#   "sub_questions": ["What does the paper propose?", "What experiments were run?"],
#   "citations": [{ "source": "paper1.pdf", "page": 3, "score": 0.92 }]
# }
```

### Ask a question (streaming SSE)
```bash
curl -N -X POST http://localhost:5000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the methodology", "session_id": "my-session"}'

# Yields:
# data: {"type": "sub_questions", "content": ["..."]}
# data: {"type": "citations", "content": [...]}
# data: {"type": "token", "content": "The"}
# data: {"type": "token", "content": " methodology"}
# ...
# data: [DONE]
```

### Other endpoints
```bash
GET  /health              # { status, docs_indexed }
GET  /docs                # list all indexed PDFs
DELETE /sessions/:id      # clear conversation memory
```

## S3 Integration (optional)

Set in `.env`:
```
AWS_BUCKET=your-bucket-name
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

When configured, every uploaded PDF is archived to `s3://<bucket>/uploads/<uuid>_<filename>` before indexing. The app works fine without it — S3 is best-effort.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Design Decisions (for interviews)

**Why ChromaDB instead of Pinecone/Weaviate?**
ChromaDB runs locally with zero infrastructure — perfect for a portfolio project. In production I'd swap in a managed vector DB (Pinecone, pgvector on RDS) by changing the `VectorStore` class only — the agent doesn't care.

**Why decompose questions first?**
Single-step retrieval fails on compound questions ("compare X and Y" → needs two searches). Decomposing into sub-questions improves recall significantly at the cost of 1 extra LLM call.

**Why rerank after retrieval?**
Retrieval per sub-question returns overlapping chunks. Deduplication + score-based reranking ensures the LLM sees the most relevant, non-redundant context within the token budget.

**Why streaming?**
RAG synthesis takes 2-5 seconds. Streaming (SSE) lets the UI show tokens as they arrive, which feels instant. The `/ask` endpoint exists for simpler clients that don't support SSE.

**Why conversation memory?**
The full `_history` list is passed on every call. This means follow-up questions ("what about the second paper?") work correctly — the LLM has full context of what was already discussed.

**Why Anthropic over OpenAI?**
Claude's large context window handles long retrieved chunks well. Also relevant: Rox uses Claude-family models.
