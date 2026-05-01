"""
Ask The Paper — Flask API
Agentic RAG pipeline: upload PDFs → chunk → embed → multi-step Q&A with citations
"""

import os
import uuid
import json
import boto3
import hashlib
from pathlib import Path
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

from agent import ResearchAgent
from vector_store import VectorStore
from config import Config

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

vector_store = VectorStore()
sessions: dict[str, ResearchAgent] = {}   # session_id → agent (conversation memory)

# ── S3 client (optional — falls back to local if not configured) ───────────────
def get_s3():
    if Config.AWS_BUCKET:
        return boto3.client(
            "s3",
            aws_access_key_id=Config.AWS_ACCESS_KEY,
            aws_secret_access_key=Config.AWS_SECRET_KEY,
            region_name=Config.AWS_REGION,
        )
    return None


def upload_to_s3(local_path: str, filename: str) -> str | None:
    s3 = get_s3()
    if not s3:
        return None
    key = f"uploads/{uuid.uuid4()}_{filename}"
    s3.upload_file(local_path, Config.AWS_BUCKET, key)
    return key


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "docs_indexed": vector_store.count()})


@app.route("/upload", methods=["POST"])
def upload():
    """Accept one or more PDFs, chunk + embed them, optionally archive to S3."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    results = []
    for f in request.files.getlist("files"):
        filename = secure_filename(f.filename)
        if not filename.endswith(".pdf"):
            results.append({"file": filename, "status": "skipped — not a PDF"})
            continue

        # Save locally
        local_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        f.save(local_path)

        # Archive to S3 (non-blocking — best effort)
        s3_key = upload_to_s3(local_path, filename)

        # Chunk + embed
        doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]
        chunk_count = vector_store.ingest_pdf(local_path, doc_id=doc_id, source=filename)

        results.append({
            "file": filename,
            "doc_id": doc_id,
            "chunks_indexed": chunk_count,
            "s3_key": s3_key,
            "status": "indexed",
        })

    return jsonify({"results": results})


@app.route("/ask", methods=["POST"])
def ask():
    """
    Non-streaming Q&A endpoint.
    Body: { "question": "...", "session_id": "optional" }
    """
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    session_id = body.get("session_id") or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = ResearchAgent(vector_store)

    agent = sessions[session_id]
    result = agent.run(question)
    return jsonify({"session_id": session_id, **result})


@app.route("/ask/stream", methods=["POST"])
def ask_stream():
    """
    Streaming Q&A — yields Server-Sent Events so the UI can show tokens as they arrive.
    Body: { "question": "...", "session_id": "optional" }
    """
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    session_id = body.get("session_id") or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = ResearchAgent(vector_store)

    agent = sessions[session_id]

    def event_stream():
        for chunk in agent.stream(question):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"X-Session-Id": session_id},
    )


@app.route("/sessions/<session_id>", methods=["DELETE"])
def clear_session(session_id):
    """Clear conversation memory for a session."""
    sessions.pop(session_id, None)
    return jsonify({"status": "cleared"})


@app.route("/docs", methods=["GET"])
def list_docs():
    """List all indexed documents."""
    return jsonify({"documents": vector_store.list_sources()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
