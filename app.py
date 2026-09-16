"""
RAG (Retrieval-Augmented Generation) system — HYBRID RETRIEVAL
--------------------------------------------------------------
Python 3.14 compatible - NO torch, NO sentence-transformers.

Retrieval strategy: HYBRID
  - FAISS (semantic / dense)
  - BM25 (keyword / sparse)
  - Reciprocal Rank Fusion (RRF) to merge results

- LLM:        Groq API (openai/gpt-oss-120b / 20b / qwen/qwen3.8-27b)
- Embeddings: HuggingFace Inference API (HTTP calls, no local model weights)
- Vector DB:  FAISS (pure CPU, persisted to disk)
- Loaders:    PDF (pypdf), DOCX (python-docx), TXT, CSV (stdlib csv)

Run:
    pip install -r requirements.txt
    cp .env.example .env
    python app.py
"""

import os
import csv
import json
import shutil
import pickle
import traceback
from io import StringIO

from flask import (
    Flask, request, jsonify, render_template_string,
    send_from_directory, Response, stream_with_context
)
from dotenv import load_dotenv

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from rank_bm25 import BM25Okapi
import requests

from source_attribution import (
    build_corpus_idf, filter_relevant_sources
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "8"))

UPLOAD_DIR = "uploaded_docs"
INDEX_DIR = "faiss_index"
BM25_INDEX_FILE = "bm25_index.pkl"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

# --------------------------------------------------------------------------
# Embeddings (HF Inference API — no torch)
# --------------------------------------------------------------------------
class HFInferenceAPIEmbeddingsFallback(Embeddings):
    def __init__(self, model_name, api_token, timeout=60):
        if not api_token:
            raise ValueError("HF_API_TOKEN required")
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.timeout = timeout

    @staticmethod
    def _mean_pool(vectors):
        n = len(vectors); dim = len(vectors[0])
        summed = [0.0] * dim
        for vec in vectors:
            for i, v in enumerate(vec):
                summed[i] += v
        return [s / n for s in summed]

    def _normalize_output(self, item):
        if len(item) == 0:
            raise ValueError("Empty embedding")
        first = item[0]
        if isinstance(first, list):
            return self._mean_pool(item)
        return item

    def _call_api(self, texts):
        r = requests.post(self.api_url, headers=self.headers,
                          json={"inputs": texts, "options": {"wait_for_model": True}},
                          timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"HF API error {r.status_code}: {r.text[:500]}")
        return r.json()

    def embed_documents(self, texts):
        if not texts: return []
        results = []
        bs = 8
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            data = self._call_api(batch)
            if isinstance(data, list) and len(batch) == 1 and len(data) > 0 and isinstance(data[0], (int, float)):
                results.append(data)
            else:
                for item in data:
                    results.append(self._normalize_output(item))
        return results

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def get_embeddings_model():
    try:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        emb = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            huggingfacehub_api_token=HF_API_TOKEN,
        )
        emb.embed_query("connection test")
        print("[INFO] Using HuggingFaceEndpointEmbeddings")
        return emb
    except Exception as e:
        print(f"[WARN] HuggingFaceEndpointEmbeddings failed ({e}); using fallback")
        return HFInferenceAPIEmbeddingsFallback(EMBEDDING_MODEL, HF_API_TOKEN)


embeddings_model = None
vectorstore = None
bm25_index = None
bm25_corpus = None   # list of (text, source) tuples
corpus_idf = None    # dict of {token: idf_weight}

USE_LLM_RERANK = os.getenv("USE_LLM_RERANK", "false").lower() == "true"


def ensure_embeddings():
    global embeddings_model
    if embeddings_model is None:
        embeddings_model = get_embeddings_model()
    return embeddings_model


def load_existing_index():
    """Load FAISS + BM25 from disk if present."""
    global vectorstore, bm25_index, bm25_corpus
    if os.path.isdir(INDEX_DIR) and os.listdir(INDEX_DIR):
        try:
            emb = ensure_embeddings()
            vectorstore = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
            print("[INFO] Loaded existing FAISS index")
        except Exception as e:
            print(f"[WARN] FAISS load failed: {e}")
            vectorstore = None

    if os.path.isfile(BM25_INDEX_FILE):
        try:
            with open(BM25_INDEX_FILE, "rb") as f:
                saved = pickle.load(f)
            bm25_index = saved["bm25"]
            bm25_corpus = saved["corpus"]
            print(f"[INFO] Loaded BM25 index with {len(bm25_corpus)} chunks")
            global corpus_idf
            corpus_idf = build_corpus_idf([t for t, _ in bm25_corpus])
        except Exception as e:
            print(f"[WARN] BM25 load failed: {e}")
            bm25_index = None
            bm25_corpus = None


# --------------------------------------------------------------------------
# Document loaders
# --------------------------------------------------------------------------
def load_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)

def load_docx(path):
    import docx
    doc = docx.Document(path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            rt = " | ".join(cell.text for cell in row.cells)
            if rt.strip(): paras.append(rt)
    return "\n".join(paras)

def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_csv(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.reader(f))
    if not rows: return ""
    header, *body = rows
    lines = [", ".join(header)]
    for row in body:
        lines.append(" | ".join(f"{h}: {v}" for h, v in zip(header, row)))
    return "\n".join(lines)

LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt, ".csv": load_csv}


def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return LOADERS[ext](path)


def tokenize(text):
    """Simple lowercase word tokenizer for BM25."""
    import re
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks):
    """chunks is list of (text, source). Returns (bm25, chunks)."""
    tokenized = [tokenize(text) for text, _ in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

INDEX_HTML_PATH = os.path.join("static", "rag_dashboard.html")


@app.route("/")
def index():
    return send_from_directory("static", "rag_dashboard.html")


@app.route("/upload", methods=["POST"])
def upload():
    global vectorstore, bm25_index, bm25_corpus

    if not HF_API_TOKEN:
        return jsonify({"error": "HF_API_TOKEN not set in .env"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    all_chunks = []       # list of (text, source)
    skipped = []
    preview_map = {}      # filename -> first ~200 chars of extracted text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for file in files:
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped.append(filename); continue

        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        try:
            text = extract_text(save_path)
        except Exception as e:
            skipped.append(f"{filename} (extract failed: {e})"); continue

        if not text.strip():
            skipped.append(f"{filename} (no text)"); continue

        # Capture preview (first ~200 chars, whitespace collapsed)
        preview = " ".join(text[:400].split())[:200]
        preview_map[filename] = preview

        for chunk in splitter.split_text(text):
            all_chunks.append((chunk, filename))

    if not all_chunks:
        return jsonify({"error": "No text extracted", "skipped": skipped}), 400

    try:
        emb = ensure_embeddings()
        docs = [Document(page_content=t, metadata={"source": s}) for t, s in all_chunks]

        # FAISS — append to existing or create new
        if vectorstore is None:
            vectorstore = FAISS.from_documents(docs, emb)
        else:
            vectorstore.add_documents(docs)
        vectorstore.save_local(INDEX_DIR)

        # BM25 — append to existing or create new
        if bm25_corpus is None:
            bm25_corpus = list(all_chunks)
        else:
            bm25_corpus = bm25_corpus + list(all_chunks)
        bm25_index, bm25_corpus = build_bm25_index(bm25_corpus)

        with open(BM25_INDEX_FILE, "wb") as f:
            pickle.dump({"bm25": bm25_index, "corpus": bm25_corpus}, f)

        # Build / refresh IDF weights for source attribution
        global corpus_idf
        corpus_idf = build_corpus_idf([t for t, _ in bm25_corpus])

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Indexing failed: {e}"}), 500

    return jsonify({
        "message": f"Indexed {len(all_chunks)} chunks from {len(files) - len(skipped)} file(s) with HYBRID search.",
        "skipped": skipped,
        "batch_files": len(files) - len(skipped),
        "batch_chunks": len(all_chunks),
        "total_chunks": len(bm25_corpus) if bm25_corpus else 0,
        "retrieval": "hybrid (FAISS + BM25 + RRF)",
        "previews": preview_map,
    })


def compute_confidence(merged_results, bm25_scores_all=None):
    """
    Compute a 0-100% confidence score based on retrieval strength.
    Blends:
      - top RRF score (how strong the best match is)
      - BM25 raw score of top result (keyword strength)
      - number of merged results (coverage)
    """
    if not merged_results:
        return {"percent": 0, "level": "LOW"}

    top_rrf = merged_results[0].get("rrf", 0.0) if isinstance(merged_results[0], dict) else 0.0

    # Get BM25 raw score if available
    top_bm25 = 0.0
    bm25_count = 0
    for m in merged_results:
        d = m["doc"] if isinstance(m, dict) and "doc" in m else m
        if "bm25_score" in d:
            top_bm25 = max(top_bm25, d["bm25_score"])
            bm25_count += 1

    # Normalize
    # RRF scores in our system range ~0.005 to 0.033 (based on K=60)
    rrf_norm = min(1.0, top_rrf * 40)  # 0.025 → 1.0

    # BM25 raw scores vary widely; use soft saturation
    bm25_norm = min(1.0, top_bm25 / 25.0)  # ~25 is a strong match

    # Coverage: more merged results = higher confidence
    coverage_norm = min(1.0, len(merged_results) / 8.0)

    # Weighted blend
    confidence = (rrf_norm * 0.5) + (bm25_norm * 0.3) + (coverage_norm * 0.2)
    percent = int(max(5, min(100, confidence * 100)))

    if percent >= 70:
        level = "HIGH"
    elif percent >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {"percent": percent, "level": level}


def hybrid_retrieve(query, top_k=8):
    """
    Hybrid retrieval with Reciprocal Rank Fusion.
    Returns (list of dicts, confidence_dict)
    """
    global vectorstore, bm25_index, bm25_corpus

    if vectorstore is None:
        return [], {"percent": 0, "level": "LOW"}

    # --- 1. Semantic (FAISS) ---
    semantic_docs = []
    try:
        results = vectorstore.similarity_search(query, k=top_k * 2)
        for rank, doc in enumerate(results):
            semantic_docs.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "semantic_rank": rank + 1,
            })
    except Exception as e:
        print(f"[WARN] Semantic search failed: {e}")

    # --- 2. Keyword (BM25) ---
    bm25_docs = []
    if bm25_index is not None and bm25_corpus:
        try:
            tokenized_query = tokenize(query)
            scores = bm25_index.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k * 2]
            for rank, idx in enumerate(top_indices):
                if scores[idx] <= 0: continue
                text, source = bm25_corpus[idx]
                bm25_docs.append({
                    "text": text,
                    "source": source,
                    "bm25_rank": rank + 1,
                    "bm25_score": float(scores[idx]),
                })
        except Exception as e:
            print(f"[WARN] BM25 search failed: {e}")

    # --- 3. Reciprocal Rank Fusion ---
    K = 60
    rrf_scores = {}

    def doc_key(d):
        return (d["text"][:200], d["source"])

    for d in semantic_docs:
        key = doc_key(d)
        rrf_scores.setdefault(key, {"doc": d, "rrf": 0.0})
        rrf_scores[key]["rrf"] += 1.0 / (K + d["semantic_rank"])
        rrf_scores[key]["doc"]["semantic_rank"] = d["semantic_rank"]

    for d in bm25_docs:
        key = doc_key(d)
        rrf_scores.setdefault(key, {"doc": d, "rrf": 0.0})
        rrf_scores[key]["rrf"] += 1.0 / (K + d["bm25_rank"])
        rrf_scores[key]["doc"]["bm25_rank"] = d["bm25_rank"]
        rrf_scores[key]["doc"]["bm25_score"] = d["bm25_score"]

    merged = sorted(rrf_scores.values(), key=lambda x: -x["rrf"])[:top_k]

    docs = [m["doc"] for m in merged]

    # --- Build merged_chunks list for the source-attribution module ---
    merged_for_attribution = []
    for m in merged:
        d = m["doc"]
        merged_for_attribution.append({
            "source": d.get("source", "unknown"),
            "text": d.get("text", ""),
            "rrf_score": m["rrf"],
            "page": d.get("page"),
        })

    # --- Compute retrieval confidence (existing logic) ---
    confidence = compute_confidence(merged)

    # --- NEW: smart source filtering via Claude's module ---
    try:
        relevant_sources = filter_relevant_sources(
            query=query,
            merged_chunks=merged_for_attribution,
            idf=corpus_idf,
            groq_client=None,          # no LLM rerank for now
            use_llm=USE_LLM_RERANK,
            gap_ratio=0.55,
            min_keep_score=0.15,
        )
    except Exception as e:
        print(f"[WARN] Source attribution failed: {e}")
        # Fallback: old behavior
        relevant_sources = [
            {"source": s, "score": 0, "page": None, "preview": ""}
            for s in sorted(set(d["source"] for d in docs))
        ]

    confidence["relevant_sources"] = relevant_sources
    return docs, confidence


@app.route("/ask", methods=["POST"])
def ask():
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not set"}), 400

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    requested_model = data.get("model", GROQ_MODEL)
    requested_top_k = int(data.get("top_k", TOP_K))
    requested_temp = float(data.get("temperature", 0.2))

    requested_top_k = max(1, min(requested_top_k, 20))
    requested_temp = max(0.0, min(requested_temp, 1.0))

    ALLOWED_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"}
    if requested_model not in ALLOWED_MODELS:
        requested_model = GROQ_MODEL

    global vectorstore
    if vectorstore is None:
        load_existing_index()
    if vectorstore is None:
        return jsonify({"error": "No documents indexed"}), 400

    results, confidence = hybrid_retrieve(question, top_k=requested_top_k)

    context = "\n\n---\n\n".join(d["text"] for d in results)
    # Use filtered sources from confidence (now list of dicts)
    sources = confidence.get("relevant_sources", [])

    prompt = (
        "You are a precise, factual assistant. Answer using ONLY the context below.\n"
        "IMPORTANT RULES:\n"
        "1. If the user's query contains exact phrases, look for those phrases character-by-character.\n"
        "2. If the answer isn't in the context, say 'I don't know.'\n"
        "3. Do not add information not present in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    try:
        llm = ChatGroq(model=requested_model, api_key=GROQ_API_KEY, temperature=requested_temp)
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Groq failed: {e}"}), 500

    return jsonify({
        "answer": answer,
        "sources": sources,
        "model_used": requested_model,
        "top_k_used": requested_top_k,
        "temperature_used": requested_temp,
        "retrieval": "hybrid",
        "confidence": {k: v for k, v in confidence.items() if k != "relevant_sources"},
    })


@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not set"}), 400

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    requested_model = data.get("model", GROQ_MODEL)
    requested_top_k = int(data.get("top_k", TOP_K))
    requested_temp = float(data.get("temperature", 0.2))

    requested_top_k = max(1, min(requested_top_k, 20))
    requested_temp = max(0.0, min(requested_temp, 1.0))

    ALLOWED_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"}
    if requested_model not in ALLOWED_MODELS:
        requested_model = GROQ_MODEL

    global vectorstore
    if vectorstore is None:
        load_existing_index()
    if vectorstore is None:
        return jsonify({"error": "No documents indexed"}), 400

    results, confidence = hybrid_retrieve(question, top_k=requested_top_k)
    context = "\n\n---\n\n".join(d["text"] for d in results)
    sources = confidence.get("relevant_sources", [])

    prompt = (
        "You are a precise, factual assistant. Answer using ONLY the context below.\n"
        "IMPORTANT RULES:\n"
        "1. If the user's query contains exact phrases, look for those phrases character-by-character.\n"
        "2. If the answer isn't in the context, say 'I don't know.'\n"
        "3. Do not add information not present in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    def generate():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'retrieval': 'hybrid', 'confidence': {k: v for k, v in confidence.items() if k != 'relevant_sources'}})}\n\n"
        try:
            llm = ChatGroq(model=requested_model, api_key=GROQ_API_KEY,
                           temperature=requested_temp, streaming=True)
            chunk_count = 0
            for chunk in llm.stream(prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    chunk_count += 1
                    event = f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    padding = ":" + " " * 2048 + "\n\n"
                    yield event + padding
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )




@app.route("/remove_document", methods=["POST"])
def remove_document():
    """Remove all chunks with a given source filename from FAISS + BM25."""
    global vectorstore, bm25_index, bm25_corpus

    data = request.get_json(silent=True) or {}
    filename = (data.get("filename") or "").strip()
    if not filename:
        return jsonify({"error": "Missing 'filename'"}), 400

    if bm25_corpus is None and vectorstore is None:
        return jsonify({"error": "No index to remove from"}), 400

    # --- Remove from BM25 corpus ---
    removed_count = 0
    new_corpus = []
    if bm25_corpus:
        for text, source in bm25_corpus:
            if source == filename:
                removed_count += 1
            else:
                new_corpus.append((text, source))

    if removed_count == 0:
        return jsonify({"error": f"No chunks found for '{filename}'", "removed": 0}), 404

    try:
        # Rebuild BM25
        if new_corpus:
            bm25_index_new, bm25_corpus_new = build_bm25_index(new_corpus)
            bm25_index = bm25_index_new
            bm25_corpus = bm25_corpus_new
            with open(BM25_INDEX_FILE, "wb") as f:
                pickle.dump({"bm25": bm25_index, "corpus": bm25_corpus}, f)
        else:
            # Nothing left
            bm25_index = None
            bm25_corpus = []
            if os.path.isfile(BM25_INDEX_FILE):
                os.remove(BM25_INDEX_FILE)

        # Rebuild FAISS from remaining corpus
        if vectorstore is not None:
            if new_corpus:
                emb = ensure_embeddings()
                docs = [Document(page_content=t, metadata={"source": s}) for t, s in new_corpus]
                vectorstore_new = FAISS.from_documents(docs, emb)
                vectorstore = vectorstore_new
                vectorstore.save_local(INDEX_DIR)
            else:
                # Nothing left
                vectorstore = None
                if os.path.isdir(INDEX_DIR):
                    shutil.rmtree(INDEX_DIR)

        # Also remove the physical file
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        return jsonify({
            "success": True,
            "removed_chunks": removed_count,
            "remaining_chunks": len(new_corpus),
            "message": f"Removed {removed_count} chunks from '{filename}'"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Removal failed: {e}"}), 500



@app.route("/suggest_questions", methods=["POST"])
def suggest_questions():
    """Generate 3 suggested questions from the indexed chunks."""
    global bm25_corpus

    if not GROQ_API_KEY:
        return jsonify({"success": False, "error": "GROQ_API_KEY not set"}), 400

    if not bm25_corpus or len(bm25_corpus) == 0:
        return jsonify({"success": False, "error": "No documents indexed yet"}), 400

    # Sample up to 5 diverse chunks
    import random
    sample_size = min(5, len(bm25_corpus))
    sampled = random.sample(bm25_corpus, sample_size)

    # Build context from sampled chunks (truncate each to 500 chars)
    context = "\n\n---\n\n".join(
        f"Source: {src}\n{text[:500]}"
        for text, src in sampled
    )

    prompt = (
        "You are analyzing document content to suggest starter questions.\n\n"
        "Read the following excerpts and generate EXACTLY 3 short, diverse questions "
        "that a user might want to ask. The questions should:\n"
        "- Be answerable from the content shown\n"
        "- Cover different topics or aspects\n"
        "- Be concise (max 12 words each)\n"
        "- Sound natural (like a real user asking)\n\n"
        f"Content excerpts:\n{context}\n\n"
        "Respond with ONLY a JSON array of 3 strings. No explanation, no markdown.\n"
        'Example format: ["What is X?", "How does Y work?", "Why is Z important?"]'
    )

    try:
        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.5)
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Clean markdown fences if present
        import re as _re
        raw = _re.sub(r"^```(json)?|```$", "", raw, flags=_re.MULTILINE).strip()

        # Parse JSON array
        import json as _json
        questions = _json.loads(raw)

        if not isinstance(questions, list):
            raise ValueError("Not a list")

        # Keep only first 3 valid strings
        questions = [str(q).strip() for q in questions if isinstance(q, str) and q.strip()][:3]

        if not questions:
            raise ValueError("No valid questions")

        return jsonify({"success": True, "questions": questions})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to generate: {e}"}), 500


@app.route("/clear", methods=["POST"])
def clear():
    global vectorstore, bm25_index, bm25_corpus
    vectorstore = None
    bm25_index = None
    bm25_corpus = None
    for path in (UPLOAD_DIR, INDEX_DIR):
        if os.path.isdir(path):
            shutil.rmtree(path)
    if os.path.isfile(BM25_INDEX_FILE):
        os.remove(BM25_INDEX_FILE)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return jsonify({"message": "Cleared all documents and indices."})


if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("[WARN] GROQ_API_KEY not set")
    if not HF_API_TOKEN:
        print("[WARN] HF_API_TOKEN not set")

    load_existing_index()

    port = int(os.environ.get("PORT", 5002))
    print(f"🌐 RAG Assistant running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
