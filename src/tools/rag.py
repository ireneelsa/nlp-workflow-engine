import os
import time
from collections import OrderedDict
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai
from loguru import logger
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from src.tools.base import BaseTool
from src.models import ToolResult

load_dotenv(override=True)

_CACHE_MAX = 100
_CACHE_TTL = 300  # seconds

# OrderedDict preserves insertion order so popitem(last=False) evicts the oldest entry.
_context_cache: OrderedDict = OrderedDict()


def _cache_get(key: Tuple) -> Optional[Tuple[List, List, str]]:
    entry = _context_cache.get(key)
    if entry is None:
        return None
    chunks, chunk_types_used, doc_summary, ts = entry
    if time.time() - ts > _CACHE_TTL:
        del _context_cache[key]
        return None
    _context_cache.move_to_end(key)
    return chunks, chunk_types_used, doc_summary


def _cache_set(key: Tuple, chunks: List, chunk_types_used: List, doc_summary: str) -> None:
    if key in _context_cache:
        _context_cache.move_to_end(key)
    _context_cache[key] = (chunks, chunk_types_used, doc_summary, time.time())
    while len(_context_cache) > _CACHE_MAX:
        _context_cache.popitem(last=False)


class RAGTool(BaseTool):
    name = "rag"
    description = "Answers a specific question by searching a document that was previously uploaded and indexed."
    input_schema = {
        "question": "The question to answer based on the document",
        "collection_name": "The name of the indexed document collection to search",
        "topic_filter": "Optional topic category to restrict search to matching chunks"
    }

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")
        else:
            self.gemini_model = None

        self.chroma = chromadb.PersistentClient(path="./chroma_store")
        self.embed_fn = None
        self._init_error: str | None = None

    def _ensure_embedding_fn(self):
        if self.embed_fn is not None or self._init_error is not None:
            return
        try:
            self.embed_fn = ONNXMiniLM_L6_V2()
        except Exception as e:
            self._init_error = str(e)
            logger.error(f"[rag] Embedding init failed: {e}")

    def run(self, input: Dict[str, Any]) -> ToolResult:
        start = time.time()
        question = input.get("question", "")
        collection_name = input.get("collection_name", "default")
        topic_filter = input.get("topic_filter") or None

        self._ensure_embedding_fn()
        if self._init_error:
            return self._make_result(input, None, start, error=f"RAG initialization failed: {self._init_error}")

        if not question:
            return self._make_result(input, None, start, error="No question provided")

        logger.info(f"[rag] Searching '{collection_name}' for: {question[:60]}")

        try:
            cache_key = (collection_name, question)
            cached = _cache_get(cache_key)

            if cached is not None:
                chunks, chunk_types_used, doc_summary = cached
                logger.info(f"[rag] Cache hit ({len(_context_cache)} entries)")
            else:
                collection = self.chroma.get_collection(name=collection_name)

                query_embedding = self.embed_fn([question])
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(3, collection.count()),
                    where={"topic": topic_filter} if topic_filter else None,
                    include=["documents", "metadatas"]
                )

                chunks = results["documents"][0]
                if not chunks:
                    return self._make_result(input, {
                        "answer": "No relevant content found.",
                        "chunks_used": 0,
                        "collection": collection_name,
                        "chunk_types_used": []
                    }, start)

                raw_types = []
                doc_summary = ""
                if "metadatas" in results and results["metadatas"][0]:
                    for meta in results["metadatas"][0]:
                        if meta:
                            raw_types.append(meta.get("chunk_type", "text"))
                            if not doc_summary:
                                doc_summary = meta.get("doc_summary", "") or doc_summary
                else:
                    raw_types = ["text"] * len(chunks)

                chunk_types_used = list(dict.fromkeys(raw_types))
                _cache_set(cache_key, chunks, chunk_types_used, doc_summary)

            logger.info(f"[rag] Found {len(chunks)} relevant chunks")

            context = f"{doc_summary}\n\n" if doc_summary else ""
            context += "\n\n".join(chunks)

            if self.gemini_model is None:
                raise RuntimeError("[rag] GEMINI_API_KEY is not configured for Gemini model")

            prompt = f"""You are answering questions based only on this document context:\n\n{context}\n\nQuestion: {question}\n\nAnswer clearly and concisely based only on the context above."""

            response = self.gemini_model.generate_content(prompt)
            answer = getattr(response, "text", "").strip()
            logger.info(f"[rag] Answer: {answer[:80]}...")

            return self._make_result(input, {
                "answer": answer,
                "chunks_used": len(chunks),
                "collection": collection_name,
                "chunk_types_used": chunk_types_used
            }, start)

        except Exception as e:
            logger.error(f"[rag] Failed: {e}")
            return self._make_result(input, None, start, error=str(e))
