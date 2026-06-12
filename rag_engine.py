#!/usr/bin/env python3
"""
Regulatory Compliance RAG Assistant Engine.

Implements an enterprise-grade hybrid retrieval and reranking pipeline for
processing regulatory documents from RBI, SEBI, and the Basel Committee.

Pipeline architecture:
  1. Dense semantic search via ChromaDB
  2. Sparse keyword search via BM25 Okapi
  3. Reciprocal Rank Fusion (RRF) for ensemble blending
  4. Cross-Encoder deep-learning reranking
  5. Multi-LLM provider orchestration with citation enforcement
"""

import os
import json
import logging
import time
import string
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from collections import Counter

import numpy as np
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import sentence_transformers
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import ollama


# ------------------------------------------------------------------------------
# DATA CLASSES  (mirrors ingest.py — replace with `from ingest import ...` if available)
# ------------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    """Metadata container for a regulatory text chunk."""
    doc_title: str
    page_number: int
    regulatory_body: str
    chunk_id: str
    file_path: Optional[str] = None
    section_heading: Optional[str] = None
    additional_properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "doc_title":       self.doc_title,
            "page_number":     self.page_number,
            "regulatory_body": self.regulatory_body,
            "chunk_id":        self.chunk_id,
            "file_path":       self.file_path or "",
            "section_heading": self.section_heading or ""
        }
        for k, v in self.additional_properties.items():
            if isinstance(v, (str, int, float, bool)):
                base[f"prop_{k}"] = v
        return base

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkMetadata":
        props = {k[5:]: v for k, v in d.items() if k.startswith("prop_")}
        return cls(
            doc_title=str(d.get("doc_title", "Unknown")),
            page_number=int(d.get("page_number", 0)),
            regulatory_body=str(d.get("regulatory_body", "Unknown")),
            chunk_id=str(d.get("chunk_id", "")),
            file_path=d.get("file_path") or None,
            section_heading=d.get("section_heading") or None,
            additional_properties=props
        )


@dataclass
class TextChunk:
    """Complete text chunk payload."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata


# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

VECTOR_TOP_K:       int   = 10
BM25_TOP_K:         int   = 10
FINAL_TOP_K:        int   = 5
RRF_K:              int   = 10
RRF_WEIGHT_VECTOR:  float = 0.5
RRF_WEIGHT_BM25:    float = 0.5
RERANK_TOP_K:       int   = 20
EMBEDDING_MODEL:    str   = "all-MiniLM-L6-v2"
RERANK_MODEL:       str   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DB_PATH:     str   = "./chroma_db"
BM25_INDEX_PATH:    str   = "./bm25_index"
MAX_CONTEXT_LENGTH: int   = 3000


# ------------------------------------------------------------------------------
# PIPELINE RESULT DATA CLASSES
# ------------------------------------------------------------------------------

@dataclass
class RetrieverResult:
    """Single retrieval result traversing the full pipeline."""
    chunk_text:   str
    metadata:     ChunkMetadata
    vector_score: float   # Cosine similarity from ChromaDB
    bm25_score:   float   # BM25 Okapi lexical score
    rrf_score:    float   # Reciprocal Rank Fusion blended score
    rerank_score: float   # Cross-Encoder neural rerank score


@dataclass
class QueryResult:
    """Full pipeline response including answer, citations, and timing."""
    query:                        str
    answers:                      str
    citations:                    List[RetrieverResult]
    retrieved_count:              int
    total_processing_time_seconds: float
    retrieval_stage_times:        Dict[str, float]


@dataclass
class RetrievalMetrics:
    """Evaluation metrics for retrieval quality profiling."""
    hit_rate_at_3:          float
    hit_rate_at_5:          float
    hit_rate_at_10:         float
    mean_reciprocal_rank:   float
    average_vector_score:   float
    average_bm25_score:     float
    average_rerank_score:   float
    total_queries_evaluated: int


# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("RAGEngine")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        h = logging.StreamHandler()
        h.setFormatter(formatter)
        logger.addHandler(h)
    return logger

logger = setup_logging()


# ------------------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------------------

def tokenize_text(text: str) -> List[str]:
    """Lowercase, strip punctuation, and split into tokens."""
    if not text:
        return []
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


# ------------------------------------------------------------------------------
# EMBEDDING GENERATOR
# ------------------------------------------------------------------------------

class EmbeddingGenerator:
    """Thin wrapper around a SentenceTransformer for query embedding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_single(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()


# ------------------------------------------------------------------------------
# HYBRID RETRIEVER
# ------------------------------------------------------------------------------

class HybridRetriever:
    """Manages dense (ChromaDB) and sparse (BM25) retrieval."""

    def __init__(
        self,
        chroma_path: str = CHROMA_DB_PATH,
        bm25_path:   str = BM25_INDEX_PATH,
        logger_instance: Optional[logging.Logger] = None
    ):
        self.logger     = logger_instance or logger
        self.chroma_path = chroma_path
        self.bm25_path   = bm25_path

       # ----------------------------------------------------------------------
        # ChromaDB - Modified to allow graceful blank index initialization
        # ----------------------------------------------------------------------
        try:
            self.chroma_client     = chromadb.PersistentClient(path=self.chroma_path)
            # CHANGED: Using get_or_create_collection so it boots clean if empty
            self.chroma_collection = self.chroma_client.get_or_create_collection(name="regulatory_documents")
            self.logger.info(f"ChromaDB initialized at {self.chroma_path} (Current doc count: {self.chroma_collection.count()})")
        except Exception as e:
            self.logger.critical(f"ChromaDB connection failed: {e}")
            # Dynamic recovery guardrail to keep the Streamlit Cloud dashboard alive
            self.chroma_collection = None
            self.logger.warning("Pipeline proceeding in degraded state. Data ingestion required.")

        # BM25
        self.bm25_index: Optional[BM25Okapi] = None
        self.doc_id_map: Dict[int, str] = {}
        self._load_bm25_index()

        # Embeddings
        self.embedding_generator = EmbeddingGenerator(model_name=EMBEDDING_MODEL)
        self.logger.info(f"Embedding model loaded: {EMBEDDING_MODEL}")

    def _load_bm25_index(self) -> None:
        bm25_file = Path(self.bm25_path) / "bm25_store.json"
        if not bm25_file.exists():
            self.logger.warning(f"BM25 index not found at {bm25_file}. Hybrid search degraded.")
            return
        try:
            with open(bm25_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            corpus  = payload.get("corpus", [])
            mapping = payload.get("mapping", {})
            tokenized = [tokenize_text(doc) for doc in corpus]
            if tokenized:
                self.bm25_index = BM25Okapi(tokenized)
                self.doc_id_map = {int(k): v for k, v in mapping.items()}
                self.logger.info(f"BM25 index loaded ({len(self.doc_id_map)} docs)")
            else:
                self.logger.error("BM25 corpus is empty.")
        except Exception as e:
            self.logger.error(f"Failed to load BM25 index: {e}")

    def retrieve_vector(
        self,
        query: str,
        top_k: int = VECTOR_TOP_K,
        regulatory_body: Optional[str] = None
    ) -> List[Tuple[str, Dict, float]]:
        """Dense retrieval via ChromaDB cosine similarity."""
        try:
            embedding   = self.embedding_generator.embed_single(query)
            filter_dict = {"regulatory_body": regulatory_body} if regulatory_body else None

            results = self.chroma_collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=filter_dict,
                include=["metadatas", "documents", "distances"]
            )
            if not results or not results["ids"] or not results["ids"][0]:
                return []

            extracted = []
            for idx in range(len(results["ids"][0])):
                dist = results["distances"][0][idx]
                sim  = max(0.0, 1.0 - dist / 2.0)
                meta = dict(results["metadatas"][0][idx] or {})
                meta["text"] = results["documents"][0][idx] or ""
                extracted.append((results["ids"][0][idx], meta, sim))

            avg = np.mean([r[2] for r in extracted]) if extracted else 0.0
            self.logger.info(f"Vector retrieval: {len(extracted)} results, avg sim={avg:.4f}")
            return extracted

        except Exception as e:
            self.logger.error(f"Vector retrieval error: {e}")
            return []

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
        regulatory_body: Optional[str] = None
    ) -> List[Tuple[str, Dict, float]]:
        """Sparse BM25 retrieval."""
        if not self.bm25_index:
            self.logger.error("BM25 index not loaded.")
            return []
        try:
            tokens = tokenize_text(query)
            scores = np.array(self.bm25_index.get_scores(tokens))
            if not scores.any():
                return []

            results = []
            for idx in np.argsort(scores)[::-1]:
                if scores[idx] <= 0:
                    continue
                chunk_id = self.doc_id_map.get(idx)
                if not chunk_id:
                    continue
                meta = self._get_metadata_by_chunk_id(chunk_id)
                results.append((chunk_id, meta, float(scores[idx])))

            if regulatory_body:
                results = sorted(
                    [r for r in results if r[1].get("regulatory_body") == regulatory_body],
                    key=lambda x: x[2], reverse=True
                )

            truncated = results[:top_k]
            avg = np.mean([r[2] for r in truncated]) if truncated else 0.0
            self.logger.info(f"BM25 retrieval: {len(truncated)} results, avg score={avg:.4f}")
            return truncated

        except Exception as e:
            self.logger.error(f"BM25 retrieval error: {e}")
            return []

    def _get_metadata_by_chunk_id(self, chunk_id: str) -> Dict:
        try:
            res = self.chroma_collection.get(ids=[chunk_id], include=["metadatas", "documents"])
            if res and res["metadatas"]:
                meta = dict(res["metadatas"][0])
                meta["text"] = res["documents"][0] if res["documents"] else ""
                return meta
        except Exception as e:
            self.logger.error(f"Metadata lookup error for {chunk_id}: {e}")
        return {}


# ------------------------------------------------------------------------------
# RECIPROCAL RANK FUSION
# ------------------------------------------------------------------------------

def reciprocal_rank_fusion(
    vector_results: List[Tuple[str, Dict, float]],
    bm25_results:   List[Tuple[str, Dict, float]],
    k:              int   = RRF_K,
    weight_vector:  float = RRF_WEIGHT_VECTOR,
    weight_bm25:    float = RRF_WEIGHT_BM25
) -> List[RetrieverResult]:
    """Blend vector and BM25 rankings using Reciprocal Rank Fusion."""
    v_rank = {cid: r + 1 for r, (cid, _, _) in enumerate(vector_results)}
    b_rank = {cid: r + 1 for r, (cid, _, _) in enumerate(bm25_results)}

    all_ids: Set[str] = set(v_rank) | set(b_rank)
    rrf_scores: Dict[str, float] = {}

    for cid in all_ids:
        vr = v_rank.get(cid)
        br = b_rank.get(cid)
        sv = weight_vector / (k + vr)  if vr is not None else weight_vector / (k + len(vector_results) + 1)
        sb = weight_bm25   / (k + br)  if br is not None else weight_bm25   / (k + len(bm25_results)   + 1)
        rrf_scores[cid] = sv + sb

    v_lookup = {cid: (m, s) for cid, m, s in vector_results}
    b_lookup = {cid: (m, s) for cid, m, s in bm25_results}

    combined: List[RetrieverResult] = []
    for cid in sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True):
        meta_dict = None
        v_score = b_score = 0.0

        if cid in v_lookup:
            meta_dict, v_score = v_lookup[cid]
        if cid in b_lookup:
            bm, b_score = b_lookup[cid]
            if meta_dict is None:
                meta_dict = bm

        if not meta_dict:
            continue

        chunk_text = meta_dict.pop("text", "")
        combined.append(RetrieverResult(
            chunk_text=chunk_text,
            metadata=ChunkMetadata.from_dict(meta_dict),
            vector_score=v_score,
            bm25_score=b_score,
            rrf_score=rrf_scores[cid],
            rerank_score=0.0
        ))

    logger.info(f"RRF fusion: {len(combined)} combined results")
    return combined


# ------------------------------------------------------------------------------
# CROSS-ENCODER RERANKER
# ------------------------------------------------------------------------------

class CrossEncoderReranker:
    """Reranks candidates using a cross-attention neural model."""

    def __init__(self, model_name: str = RERANK_MODEL, logger_instance: Optional[logging.Logger] = None):
        self.logger     = logger_instance or logger
        self.model_name = model_name
        try:
            self.model = CrossEncoder(model_name)
            self.logger.info(f"Cross-encoder loaded: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder: {e}")
            self.model = None

def rerank(self, query: str, results: List[RetrieverResult], top_k: int = RERANK_TOP_K) -> List[RetrieverResult]:
        if not self.model or not results:
            self.logger.warning("Reranking skipped — model unavailable or empty results.")
            return results[:top_k]
        try:
            candidates = results[:RERANK_TOP_K]
            pairs      = [[query, r.chunk_text] for r in candidates]
            scores     = self.model.predict(pairs)

            if isinstance(scores, float):
                scores = np.array([scores])

            # --- PATCH: Convert raw logit scores to a clean 0-1 probability scale via Sigmoid ---
            probabilities = 1 / (1 + np.exp(-scores))

            for i, prob in enumerate(probabilities):
                candidates[i].rerank_score = float(prob)
            # ----------------------------------------------------------------------------------

            ranked = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)
            self.logger.info(f"Reranked {len(candidates)} results; top={max(probabilities):.4f}, bottom={min(probabilities):.4f}")
            return ranked[:top_k]

        except Exception as e:
            self.logger.error(f"Reranking error: {e}")
            return results[:top_k]


# ------------------------------------------------------------------------------
# RAG PIPELINE
# ------------------------------------------------------------------------------

class RAGPipeline:
    """
    Main pipeline coordinator: retrieval → RRF fusion → reranking → LLM generation.

    Parameters
    ----------
    llm_type : str
        One of "openai" or "ollama".
    llm_model_name : str
        Ollama model name (used when llm_type="ollama").
    openai_model_name : str
        OpenAI-compatible model name (used when llm_type="openai").
    openai_base_url : Optional[str]
        Custom base URL for the OpenAI-compatible API endpoint.
        Pass "https://api.groq.com/openai/v1" to route requests through Groq.
        Defaults to None, which uses the standard OpenAI endpoint.
    logger_instance : Optional[logging.Logger]
        Inject a custom logger; falls back to the module-level logger.
    """

    def __init__(
        self,
        llm_type:          str = "ollama",
        llm_model_name:    str = "llama3",
        openai_model_name: str = "gpt-4o-mini",
        openai_base_url:   Optional[str] = None,    # ← FIX: new parameter accepted here
        logger_instance:   Optional[logging.Logger] = None
    ):
        self.logger            = logger_instance or logger
        self.llm_type          = llm_type.lower()
        self.llm_model_name    = llm_model_name
        self.openai_model_name = openai_model_name
        self.openai_base_url   = openai_base_url    # ← stored for use in _setup_llm_client

        self.hybrid_retriever = HybridRetriever(logger_instance=self.logger)
        self.reranker         = CrossEncoderReranker(logger_instance=self.logger)

        load_dotenv()
        self._setup_llm_client()
        self.logger.info(f"RAGPipeline initialized | provider={self.llm_type} | base_url={self.openai_base_url or 'default'}")

    def _setup_llm_client(self) -> None:
        """Establish the LLM client connection."""
        if self.llm_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.error("OPENAI_API_KEY not set.")
            try:
                from openai import OpenAI
                # FIX: pass base_url only when explicitly provided (e.g. for Groq)
                if self.openai_base_url:
                    self.openai_client = OpenAI(api_key=api_key, base_url=self.openai_base_url)
                    self.logger.info(f"OpenAI client configured with custom base_url: {self.openai_base_url}")
                else:
                    self.openai_client = OpenAI(api_key=api_key)
                    self.logger.info("OpenAI client configured with default endpoint.")
            except ImportError:
                self.logger.error("openai package missing. Run: pip install openai")

        elif self.llm_type == "ollama":
            pass  # Uses direct HTTP calls at inference time

        else:
            raise ValueError(f"Unsupported llm_type: '{self.llm_type}'. Choose 'openai' or 'ollama'.")

    def query(
        self,
        query_text:      str,
        regulatory_body: Optional[str] = None,
        final_top_k:     int = FINAL_TOP_K
    ) -> QueryResult:
        """Execute the full retrieval-augmented generation pipeline."""
        start_time   = time.time()
        stage_times: Dict[str, float] = {}

        # 1. Dense retrieval
        t = time.time()
        vector_results = self.hybrid_retriever.retrieve_vector(query_text, top_k=VECTOR_TOP_K, regulatory_body=regulatory_body)
        stage_times["vector_retrieval"] = time.time() - t

        # 2. Sparse retrieval
        t = time.time()
        bm25_results = self.hybrid_retriever.retrieve_bm25(query_text, top_k=BM25_TOP_K, regulatory_body=regulatory_body)
        stage_times["bm25_retrieval"] = time.time() - t

        # 3. RRF fusion
        t = time.time()
        rrf_results = reciprocal_rank_fusion(vector_results, bm25_results, k=RRF_K)
        stage_times["rrf_fusion"] = time.time() - t

        # 4. Cross-encoder reranking
        t = time.time()
        reranked = self.reranker.rerank(query_text, rrf_results, top_k=RERANK_TOP_K)
        stage_times["reranking"] = time.time() - t

        # 5. Trim to final_top_k
        final = reranked[:final_top_k]

        # 6. Build context and generate answer
        t = time.time()
        context = self._build_context(final)
        answer  = self._generate_answer(query_text, context)
        stage_times["answer_generation"] = time.time() - t

        total = time.time() - start_time
        stage_times["total"] = total

        self.logger.info(f"Query complete in {total:.4f}s | {len(final)} citations returned.")
        return QueryResult(
            query=query_text,
            answers=answer,
            citations=final,
            retrieved_count=len(final),
            total_processing_time_seconds=total,
            retrieval_stage_times=stage_times
        )

    def _build_context(self, results: List[RetrieverResult]) -> str:
        parts = []
        for i, res in enumerate(results):
            parts.append(
                f"[Document {i+1}]\n"
                f"Source: {res.metadata.doc_title} (Page {res.metadata.page_number})\n"
                f"Regulatory Body: {res.metadata.regulatory_body}\n"
                f"Rerank Score: {res.rerank_score:.4f}\n"
                f"Text:\n{res.chunk_text.strip()}\n"
                f"{'-'*40}"
            )
        context = "\n\n".join(parts)
        if len(context) > MAX_CONTEXT_LENGTH:
            self.logger.warning(f"Context truncated to {MAX_CONTEXT_LENGTH} chars.")
            context = context[:MAX_CONTEXT_LENGTH] + "\n[TRUNCATED]"
        self.logger.info(f"Context built: {len(results)} chunks, {len(context)} chars")
        return context

    def _generate_answer(self, query: str, context: str) -> str:
        system_prompt = (
            "You are a regulatory compliance assistant for financial institutions.\n"
            "Answer questions about RBI, Basel Committee, and SEBI regulations.\n\n"
            "RULES:\n"
            "1. Cite exact document sources for every fact.\n"
            "2. Never hallucinate — only use the provided context.\n"
            "3. If context is insufficient, say: \"I cannot find this information in the provided regulatory documents\"\n"
            "4. Use precise legal terminology.\n"
            "5. Include specific numbers, dates, and percentages from documents.\n"
            "6. Format citations as: [Document Name, Page X]\n\n"
            "Answer based ONLY on the context below."
        )
        user_prompt = f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"

        if self.llm_type == "openai":
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content
            except Exception as e:
                self.logger.error(f"OpenAI API error: {e}")
                return f"LLM generation failed: {e}"

        elif self.llm_type == "ollama":
            try:
                import requests
                res = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model":   self.llm_model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_prompt}
                        ],
                        "options": {"temperature": 0.1},
                        "stream":  False
                    },
                    timeout=180  # <-- Increased to 3 minutes to give your hardware breathing room
                )
                if res.status_code == 200:
                    return res.json().get("message", {}).get("content", "Empty response from Ollama.")
                return f"Ollama returned HTTP {res.status_code}"
            except Exception as e:
                self.logger.error(f"Ollama request error: {e}")
                return f"Ollama communication error: {e}"

        return "No LLM provider matched — check llm_type configuration."


# ------------------------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Regulatory Compliance RAG CLI")
    parser.add_argument("--query",           type=str,   required=True)
    parser.add_argument("--regulatory-body", type=str,   choices=["RBI", "SEBI", "Basel Committee"], default=None)
    parser.add_argument("--llm-type",        type=str,   choices=["openai", "ollama"], default="ollama")
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"🚀 REGULATORY COMPLIANCE RAG  |  LLM: {args.llm_type.upper()}")
    print("="*80 + "\n")

    try:
        Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        bm25_dir = Path(BM25_INDEX_PATH)
        bm25_dir.mkdir(parents=True, exist_ok=True)

        mock_file = bm25_dir / "bm25_store.json"
        if not mock_file.exists():
            mock_data = {
                "corpus": [
                    "The liquidity coverage ratio requires high quality liquid assets to exceed projected net cash outflows over a 30-day stress period.",
                    "SEBI PIT regulations govern handling of unpublished price sensitive information (UPSI)."
                ],
                "mapping": {"0": "chunk_rbi_mock_001", "1": "chunk_sebi_mock_002"}
            }
            with open(mock_file, "w", encoding="utf-8") as f:
                json.dump(mock_data, f)

        pipeline = RAGPipeline(llm_type=args.llm_type)
        result   = pipeline.query(query_text=args.query, regulatory_body=args.regulatory_body)

        print("\n" + "#"*40 + " ANSWER " + "#"*40)
        print(result.answers)
        print("#"*88 + "\n")

        print(f"📚 CITATIONS ({result.retrieved_count}):")
        for i, c in enumerate(result.citations):
            print(f"  {i+1}. [{c.metadata.regulatory_body}] {c.metadata.doc_title} (Page {c.metadata.page_number})")
            print(f"     Chunk: {c.metadata.chunk_id} | Score: {c.rerank_score:.4f}")
            print(f"     Preview: {c.chunk_text[:150].strip()}...\n")

        print("⏱️  STAGE TIMINGS:")
        print(f"  Total: {result.total_processing_time_seconds:.4f}s")
        for stage, dur in result.retrieval_stage_times.items():
            if stage != "total":
                print(f"    • {stage.replace('_', ' ').title()}: {dur:.4f}s")
        print()

    except Exception as e:
        logger.error(f"Fatal pipeline error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
