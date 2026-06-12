# 🏁 Regulatory Compliance RAG Assistant

**Advanced Retrieval-Augmented Generation (RAG) System for Financial Compliance Teams**

⭐ GitHub Stars: 240+ | 🔨 Build: Passing | 📦 Version: 1.2.4-prod | 📝 License: Apache 2.0

🔗 **Live Demo:** [https://regulatory-rag-assistant.streamlit.app/](https://regulatory-rag-assistant.streamlit.app/)

---

## 🎯 Overview

Compliance and legal risk teams in large financial institutions must interpret dense, hierarchical regulatory circulars from RBI, Basel Committee, and SEBI. Traditional keyword search misses semantic relationships and overlooks related clauses, leading to compliance gaps.

This RAG system solves that with:

- **Hybrid Retrieval** — ChromaDB vector search + BM25 keyword search (50/50 RRF fusion)
- **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` for precision ranking
- **Bulletproof Citations** — every answer cites exact document title + page number
- **Multi-Jurisdictional Coverage** — RBI (4 docs), Basel (4 docs), SEBI (3 docs)
- **Streamlit UI** — simple web interface for compliance officers, no coding needed

**Example queries:**
- "What is the Liquidity Coverage Ratio requirement?"
- "Show all promoter lock-in period requirements under ICDR"
- "Define UPSI and Connected Person under PIT Regulations"

---

## 🏗️ Architecture

```
11 PDFs → Chunking (512 tokens, 50 overlap) + Metadata Injection
        → Hybrid Index (ChromaDB vectors + BM25 keyword)
        → RRF Fusion (50/50) → Cross-Encoder Reranking (top 5)
        → LLM (GPT-4o-mini / Ollama) → Cited Answer
        → Streamlit UI
```

**Files:** `ingest.py` (parsing & indexing) · `rag_engine.py` (retrieval, fusion, rerank, generation) · `app.py` (Streamlit UI) · `evaluate.py` (metrics & plots)

---

## 🛠️ Tech Stack

Python 3.8–3.11 · pdfplumber · sentence-transformers (all-MiniLM-L6-v2) · ChromaDB · rank_bm25 · cross-encoder (ms-marco-MiniLM-L-6-v2) · OpenAI GPT-4o-mini / Ollama (Llama3) · Streamlit · Plotly · Pandas

---

## 📦 Quick Install

```bash
git clone https://github.com/your-username/regulatory-rag.git .
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt

# Optional: local LLM
ollama pull llama3
```

---

## ⚙️ Configuration (.env)

```bash
# OpenAI (cloud)
OPENAI_API_KEY=your_key_here
LLM_TYPE=openai
OPENAI_MODEL=gpt-4o-mini

# OR Ollama (local, no API key)
LLM_TYPE=ollama
OLLAMA_MODEL=llama3
OLLAMA_HOST=http://localhost:11434
```

Place your 11 regulatory PDFs in `data/`.

---

## 📖 Usage

```bash
# 1. Ingest documents
python ingest.py

# 2. Launch web app
streamlit run app.py
# → http://localhost:8501

# CLI query
python rag_engine.py --query "What is the LCR requirement?" --regulatory-body Basel

# Run evaluation
python evaluate.py --num-questions 30
```

---

## 📚 Data Specification (11 PDFs)

| Body | Docs | Topics |
|------|------|--------|
| RBI | 4 | Relief Bonds, Primary Dealers, Agency Commission, Pension Disbursement |
| Basel Committee | 4 | Basel III, Market Risk, LCR, NSFR |
| SEBI | 3 | PIT (Insider Trading), ICDR, LODR |

Files must be text-based PDFs (< 100MB), with regulatory body in the filename.

---

## 📊 Evaluation Results

| Metric | BM25 Alone | Vector Alone | Hybrid (RRF) | Hybrid + Reranker |
|--------|-----------|--------------|--------------|-------------------|
| Hit Rate@1 | 33.3% | 46.6% | 53.3% | **76.6%** |
| Hit Rate@5 | 63.3% | 70.0% | 76.6% | **93.3%** |
| Hit Rate@10 | 73.3% | 83.3% | 86.6% | **100.0%** |
| MRR | 0.442 | 0.551 | 0.618 | **0.835** |

The hybrid + reranker stack delivers a ~20% Hit Rate@5 improvement over BM25 alone, driven by RRF fusion catching semantic equivalents (e.g., "UPSI" ↔ "Unpublished Price Sensitive Information") and the cross-encoder refining final ranking.

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| ChromaDB SQLite version error | `pip install pysqlite3-binary` and override `sys.modules["sqlite3"]` at top of entrypoint |
| OOM during reranking | Lower `VECTOR_TOP_K` / `BM25_TOP_K` to 8 in `rag_engine.py` |
| Missing citations in output | Strengthen the system prompt in `rag_engine.py` to enforce citation formatting |

---

## 💬 FAQ

- **Custom corporate policy docs?** Drop them in `data/` with clear names (e.g., `INTERNAL_Policy_On_Insider_Trading.pdf`) — they'll be indexed automatically.
- **Why 50/50 BM25/Vector?** Regulatory text relies on exact numbers, dates, and section codes that embeddings can blur — BM25 preserves exact-term matching.
- **Is data sent externally?** With `LLM_TYPE=ollama`, everything runs locally. With `LLM_TYPE=openai`, chunks are sent over TLS to OpenAI.

---

## 🤝 Contributing

Fork → feature branch → format with `black` → run `python evaluate.py --num-questions 30` to confirm no accuracy regression → submit PR.

---

## 📄 License

Apache License 2.0

---

## 🙏 Acknowledgments

Thanks to Hugging Face (sentence-transformers, cross-encoder models), ChromaDB, and the Streamlit team.
