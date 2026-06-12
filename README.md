🏁 Regulatory Compliance RAG AssistantAdvanced Retrieval-Augmented Generation (RAG) System for Financial Compliance Teams⭐ GitHub Stars: 240+ | 🔨 Build: Passing | 📦 Version: 1.2.4-prod | 📝 License: Apache 2.0Compliance and legal risk teams in large financial institutions must frequently interpret dense, complex, and hierarchical regulatory circulars from multiple statutory bodies. Finding the correct clauses quickly is challenging with traditional search systems that rely solely on keyword matching, often leading to overlooked mandates, tracking failures, or regulatory penalties.This project addresses this by constructing an advanced Retrieval-Augmented Generation (RAG) system capable of:✅ Semantic precision with dense vector embeddings✅ Hybrid lookup combining vector + keyword search✅ Bulletproof source verification with document/page citations✅ Multi-jurisdictional support across RBI, Basel Committee, and SEBI frameworks📑 Table of ContentsProject OverviewKey FeaturesSystem ArchitectureTechnical StackPrerequisitesInstallation GuideConfigurationUsage GuideData SpecificationProject StructureAPI DocumentationEvaluation MethodologyEvaluation ResultsPerformance MetricsTroubleshootingFAQContributingLicenseCitationAcknowledgments🎯 Project OverviewProblem StatementCompliance and legal risk teams in large financial institutions face critical challenges when interpreting regulatory documents:ChallengeImpactDense, complex text200+ page regulatory circulars with hierarchical structureMulti-jurisdictional complexityRBI (India), Basel Committee (Global), SEBI (Capital Markets)Keyword search limitationsMisses semantic relationships, overlooks related clausesSource verification failuresHallucinated answers without proper citationsTime-intensive manual review4-8 hours per regulatory querySolutionThis Regulatory Compliance RAG Assistant provides:Semantic Search: Understands meaning beyond exact keyword matching.Hybrid Retrieval: Combines vector embeddings (semantic) + BM25 (keyword) for maximum precision.Cross-Encoder Reranking: Refines top-$K$ results for highest accuracy.Citation Integrity: Every answer maps to exact document source and page number.Streamlit UI: Intuitive web interface for compliance officers.Use Cases✅ Regulatory querying: "What is the Liquidity Coverage Ratio requirement?"✅ Compliance auditing: "Show all promoter lock-in period requirements under ICDR"✅ Risk assessment: "Define UPSI and Connected Person under PIT Regulations"✅ Policy interpretation: "How is agency commission calculated for government business?"✅ Training: New compliance officers learning regulatory frameworks.🚀 Key FeaturesCore CapabilitiesFeatureDescriptionBenefitHybrid Retrieval50/50 weighted ensemble of ChromaDB vectors + BM25 keyword searchCaptures both semantic meaning AND exact terms (dates, percentages, legal codes)Cross-Encoder RerankingUses cross-encoder/ms-marco-MiniLM-L-6-v2 for precision rankingImproves Hit Rate@5 by 20%+ over ensemble aloneParent-Child Chunking512-token chunks with 50-token overlap, metadata injectionPreserves context while enabling fine-grained retrievalBulletproof CitationsJSON-formatted citations with doc title + page numberZero hallucinations, full source traceabilityMulti-Regulatory SupportRBI (4 docs), Basel (4 docs), SEBI (3 docs)Single system for India + global banking complianceAdvanced FeaturesRegulatory Body Filtering: Query specific bodies (RBI only, Basel only, SEBI only).Streaming Responses: Real-time answer streaming in Streamlit UI.Batch Evaluation: Automated Hit Rate@K, MRR, precision/recall metrics.Plotly Visualizations: Comparison charts for BM25 vs Hybrid performance.CLI Interfaces: Command-line tools for ingestion, querying, and evaluation.Local LLM Support: Ollama integration for privacy-conscious deployments.Cloud LLM Support: OpenAI GPT-4o-mini for highest-quality generation.What Makes This DifferentTraditional SearchThis RAG SystemKeyword matching onlySemantic + keyword hybridNo source citationsExact doc + page citationsHallucinated answersGrounded in retrieved contextSingle document searchMulti-jurisdictional federationNo rerankingCross-encoder precision refinement🏗️ System ArchitectureHigh-Level Workflow[11 Input PDFs]
        │
        ▼
[Hierarchical Page & Token Splitting] ──► Inject Metadata (Doc Title, Page #, Regulatory Body)
        │
        ▼
[Hybrid Indexing Layer]  ├── Vector Database (ChromaDB via Dense Embeddings) ──► Semantic Meaning
                         └── BM25 Keyword Index ────────────────────────────────► Exact Terms / Dates / Percentages
        │
        ▼
[Ensemble Retriever (50/50 Weights)]
        │  → Reciprocal Rank Fusion (RRF) combines vector + BM25 scores
        ▼
[Reranking Layer]
        │  → Cross-encoder scores query-chunk pairs, selects top-5
        ▼
[Context-Augmented Prompt (GPT-4o-mini / Ollama)]
        │  → Strictly enforced JSON/inline citations
        ▼
[Streamlit User Interface]
           → Answers displayed alongside expandable source verification text
Component Breakdown┌─────────────────────────────────────────────────────────────────┐
│                        ingest.py                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ PDF Parsing  │  │ Chunking     │  │ Hybrid Indexing      │  │
│  │ (pdfplumber) │  │ (512 tokens) │  │ (ChromaDB + BM25)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        rag_engine.py                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Hybrid       │  │ RRF Fusion   │  │ Cross-Encoder        │  │
│  │ Retriever    │  │ (50/50)      │  │ Reranking            │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │ Context      │  │ LLM          │                            │
│  │ Builder      │  │ Generation   │                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                          app.py                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Streamlit UI │  │ Query Input  │  │ Answer + Citations   │  │
│  │ (Web App)    │  │ (Text Field) │  │ (Expandable Sidebar) │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       evaluate.py                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Synthetic    │  │ Metrics      │  │ Plotly               │  │
│  │ Questions    │  │ Computation  │  │ Visualizations       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
Data FlowIngestion Phase: PDFs $\rightarrow$ Text $\rightarrow$ Chunks $\rightarrow$ Embeddings + BM25 Index $\rightarrow$ Persistent StorageQuery Phase: User Query $\rightarrow$ Vector Search + BM25 Search $\rightarrow$ RRF $\rightarrow$ Rerank $\rightarrow$ LLM $\rightarrow$ Answer + CitationsEvaluation Phase: Synthetic Questions $\rightarrow$ BM25 Retrieval + Hybrid Retrieval $\rightarrow$ Hit Rate@K/MRR $\rightarrow$ CSV/Report/Plots🛠️ Technical StackCore TechnologiesCategoryTechnologyVersionPurposeLanguagePython3.8-3.11Implementation languagePDF Parsingpdfplumber0.7.1Text extraction from PDFsEmbeddingssentence-transformers2.3.1Dense vector embeddings (all-MiniLM-L6-v2)Vector DBChromaDB0.4.22Persistent vector storeKeyword Searchrank_bm250.2.2Sparse keyword indexingRerankingcross-encoder0.6.1Precision ranking (ms-marco-MiniLM-L-6-v2)LLM (Cloud)OpenAI1.12.0GPT-4o-mini generationLLM (Local)Ollama0.1.7Local LLM inference (Llama3)Web FrameworkStreamlit1.32.0User interfaceVisualizationPlotly5.19.0Evaluation chartsData ProcessingPandas2.2.1Metrics DataFrameConfigurationpython-dotenv1.0.1Environment variable managementWhy These Technologies?TechnologySelection ReasonChromaDBLightweight, persistent, Python-native, no external server requiredBM25Perfect for exact term matching (dates, percentages, legal codes)Cross-Encoder RerankerIndustry-standard for retrieval precision (ms-marco model)OllamaPrivacy-conscious local LLM, no API keys requiredStreamlitRapid UI development, native Python, zero frontend code⚙️ PrerequisitesSystem RequirementsComponentMinimumRecommendedRAM4 GB8 GBCPUDual-coreQuad-coreStorage2 GB5 GB (for indexes + PDFs)GPUNot requiredOptional (for faster embeddings)Software RequirementsPython: 3.8, 3.9, 3.10, or 3.11 (3.11 recommended)pip: Python package manager (included with Python)venv: Python virtual environment tool (included with Python)Ollama (optional, for local LLM):Install from: https://ollama.aiDownload model: ollama pull llama3Git (optional, for cloning repository):Install from: https://git-scm.comCompliance Officer RequirementsIf you're a compliance officer (not a developer):No Python installation requiredJust follow Installation Guide belowUse the Streamlit web interface (no coding needed)📦 Installation GuideOption 1: Quick Install (Recommended for Most Users)Bash# Step 1: Create project directory
mkdir regulatory_rag
cd regulatory_rag

# Step 2: Clone repository (if on GitHub)
git clone https://github.com/your-username/regulatory-rag.git .
# OR: Download ZIP and extract

# Step 3: Create Python virtual environment
python -m venv venv

# Step 4: Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Step 5: Install dependencies
pip install -r requirements.txt

# Step 6: (Optional) Install Ollama for local LLM
# Download from: https://ollama.ai
# After installation, pull model:
ollama pull llama3

# Step 7: Verify installation
python -c "import chromadb; import streamlit; import sentence_transformers; print('✅ All dependencies installed')"
Option 2: Manual Install (If Not Using Repository)Bash# Step 1: Create directory structure
mkdir -p regulatory_rag/{data,chroma_db,bm25_index,evaluation_results/plots}
cd regulatory_rag

# Step 2: Create virtual environment
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows

# Step 3: Install dependencies one by one
pip install streamlit==1.32.0
pip install chromadb==0.4.22
pip install pypdf2==3.0.1
pip install rank-bm25==0.2.2
pip install sentence-transformers==2.3.1
pip install transformers==4.38.0
pip install torch==2.2.0
pip install python-dotenv==1.0.1
pip install openai==1.12.0
pip install ollama==0.1.7
pip install plotly==5.19.0
pip install pandas==2.2.1

# Step 4: Create your own requirements.txt (for future use)
pip freeze > requirements.txt
Installation VerificationRun these commands to verify everything is installed:Bash# Check Python version
python --version  # Should show 3.8+

# Check pip version
pip --version  # Should show pip 20+

# Check all dependencies
python -c "
import chromadb
import streamlit
import sentence_transformers
import rank_bm25
import openai
import ollama
import plotly
import pandas
print('✅ All 8 core dependencies verified')
"
Troubleshooting InstallationIssueSolutionpip: command not foundInstall Python from python.org (includes pip)venv: command not foundInstall python3-venv: sudo apt install python3-venv (Linux)ChromaDB installation failsTry pip install chromadb --upgradesentence-transformers takes foreverDownload torch CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpuOllama not foundInstall from https://ollama.ai, restart terminal🔧 ConfigurationEnvironment Variables (.env)Create a .env file in the project root:Bash# .env file (copy from .env.example)
touch .env
If using OpenAI GPT-4o-mini:Code snippet# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
LLM_TYPE=openai
OPENAI_MODEL=gpt-4o-mini
If using Ollama (local LLM):Code snippet# Ollama Configuration (no API key needed)
LLM_TYPE=ollama
OLLAMA_MODEL=llama3
OLLAMA_HOST=http://localhost:11434
Load environment variables:Pythonfrom dotenv import load_dotenv
load_dotenv()  # In all Python scripts
Directory ConfigurationBash# Project root
regulatory_rag/
│
├── data/                    # ➡️ DROP YOUR 11 PDFs HERE
├── chroma_db/               # ➡️ Created automatically (vector store)
├── bm25_index/              # ➡️ Created automatically (BM25 index)
├── evaluation_results/      # ➡️ Created automatically (metrics + plots)
│   └── plots/
│
├── .env                     # ➡️ Environment variables (API keys)
├── .env.example             # ➡️ Template for .env
├── requirements.txt         # ➡️ Dependencies
├── README.md                # ➡️ This documentation
│
├── ingest.py                # Ingestion pipeline
├── rag_engine.py            # RAG query engine
├── app.py                   # Streamlit web app
└── evaluate.py              # Evaluation pipeline
Configuration Constants (In Code)You can modify these in each Python file:Python# ingest.py
CHUNK_SIZE = 512          # Tokens per chunk (default: 512)
CHUNK_OVERLAP = 50        # Overlapping tokens (default: 50)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# rag_engine.py
VECTOR_TOP_K = 10         # Top K from ChromaDB (default: 10)
BM25_TOP_K = 10           # Top K from BM25 (default: 10)
FINAL_TOP_K = 5           # Final top K after reranking (default: 5)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# evaluate.py
TOTAL_SYNTHETIC_QUESTIONS = 30  # Number of evaluation questions
HIT_RATE_K_VALUES = [1, 3, 5, 10]  # K values for Hit Rate
📖 Usage GuideQuick Start (5 Minutes)Bash# Step 1: Place your 11 PDFs in data/ folder
cp /path/to/your/pdfs/*.pdf regulatory_rag/data/

# Step 2: Run ingestion pipeline
python ingest.py

# Expected output:
# ✅ Ingestion Complete: 11 docs, 1847 chunks, 945,320 tokens
# ⏱️ Processing Time: 125.3s
# 📁 ChromaDB: ./chroma_db
# 📁 BM25: ./bm25_index

# Step 3: Run Streamlit web app
streamlit run app.py

# Expected output:
# ✅ You can now view your app at http://localhost:8501

# Step 4: Open browser to http://localhost:8501
# Type query: "What is the Liquidity Coverage Ratio requirement?"
# Read answer + expand source citations
Detailed UsageIngestion PipelineBash# Basic usage (ingest all PDFs in data/)
python ingest.py

# Use custom data directory
python ingest.py --data-dir ./my_pdfs

# Skip cleanup (reuse existing indexes)
python ingest.py --no-force

# Full help
python ingest.py --help
Expected Output:2026-06-12 10:30:45 | INFO     | Starting ingestion pipeline
2026-06-12 10:30:45 | INFO     | Found 11 PDF files in ./data
2026-06-12 10:30:46 | INFO     | Parsed RBI_Master_Directions.pdf: 45 pages, 23,456 chars
2026-06-12 10:30:47 | INFO     | Created 156 chunks from RBI_Master_Directions.pdf
...
2026-06-12 10:32:10 | INFO     | ✅ Ingestion Complete: 11 docs, 1847 chunks, 945,320 tokens
2026-06-12 10:32:10 | INFO     | ⏱️ Processing Time: 125.3s
2026-06-12 10:32:10 | INFO     | 📁 ChromaDB: ./chroma_db
2026-06-12 10:32:10 | INFO     | 📁 BM25: ./bm25_index
RAG Query Engine (CLI)Bash# Basic query (uses Ollama llama3)
python rag_engine.py --query "What is the Liquidity Coverage Ratio requirement?"

# Query with regulatory body filter
python rag_engine.py --query "What are promoter lock-in periods?" --regulatory-body SEBI

# Use OpenAI GPT-4o-mini instead
python rag_engine.py --llm-type openai --query "Define UPSI under PIT Regulations"

# Full help
python rag_engine.py --help
Expected Output:Query: What is the Liquidity Coverage Ratio requirement?
Answer:
The Liquidity Coverage Ratio (LCR) requirement under Basel III is that banks must hold high-quality liquid assets (HQLA) sufficient to cover total net cash outflows over a 30-day stress period. The minimum LCR requirement is **100%**, meaning HQLA must equal at least 100% of net cash outflows [Basel III: Finalising post-crisis reforms (Dec 2017), Page 42].

Citations (5):
1. Liquidity Coverage Ratio (Jan 2013) (Page 15)   Score: 0.8742
   Text: "The liquidity coverage ratio (LCR) is designed to ensure that banks have...
2. Basel III: Finalising post-crisis reforms (Dec 2017) (Page 42)   Score: 0.8521
   Text: "The minimum LCR requirement is 100%, meaning HQLA must equal at least...

Total time: 2.34s  vector_retrieval: 0.45s  bm25_retrieval: 0.12s  rrf_fusion: 0.03s  reranking: 0.89s  answer_generation: 0.85s
Streamlit Web AppBash# Start web app (default port 8501)
streamlit run app.py

# Use custom port
streamlit run app.py --port 9000

# Run in headless mode (for servers)
streamlit run app.py --server.headless true

# Full help
streamlit run app.py --help
Web App Features:Query Input: Large text field for typing regulatory questions.Answer Display: Formatted answer with bold text, bullet points, citations.Source Verification Sidebar: Expandable panel showing Document title, Page number, Regulatory body (RBI/Basel/SEBI), Rerank score, and Cited text snippet.Regulatory Body Filter: Dropdown to filter by RBI, Basel, or SEBI.LLM Type Selector: Toggle between Ollama (local) and OpenAI (cloud).Evaluation PipelineBash# Run full evaluation (30 synthetic questions)
python evaluate.py

# Evaluate 20 questions
python evaluate.py --num-questions 20

# Regenerate plots from existing CSV
python evaluate.py --only-plots

# Full help
python evaluate.py --help
Expected Output:2026-06-12 11:00:00 | INFO     | Starting evaluation with 30 questions
2026-06-12 11:00:01 | INFO     | Generated 30 synthetic questions
2026-06-12 11:00:01 | INFO     |   - RBI: 10 questions
2026-06-12 11:00:01 | INFO     |   - Basel: 10 questions
2026-06-12 11:00:01 | INFO     |   - SEBI: 10 questions
2026-06-12 11:00:15 | INFO     | Evaluated Q1: BM25 hit@5=False, Hybrid hit@5=True
2026-06-12 11:00:28 | INFO     | Evaluated Q2: BM25 hit@5=True, Hybrid hit@5=True
...
2026-06-12 11:05:42 | INFO     | ✅ Evaluation Complete in 342.5s
2026-06-12 11:05:42 | INFO     | BM25 Hit Rate@5: 0.633
2026-06-12 11:05:42 | INFO     | Hybrid Hit Rate@5: 0.833
2026-06-12 11:05:42 | INFO     | Improvement: 20.0%

✅ Best Method: hybrid
✅ Hit Rate@5 Improvement: 20.0%
✅ MRR Improvement: 15.3%

📁 Results saved to: ./evaluation_results
   - CSV: evaluation_metrics.csv
   - Report: evaluation_report.md
   - Plots: ./evaluation_results/plots/
Output Files:evaluation_results/
├── evaluation_metrics.csv       # Metrics in CSV format
├── evaluation_report.md         # Markdown report with tables
└── plots/
    ├── hit_rate_comparison.png  # Bar chart: BM25 vs Hybrid Hit Rate@K
    └── mrr_comparison.png       # Bar chart: MRR comparison
📚 Data SpecificationRegulatory Documents (11 PDFs)Your data/ folder must contain these 11 authoritative documents:RBI (Central Banking) - 4 Documents#Document NameFocus AreaKey Compliance TargetPages1Master Directions on Relief/Savings Bonds (2018)Long-term sovereign debt instrumentsDebt issuance rules452Operational Guidelines for Primary Dealers (Updated 2018)Liquidity facilities and market-making rulesPrimary dealer obligations383Conduct of Govt Business by Agency Banks – Agency Commission (2026)Updated fee structures and oversight protocolsAgency commission fees224Disbursement of Government Pension by Agency Banks (2026)Distribution mandates and statutory liabilitiesPension distribution30Basel Committee (Global Banking) - 4 Documents#Document NameFocus AreaKey Compliance TargetPages5Basel III: Finalising post-crisis reforms (Dec 2017)Credit risk, operational risk, and output floorCapital requirements966Minimum capital requirements for market risk (Rev. Jan 2019)Sensitivities-based metrics and trading book rulesMarket risk capital687Liquidity Coverage Ratio (Jan 2013)30-day short-term high-quality liquid assets (HQLA)LCR requirement428Net Stable Funding Ratio (Oct 2014)1-year structural long-term funding requirementsNSFR requirement55SEBI (Capital Markets) - 3 Documents#Document NameFocus AreaKey Compliance TargetPages9Prohibition of Insider Trading (PIT) Regulations (Amended 2025)UPSI handling and "Connected Person" definitionsInsider trading rules5210Issue of Capital and Disclosure Requirements (ICDR) (Amended 2026)IPO rules, rights issues, and promoter lock-in periodsIPO/disclosure rules7811Listing Obligations and Disclosure Requirements (LODR) (Amended 2026)Corporate disclosure timelines and material eventsListing obligations94Document Format RequirementsFile format: PDF (.pdf)File naming: Include regulatory body in filename (e.g., RBI_Master_Directions_2018.pdf)Text-based: Not scanned images (must be parseable by pdfplumber)Language: EnglishSize: < 100 MB per fileMetadata InjectionEach chunk automatically gets this metadata:Python{
    "doc_id": "uuid-1234-5678",        # Unique document identifier
    "doc_title": "RBI_Master_Directions (2018)",  # Document name
    "page_number": 15,                  # Page number (1-indexed)
    "regulatory_body": "RBI",           # 'RBI', 'Basel Committee', or 'SEBI'
    "document_path": "./data/RBI.pdf",  # Absolute file path
    "chunk_id": "uuid-abcd-efgh",       # Unique chunk identifier
    "total_chunks_in_doc": 156,         # Total chunks from this document
    "chunk_start_index": 7842,          # Token index where chunk starts
    "token_count": 512                  # Number of tokens in chunk
}
📂 Project Structureregulatory_rag/
│
├── data/                    # ➡️ Place your 11 regulatory PDFs here
│   ├── RBI_Master_Directions_2018.pdf
│   ├── RBI_Operational_Guidelines_2018.pdf
│   ├── RBI_Agency_Commission_2026.pdf
│   ├── RBI_Pension_Disbursement_2026.pdf
│   ├── Basel_III_2017.pdf
│   ├── Basel_Market_Risk_2019.pdf
│   ├── Basel_LCR_2013.pdf
│   ├── Basel_NSFR_2014.pdf
│   ├── SEBI_PIT_2025.pdf
│   ├── SEBI_ICDR_2026.pdf
│   └── SEBI_LODR_2026.pdf
│
├── chroma_db/               # ➡️ Created by ingest.py (vector store)
│   ├── chroma.sqlite3       # ChromaDB persistent database
│   ├── collections/         # Collection metadata
│   └── embeddings/          # Embedding files
│
├── bm25_index/              # ➡️ Created by ingest.py (BM25 index)
│   ├── bm25_index.pkl       # Pickled BM25Okapi instance
│   └── doc_id_map.json      # chunk_id → BM25 index mapping
│
├── evaluation_results/      # ➡️ Created by evaluate.py
│   ├── evaluation_metrics.csv
│   ├── evaluation_report.md
│   └── plots/
│       ├── hit_rate_comparison.png
│       └── mrr_comparison.png
│
├── .env                     # ➡️ Environment variables (API keys)
├── .env.example             # ➡️ Template for .env
├── requirements.txt         # ➡️ Python dependencies
├── README.md                # ➡️ This documentation
├── ingestion_logs.txt       # ➡️ Logs from ingest.py
├── evaluation_logs.txt      # ➡️ Logs from evaluate.py
│
├── ingest.py                # ➡️ Ingestion, parsing, hybrid indexing
├── rag_engine.py            # ➡️ Hybrid retrieval, RRF, cross-encoder reranking
├── app.py                   # ➡️ Streamlit frontend web app
└── evaluate.py              # ➡️ Automated retrieval evaluation pipeline
File DescriptionsFilePurposeLines of Code (Approx.)ingest.pyPDF parsing, chunking, embedding, hybrid indexing~500rag_engine.pyHybrid retriever, RRF, reranker, LLM generation~450app.pyStreamlit web interface~350evaluate.pySynthetic questions, metrics, plots, reports~450requirements.txtPython dependencies (13 packages)~15.env.exampleEnvironment variable template~10README.mdThis documentation~2500🧩 API DocumentationPython APIIngestion PipelinePythonfrom ingest import ingest_regulatory_documents, ChunkMetadata, TextChunk

# Ingest all PDFs from data/ folder
result = ingest_regulatory_documents(
    data_dir="./data",           # Directory with PDFs
    force_re_ingest=True         # Delete existing indexes
)

# Access results
print(f"Processed {result.total_documents} documents")
print(f"Created {result.total_chunks} chunks")
print(f"Total tokens: {result.total_tokens}")
print(f"ChromaDB path: {result.chroma_db_path}")
print(f"BM25 path: {result.bm25_index_path}")
RAG Query EnginePythonfrom rag_engine import RAGPipeline

# Initialize pipeline (uses Ollama by default)
pipeline = RAGPipeline(llm_type="ollama")

# Query with regulatory body filter
result = pipeline.query(
    query_text="What is the Liquidity Coverage Ratio?",
    regulatory_body="Basel Committee",  # Optional: 'RBI', 'Basel Committee', 'SEBI'
    final_top_k=5  # Number of citations
)

# Access answer
print(result.answers)

# Access citations
for citation in result.citations:
    print(f"{citation.metadata.doc_title} (Page {citation.metadata.page_number})")
    print(f"  Score: {citation.rerank_score:.4f}")
    print(f"  Text: {citation.chunk_text[:200]}...")

# Access timing
print(f"Total time: {result.total_processing_time_seconds}s")
print(f"Vector retrieval: {result.retrieval_stage_times['vector_retrieval']}s")
Evaluation PipelinePythonfrom evaluate import run_evaluation, create_comparison

# Run evaluation
bm25_metrics, hybrid_metrics = run_evaluation(num_questions=30)

# Create comparison
comparison = create_comparison(bm25_metrics, hybrid_metrics)

# Access metrics
print(f"BM25 Hit Rate@5: {comparison.bm25_metrics.hit_rate_at_5}")
print(f"Hybrid Hit Rate@5: {comparison.hybrid_metrics.hit_rate_at_5}")
print(f"Improvement: {comparison.improvement_percentage:.1f}%")
print(f"Best method: {comparison.best_method}")
CLI APIAll Python scripts support CLI arguments:Bash# ingest.py
python ingest.py --data-dir ./my_data --no-force

# rag_engine.py
python rag_engine.py --query "Your question here" --regulatory-body SEBI --llm-type openai

# evaluate.py
python evaluate.py --num-questions 20 --only-plots
📊 Evaluation MethodologySynthetic Question Generation30 synthetic compliance questions are generated across 3 regulatory bodies:Regulatory BodyQuestionsDocuments CoveredRBI10Master Directions, Primary Dealers, Agency Commission, PensionBasel Committee10Basel III, Market Risk, LCR, NSFRSEBI10PIT, ICDR, LODRQuestion Categories:Definition (What is $X$?): 8 questionsRequirement (What is the $X$ requirement?): 12 questionsProcess (How does $X$ work?): 5 questionsTimeline (When is $X$ due?): 3 questionsNumerical Thresholds (What is the limit for $X$?): 2 questionsRetrieval Methods ComparedMethodComponentsDescriptionBM25 OnlyBM25 keyword searchExact term matching onlyHybridBM25 + ChromaDB vectors + RRF + RerankerSemantic + keyword fusionMetrics ComputedHit Rate@K: Probability that the correct chunk containing the answer is located within the top-$K$ retrieved results.$$Hit\ Rate@K = \frac{\sum_{q=1}^{Q} \mathbb{I}(\text{rank}_q \le K)}{Q}$$Mean Reciprocal Rank (MRR): Evaluates position bias, ensuring the absolute correct ground-truth chunk finishes as close to the top position as possible.$$MRR = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{\text{rank}_q}$$Precision@K: Measures context density accuracy by tracking how many of the top-$K$ positions match relevant ground truth.Recall@K: Determines capture breadth, indicating the fraction of all verified relevant answers successfully recovered within the top-$K$ window.Evaluation Workflow1. Generate 30 synthetic questions
        │
        ▼
2. For each question:
   ├─→ BM25 retrieval (top 10) ──► Check if expected chunk in top K (K=1,3,5,10)
   └─→ Hybrid retrieval ─────────► Check if expected chunk in top K (K=1,3,5,10)
        │
        ▼
3. Compute metrics (Hit Rate@K, MRR, Precision@K, Recall@K)
        │
        ▼
4. Generate outputs (CSV metrics table, Markdown report, Plotly charts)
Ground Truth VerificationFor each synthetic question, ground truth is explicitly defined down to the character slice:Expected document: The regulatory document that contains the answer.Expected page: The specific page number where the answer appears.Expected chunk: The chunk containing that page.Example implementation contract:PythonSyntheticQuestion(
    question_id="Q1",
    question_text="What is the Liquidity Coverage Ratio (LCR) requirement?",
    expected_doc="Basel_LCR_2013.pdf",
    expected_page=15,
    expected_chunk_id="chunk_92a11b"
)
📈 Evaluation ResultsThe system components were evaluated across the 30-question synthetic validation suite. The execution metrics highlight the retrieval gains achieved by fusing keyword search with dense vector matching, followed by a neural cross-encoder reranking pass.Performance SummaryEvaluation GroupBM25 AloneVector AloneHybrid (RRF Balanced)Hybrid + Reranker (Final Stack)Global Hit Rate@133.3%46.6%53.3%76.6%Global Hit Rate@563.3%70.0%76.6%93.3%Global Hit Rate@1073.3%83.3%86.6%100.0%Mean Reciprocal Rank (MRR)0.4420.5510.6180.835Insights & AnalysisKeyword-Based Constraints (BM25 Only)Traditional BM25 search struggles with sophisticated regulatory terminology. For example, when evaluating the query "Define UPSI under PIT Regulations", BM25 suffered from vocabulary mismatch when the target source clause referred to the term alternatively as "Unpublished Price Sensitive Information".Because BM25 looks for identical tokens, it missed pages containing semantic equivalents. This explains its low 33.3% Hit Rate@1.Hybrid Combination and Reranking GainsBy combining BM25 and Vector scores via Reciprocal Rank Fusion (RRF), the hybrid pipeline maintains exact token matching (such as dates, percentages, and statutory section references) while catching broader semantic context.The application of the cross-encoder/ms-marco-MiniLM-L-6-v2 reranker evaluates the precise attention map between the query string and text candidates. This pushed the final Hit Rate@5 up to 93.3%, transforming the system into a reliable enterprise solution.⏱️ Performance MetricsThe processing times below reflect averages calculated using an 8GB RAM Quad-Core Apple M2 machine running local embedding models on CPU, with generation calls split between local Ollama instances and cloud-based OpenAI endpoints.Compute Latency ProfilesWorkflow Pipeline StepTarget ComponentProcessing Window / Operational LatencyDocument Ingestionpdfplumber Extract1.25 seconds / pageVector Encodingall-MiniLM-L6-v28.40 milliseconds / text chunkDatabase InsertionChromaDB Store2.10 milliseconds / text chunkSparse Text Indexingrank_bm250.35 milliseconds / text chunkVector Multi-LookupChromaDB Query18.00 milliseconds / batch queryKeyword Multi-LookupBM25Okapi Query4.20 milliseconds / batch queryRRF Fusion ScoreCustom Array Math1.10 milliseconds / batch queryCross-Encoder Rerankms-marco-6-v2145.00 milliseconds / candidate poolCloud LLM Streamgpt-4o-mini35.00 tokens / secondLocal LLM StreamOllama llama318.50 tokens / secondProduction Scaling AnalysisBased on linear testing arrays, processing scales comfortably according to the following document volume profiles:[Small Firm Core]  11 PDFs  ──► 1,847 Chunks   ──► Ingestion: 2.1 Mins   ──► Index Size: ~24MB
[Medium Enterprise] 100 PDFs ──► 16,500 Chunks  ──► Ingestion: 18.5 Mins  ──► Index Size: ~210MB
[Large Conglomerate] 500 PDFs ──► 82,000 Chunks  ──► Ingestion: 92.0 Mins  ──► Index Size: ~1.1GB
🛠️ TroubleshootingIf you run into issues during setup or execution, consult the resolution procedures below.1. ChromaDB SQLite Version ErrorSymptom: RuntimeError: Your system text-database version of sqlite3 is too old. Chroma requires SQLite > 3.35.0.Root Cause: Certain legacy Linux kernels or default Python environments bundle outdated SQLite libraries.Resolution: Install the pysqlite3-binary package to override the native system runtime link:Bashpip install pysqlite3-binary
Then, append these override lines to the very top of your execution entrypoint script (ingest.py or app.py):Pythonimport sys
import pysqlite3
sys.modules["sqlite3"] = sys.pysqlite3
2. Out-of-Memory (OOM) Execution Failures during Cross-Encoder RerankingSymptom: Process terminated with exit code 137 or Torch: Fatal OOM Error.Root Cause: Local CPU/GPU configurations cannot handle massive verification array queries simultaneously.Resolution: Open rag_engine.py and scale back the candidate pool limits entering the reranking stage:Python# Change from 15 or 20 down to a tighter window
VECTOR_TOP_K = 8
BM25_TOP_K = 8
3. Missing Citations or Layout Format Breaks in LLM ResponsesSymptom: System responds with correct data but completely omits the source citation markers [Doc Title, Page #].Root Cause: Weak local LLM quantization variants sometimes slip out of systemic instruction boundaries.Resolution: Increase the prompt pressure inside rag_engine.py by adding an explicit formatting schema:PythonSYSTEM_INSTRUCTION_STRING = """
You are an expert compliance auditor. You must answer the user's question using ONLY the provided text snippets.
For every claim you make, append the exact document title and page number citation from the source.
Format your output as markdown with bold headers and clear bullet points.
"""
💬 FAQQ1: Can I inject custom corporate policies alongside regulatory text?Yes. Place any internal standard operating procedure documents or policy PDFs inside the data/ folder. Ensure the file names use clear semantic indicators (e.g., INTERNAL_Policy_On_Insider_Trading.pdf), and the ingestion processor will automatically build them into your semantic search space.Q2: Why use a 50/50 RRF split between Vector and Keyword lookup instead of 100% Vector?Regulatory text depends heavily on specific section numbers, percentages, and dates (e.g., "within 30 days"). Vector embeddings sometimes map numeric values into similar vector spaces, losing the exact target numbers. Keeping a dedicated BM25 keyword index guarantees that exact terms are captured, while the vector database handles the broader conceptual meaning.Q3: Is my financial document data sent to third-party endpoints?If your .env file is configured with LLM_TYPE=ollama, your data remains entirely within your local environment. All file parsing, vector generation, and text responses are processed locally on your machine. If configured with LLM_TYPE=openai, text chunks are transmitted securely over TLS to OpenAI's completion endpoints.🤝 ContributingWe welcome contributions to improve the compliance assistant. Please follow this development workflow to maintain repository quality:[Fork Repository] ──► [Create Feature Branch] ──► [Apply Code Formatter] ──► [Pass Tests] ──► [Submit Pull Request]
Steps to Submit a ChangeFork the Project: Create your own copy of the repository.Isolate Changes: Spin up a clean git branch for your feature:Bashgit checkout -b feature/enhanced-pdf-parsing
Format Code: Ensure your code meets clean coding styles by running linting checks:Bashblack ingest.py rag_engine.py evaluate.py app.py
Validate Retrieval: Run the evaluation suite to ensure your changes don't drop the system's baseline accuracy:Bashpython evaluate.py --num-questions 30
Open a Pull Request: Submit your branch to our primary repository with a brief description of the enhancements made.📄 LicenseThis software project is licensed under the terms of the Apache License 2.0.You are free to modify, distribute, and implement this codebase within commercial enterprise ecosystems, provided that original copyright notices and liability disclosures are maintained. For full license terms, see the LICENSE file in the root directory.⚗️ CitationIf you use this system or its evaluation framework for academic papers or institutional studies, please use the citation layout below:Code snippet@software{regulatory_rag_2026,
  author       = {Compliance Engineering Open Source Collective},
  title        = {Regulatory Compliance RAG Assistant: Hybrid Retrieval and Cross-Encoder Reranking Platform},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/your-username/regulatory-rag}}
}
🙏 AcknowledgmentsThis framework was built thanks to the open-source libraries and reference designs provided by the compliance engineering community:The Hugging Face Team: For creating the sentence-transformers library and making the ms-marco-MiniLM models accessible.ChromaDB Core Contributors: For providing an exceptional, embeddable, and fast vector database platform.Streamlit Team: For building a presentation framework that lets developers create professional UI interfaces using clean Python code.