"""
ingest.py - Enterprise Ingestion Pipeline for Regulatory Compliance RAG Assistant
Handles robust extraction, structural metadata enrichment, hierarchical parent-child 
token chunking, and dual-indexing serialization (ChromaDB + BM25 sparse index).
"""

import os
import re
import json
import uuid
import pickle
import shutil
import logging
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

import pdfplumber
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

# Load environment configurations
load_dotenv()

# ==========================================
# 1. CONFIGURATION CONSTANTS
# ==========================================
CHUNK_SIZE = 512            # Target tokens per text chunk
CHUNK_OVERLAP = 50          # Overlapping tokens between consecutive chunks
MIN_CHUNK_LENGTH = 100      # Minimum characters required to keep a text chunk
MAX_PAGE_SIZE = 50000       # Maximum allowed characters per page before protective splitting
CHROMA_DB_PATH = "./chroma_db"
BM25_INDEX_PATH = "./bm25_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 32             # Processing chunk batch size for dense embedding generation
CHROMA_COLLECTION_NAME = "regulatory_documents"

# ==========================================
# 2. DATA CLASSES
# ==========================================
@dataclass
class ChunkMetadata:
    """Structured metadata for each individual text chunk."""
    doc_id: str
    doc_title: str
    page_number: int
    regulatory_body: str        # 'RBI', 'Basel Committee', or 'SEBI'
    document_path: str
    chunk_id: str
    total_chunks_in_doc: int
    chunk_start_index: int
    token_count: int

@dataclass
class TextChunk:
    """Represents a single parsed chunk of text with metadata and tokens."""
    text: str
    metadata: ChunkMetadata
    tokens: List[str]           # Tokenized version used strictly for BM25 tracking

@dataclass
class IngestionResult:
    """Summary metrics of the ingestion pipeline execution sequence."""
    total_documents: int
    successfully_processed: int
    failed_documents: int
    total_chunks: int
    total_tokens: int
    chroma_db_path: str
    bm25_index_path: str
    processing_time_seconds: float
    errors: List[str]

# ==========================================
# 3. LOGGING SETUP
# ==========================================
def setup_logging() -> logging.Logger:
    """Configure comprehensive logging with simultaneous file and console output."""
    logger = logging.getLogger("RAGIngestion")
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        return logger
        
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    file_handler = logging.FileHandler("ingestion_logs.txt", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# ==========================================
# 4. PDF PARSING FUNCTIONS
# ==========================================
def extract_pdf_text(pdf_path: str, logger: logging.Logger) -> Tuple[str, List[int]]:
    """Extract text contents from a target PDF with detailed error handling."""
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        logger.error(f"IO Failure: Target document path does not exist: {pdf_path}")
        raise FileNotFoundError(f"PDF document not found at: {pdf_path}")
        
    full_text_accumulator: List[str] = []
    page_start_indices: List[int] = []
    current_char_position = 0
    
    logger.info(f"Initiating binary text extraction sequence for: {path_obj.name}")
    
    try:
        with pdfplumber.open(path_obj) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                raise ValueError(f"Target document contains 0 valid pages: {pdf_path}")
                
            for page_idx, page in enumerate(pdf.pages):
                page_text = page.extract_text(layout=False)
                if page_text is None:
                    page_text = ""
                
                cleaned_page_text = "".join(ch for ch in page_text if ord(ch) >= 32 or ch in "\n\t")
                
                if len(cleaned_page_text) > MAX_PAGE_SIZE:
                    logger.warning(f"Truncation Warning: Page {page_idx + 1} bounds exceed limit constants.")
                    cleaned_page_text = cleaned_page_text[:MAX_PAGE_SIZE]
                
                page_start_indices.append(current_char_position)
                full_text_accumulator.append(cleaned_page_text)
                current_char_position += len(cleaned_page_text)
                
        complete_payload_string = "".join(full_text_accumulator)
        word_estimate = len(complete_payload_string.split())
        
        logger.debug(f"Document Metrics Summary [{path_obj.name}] - Pages: {total_pages} | Chars: {len(complete_payload_string)} | Estimated Words: {word_estimate}")
        return complete_payload_string, page_start_indices
        
    except Exception as raw_exception:
        logger.error(f"Critical execution fault during parsing phase of {path_obj.name}: {str(raw_exception)}")
        raise RuntimeError(f"Failed parsing PDF structures for {pdf_path}: {str(raw_exception)}")

def determine_regulatory_body(pdf_filename: str, logger: logging.Logger) -> str:
    """Classify structural provenance matrix based on legal compliance entity naming patterns."""
    fn_lower = pdf_filename.lower()
    
    rbi_triggers = ['rbi', 'reserve bank', 'central banking', 'agency bank', 'pension', 'bonds', 'dealers']
    basel_triggers = ['basel', 'crd', 'liquidity coverage', 'net stable', 'lcr', 'nsfr', 'reforms']
    sebi_triggers = ['sebi', 'insider trading', 'pit', 'icdr', 'lodr', 'listing', 'disclosure', 'securities']
    
    if any(trigger in fn_lower for trigger in rbi_triggers):
        return "RBI"
    elif any(trigger in fn_lower for trigger in basel_triggers):
        return "Basel Committee"
    elif any(trigger in fn_lower for trigger in sebi_triggers):
        return "SEBI"
    else:
        logger.warning(f"Classification Boundary Shift: Origin unknown for file context '{pdf_filename}'. Routing to default category.")
        return "Unknown"

def split_into_pages(text: str, page_starts: List[int]) -> List[Tuple[int, str]]:
    """Segment the massive raw extraction text stream back into clean page-level chunks."""
    segments: List[Tuple[int, str]] = []
    total_registered_pages = len(page_starts)
    
    for idx in range(total_registered_pages):
        start_char = page_starts[idx]
        end_char = page_starts[idx + 1] if idx + 1 < total_registered_pages else len(text)
        
        page_text = text[start_char:end_char]
        page_text = re.sub(r'\n{4,}', '\n\n\n', page_text)
        page_text = page_text.strip()
        
        if len(page_text) < 50:
            continue
            
        segments.append((idx + 1, page_text))
        
    return segments

# ==========================================
# 5. HIERARCHICAL TOKEN CHUNKING
# ==========================================
def tokenize_text(text: str) -> List[str]:
    """Tokenize standard compliance text using a precise character-preservation regex scanner."""
    text_lower = text.lower()
    token_pattern = re.compile(r'[a-z0-9]+(?:[-./_][a-z0-9]+)*|₹|%')
    return token_pattern.findall(text_lower)

def create_hierarchical_chunks(
    page_chunks: List[Tuple[int, str]],
    doc_title: str,
    doc_id: str,
    regulatory_body: str,
    document_path: str,
    logger: logging.Logger
) -> List[TextChunk]:
    """Transforms structural page layers into optimized child vector segments via sliding window parsing."""
    provisional_chunks: List[Tuple[str, List[str], int, int]] = []
    total_token_count = 0
    
    for page_num, page_text in page_chunks:
        tokens = tokenize_text(page_text)
        token_count = len(tokens)
        
        if token_count == 0:
            continue
            
        if token_count <= CHUNK_SIZE:
            provisional_chunks.append((page_text, tokens, page_num, 0))
            total_token_count += token_count
        else:
            start_token_idx = 0
            while start_token_idx < token_count:
                end_token_idx = start_token_idx + CHUNK_SIZE
                window_tokens = tokens[start_token_idx:end_token_idx]
                
                reconstructed_snippet = " ".join(window_tokens)
                
                if len(reconstructed_snippet) >= MIN_CHUNK_LENGTH:
                    provisional_chunks.append((reconstructed_snippet, window_tokens, page_num, start_token_idx))
                    total_token_count += len(window_tokens)
                    
                if end_token_idx >= token_count:
                    break
                    
                start_token_idx += (CHUNK_SIZE - CHUNK_OVERLAP)

    final_chunks: List[TextChunk] = []
    total_calculated_chunks = len(provisional_chunks)
    
    for text_payload, chunk_tokens, origin_page, start_idx in provisional_chunks:
        unique_chunk_uuid = str(uuid.uuid4())
        
        metadata_layer = ChunkMetadata(
            doc_id=doc_id,
            doc_title=doc_title,
            page_number=origin_page,
            regulatory_body=regulatory_body,
            document_path=os.path.abspath(document_path),
            chunk_id=unique_chunk_uuid,
            total_chunks_in_doc=total_calculated_chunks,
            chunk_start_index=start_idx,
            token_count=len(chunk_tokens)
        )
        
        final_chunks.append(TextChunk(
            text=text_payload,
            metadata=metadata_layer,
            tokens=chunk_tokens
        ))
        
    logger.debug(f"Hierarchical Chunk Generation [{doc_title}]: Extracted {total_calculated_chunks} text chunks encompassing {total_token_count} tokens.")
    return final_chunks

# ==========================================
# 6. EMBEDDING GENERATION LAYER
# ==========================================
class EmbeddingGenerator:
    """Manages dense vector sentence embeddings infrastructure with inline caching optimization layers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("RAGIngestion")
        model_resolution_path = f"sentence-transformers/{model_name}"
        
        self.logger.info(f"Loading dense semantic embedding transformer: {model_resolution_path}")
        self.model = SentenceTransformer(model_resolution_path)
        self.model.eval()
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.logger.info(f"Embedding infrastructure initialized. Vector Dimension mapping constraints: {self.embedding_dim}")

    def generate_embeddings(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        
        results_array = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        for idx, text in enumerate(texts):
            if text in self.embedding_cache:
                results_array[idx] = self.embedding_cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(idx)
                
        if uncached_texts:
            computed_vectors = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            for index_offset, target_global_idx in enumerate(uncached_indices):
                vector_payload = computed_vectors[index_offset]
                source_text = uncached_texts[index_offset]
                
                self.embedding_cache[source_text] = vector_payload
                results_array[target_global_idx] = vector_payload
                
        return results_array

    def embed_single(self, text: str) -> np.ndarray:
        return self.generate_embeddings([text])[0]

# ==========================================
# 7. CHROMADB INTEGRATION
# ==========================================
def create_chroma_db(
    chunks: List[TextChunk],
    embedding_generator: EmbeddingGenerator,
    db_path: str = CHROMA_DB_PATH,
    logger: Optional[logging.Logger] = None
) -> chromadb.Collection:
    """Constructs, initializes, and hydrates persistent local database vector store schemas."""
    logger = logger or logging.getLogger("RAGIngestion")
    logger.info(f"Initializing connection to Vector Storage at: {db_path}")
    
    persistent_client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(
            is_persistent=True,
            anonymized_telemetry=False
        )
    )
    
    collection = persistent_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": "RBI, Basel III, and SEBI structural core regulatory directives knowledge base.", "hnsw:space": "cosine"}
    )
    
    logger.info("Computing dense vectors and building index records...")
    raw_texts = [chunk.text for chunk in chunks]
    computed_embeddings = embedding_generator.generate_embeddings(raw_texts)
    
    ids_batch: List[str] = []
    documents_batch: List[str] = []
    metadatas_batch: List[Dict[str, Any]] = []
    embeddings_batch: List[List[float]] = []
    
    write_stride = 100
    total_records = len(chunks)
    
    for idx, chunk in enumerate(chunks):
        ids_batch.append(chunk.metadata.chunk_id)
        documents_batch.append(chunk.text)
        embeddings_batch.append(computed_embeddings[idx].tolist())
        
        flattened_meta = asdict(chunk.metadata)
        metadatas_batch.append(flattened_meta)
        
        if len(ids_batch) == write_stride or idx == total_records - 1:
            try:
                collection.add(
                    ids=ids_batch,
                    documents=documents_batch,
                    metadatas=metadatas_batch,
                    embeddings=embeddings_batch
                )
            except Exception as database_collision_err:
                logger.warning(f"Database write conflict detected: {str(database_collision_err)}. Retrying batch via targeted synchronization sequence...")
                collection.upsert(
                    ids=ids_batch,
                    documents=documents_batch,
                    metadatas=metadatas_batch,
                    embeddings=embeddings_batch
                )
                
            ids_batch.clear()
            documents_batch.clear()
            metadatas_batch.clear()
            embeddings_batch.clear()
            
    logger.info(f"Vector Database sync operations complete. Registered records count: {collection.count()}")
    return collection

# ==========================================
# 8. BM25 SPARSE INDEX INTEGRATION
# ==========================================
def create_bm25_index(
    chunks: List[TextChunk],
    index_path: str = BM25_INDEX_PATH,
    logger: Optional[logging.Logger] = None
) -> Tuple[BM25Okapi, Dict[str, int]]:
    """Constructs token correlation structural matrices and serializes sparse search files."""
    logger = logger or logging.getLogger("RAGIngestion")
    logger.info("Initializing serialization tasks for Lexical BM25 Sparse Inverted Index...")
    
    corpus_tokens = [chunk.tokens for chunk in chunks]
    
    for idx, tokens in enumerate(corpus_tokens):
        if len(tokens) < 5:
            logger.warning(f"Metadata Alert: Chunk index context structure '{chunks[idx].metadata.chunk_id}' shows sparse footprint ({len(tokens)} tokens).")
            
    bm25_engine_instance = BM25Okapi(corpus_tokens)
    
    doc_id_to_index_mapping = {
        chunk.metadata.chunk_id: idx for idx, chunk in enumerate(chunks)
    }
    
    output_directory = Path(index_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    sparse_index_file = output_directory / "bm25_index.pkl"
    mapping_payload_file = output_directory / "doc_id_map.json"
    
    try:
        with open(sparse_index_file, "wb") as sparse_out:
            pickle.dump(bm25_engine_instance, sparse_out)
            
        with open(mapping_payload_file, "w", encoding="utf-8") as mapping_out:
            json.dump(doc_id_to_index_mapping, mapping_out, indent=4)
            
        avg_doc_len = sum(len(t) for t in corpus_tokens) / len(corpus_tokens) if corpus_tokens else 0
        logger.info(f"Sparse inverted mapping payload saved. Avg document length: {avg_doc_len:.2f} tokens. Location: {output_directory.resolve()}")
        
        return bm25_engine_instance, doc_id_to_index_mapping
    except Exception as io_err:
        logger.error(f"Critical operational error writing sparse serialization variables to workspace disk: {str(io_err)}")
        raise OSError(f"Failed sparse indexing execution pipelines task tracking states: {str(io_err)}")

# ==========================================
# 9. MAIN PIPELINE WORKFLOW EXECUTION ENGINE
# ==========================================
def ingest_regulatory_documents(
    data_dir: str = "./data",
    force_re_ingest: bool = True,
    logger: Optional[logging.Logger] = None
) -> IngestionResult:
    """Main orchestration driver executing parsing pipeline functions sequentially."""
    logger = logger or setup_logging()
    execution_start_timestamp = time.time()
    
    error_tracking_registry: List[str] = []
    all_extracted_chunks_collector: List[TextChunk] = []
    
    target_data_path = Path(data_dir)
    logger.info("=========================================================================")
    logger.info(f"LAUNCHING ENTERPRISE REGULATORY COMPLIANCE RAG INGESTION PIPELINE")
    logger.info(f"Targeting Source Workspace: {target_data_path.resolve()}")
    logger.info("=========================================================================")
    
    if not target_data_path.exists() or not target_data_path.is_dir():
        msg = f"Incomplete Setup Environment: Directory target data path does not exist: {data_dir}"
        logger.critical(msg)
        return IngestionResult(0, 0, 1, 0, 0, CHROMA_DB_PATH, BM25_INDEX_PATH, 0.0, [msg])
        
    if force_re_ingest:
        cleanup_existing_indexes(CHROMA_DB_PATH, BM25_INDEX_PATH, logger)
        
    # FIX CODE BRANCH HERE: Standardized to catch both file strings (.pdf and .pdf.pdf)
    pdf_files_to_process = [
        f for f in os.listdir(target_data_path) 
        if f.lower().endswith('.pdf') or f.lower().endswith('.pdf.pdf')
    ]
    total_discovered_count = len(pdf_files_to_process)
    
    logger.info(f"Workspace Scan complete. Identified {total_discovered_count} source compliance files for structural ingestion processing.")
    
    if total_discovered_count == 0:
        msg = f"Validation Warning: Ingestion halts. No target PDF documents located inside folder path: {target_data_path.resolve()}"
        logger.warning(msg)
        return IngestionResult(0, 0, 0, 0, 0, CHROMA_DB_PATH, BM25_INDEX_PATH, 0.0, [msg])
        
    successful_runs_counter = 0
    
    for file_name in pdf_files_to_process:
        full_pdf_filesystem_path = target_data_path / file_name
        logger.info(f"Processing structural pipeline workflow states for: {file_name}")
        
        try:
            # Step 1: Text extraction passes
            raw_text_stream, markers_array = extract_pdf_text(str(full_pdf_filesystem_path), logger)
            
            # Step 2: Source mapping classification parameters
            assigned_regulatory_body = determine_regulatory_body(file_name, logger)
            
            # Step 3: Segment reconstitution layers
            reconstituted_page_structures = split_into_pages(raw_text_stream, markers_array)
            
            # Step 4: Token slicing logic tracking cycles
            internal_document_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_name))
            
            # SANITIZATION RULE UPDATED: Gracefully handle structural trailing duplication variants
            sanitized_title = Path(file_name).name
            if sanitized_title.lower().endswith('.pdf.pdf'):
                sanitized_title = sanitized_title[:-8]
            elif sanitized_title.lower().endswith('.pdf'):
                sanitized_title = sanitized_title[:-4]
            
            document_processed_chunks = create_hierarchical_chunks(
                page_chunks=reconstituted_page_structures,
                doc_title=sanitized_title,
                doc_id=internal_document_uuid,
                regulatory_body=assigned_regulatory_body,
                document_path=str(full_pdf_filesystem_path),
                logger=logger
            )
            
            all_extracted_chunks_collector.extend(document_processed_chunks)
            successful_runs_counter += 1
            logger.info(f"Successfully digested {file_name} -> Formatted {len(document_processed_chunks)} structural chunks.")
            
        except Exception as handling_trace_err:
            error_message_string = f"Pipeline Interruption [{file_name}]: Processing failure profile -> {str(handling_trace_err)}"
            logger.error(error_message_string)
            error_tracking_registry.append(error_message_string)
            continue
            
    total_compiled_chunks_count = len(all_extracted_chunks_collector)
    
    if total_compiled_chunks_count == 0:
        msg = "Critical Pipeline Halt: Processing cycles terminated. Extracted text chunk calculations resulted in 0 records."
        logger.error(msg)
        error_tracking_registry.append(msg)
        return IngestionResult(total_discovered_count, 0, len(error_tracking_registry), 0, 0, CHROMA_DB_PATH, BM25_INDEX_PATH, 0.0, error_tracking_registry)
        
    # Step 5: Embeddings initialization and deployment
    embedding_utility_engine = EmbeddingGenerator(model_name=EMBEDDING_MODEL, logger=logger)
    
    # Step 6: Database synchronization steps
    create_chroma_db(all_extracted_chunks_collector, embedding_utility_engine, CHROMA_DB_PATH, logger)
    
    # Step 7: Sparse Index structure operations
    create_bm25_index(all_extracted_chunks_collector, BM25_INDEX_PATH, logger)
    
    total_pipeline_tokens_count = sum(chunk.metadata.token_count for chunk in all_extracted_chunks_collector)
    total_execution_duration = time.time() - execution_start_timestamp
    
    final_ingestion_summary_report = IngestionResult(
        total_documents=total_discovered_count,
        successfully_processed=successful_runs_counter,
        failed_documents=len(error_tracking_registry),
        total_chunks=total_compiled_chunks_count,
        total_tokens=total_pipeline_tokens_count,
        chroma_db_path=CHROMA_DB_PATH,
        bm25_index_path=BM25_INDEX_PATH,
        processing_time_seconds=total_execution_duration,
        errors=error_tracking_registry
    )
    
    logger.info("=========================================================================")
    logger.info("✅ ENTERPRISE REGULATORY COMPLIANCE INGESTION SEQUENCE RUN COMPLETE")
    logger.info(f" - Total Discovered Documents: {final_ingestion_summary_report.total_documents}")
    logger.info(f" - Successfully Handled Profile Chunks: {final_ingestion_summary_report.successfully_processed}")
    logger.info(f" - Aggregated Error Fault Tracks: {final_ingestion_summary_report.failed_documents}")
    logger.info(f" - Persistent Vector Database Entries Written: {final_ingestion_summary_report.total_chunks}")
    logger.info(f" - Total Tokens Managed In Corpus Matrices: {final_ingestion_summary_report.total_tokens}")
    logger.info(f" - Total Measured Engine Execution Duration: {final_ingestion_summary_report.processing_time_seconds:.2f} seconds")
    logger.info("=========================================================================")
    
    return final_ingestion_summary_report

# ==========================================
# 10. UTILITY DISK MAINTENANCE FUNCTIONS
# ==========================================
def cleanup_existing_indexes(chroma_path: str = CHROMA_DB_PATH, bm25_path: str = BM25_INDEX_PATH, logger: Optional[logging.Logger] = None):
    """Purges prior physical output databases directory chains to enforce data integrity limits."""
    logger = logger or logging.getLogger("RAGIngestion")
    
    for path_target in [chroma_path, bm25_path]:
        path_obj = Path(path_target)
        if path_obj.exists():
            logger.info(f"System Maintenance Action: Purging historic database repository entity track: {path_obj.resolve()}")
            try:
                if path_obj.is_dir():
                    shutil.rmtree(path_obj)
                else:
                    path_obj.unlink()
            except Exception as system_maintenance_io_fault:
                logger.warning(f"File System Warning: Maintenance sweep encountered folder release lock parameters trace: {str(system_maintenance_io_fault)}")

def validate_pdf_count(data_dir: str, expected_count: int = 11, logger: Optional[logging.Logger] = None) -> List[str]:
    """Validates structural footprint counts of file collections located on machine infrastructure workspace disks."""
    logger = logger or logging.getLogger("RAGIngestion")
    target_path = Path(data_dir)
    
    if not target_path.exists():
        return []
        
    found_pdfs = [f for f in os.listdir(target_path) if f.lower().endswith('.pdf') or f.lower().endswith('.pdf.pdf')]
    actual_count = len(found_pdfs)
    
    if actual_count != expected_count:
        logger.warning(f"Audit Configuration Discrepancy: Workspace files directory matches count parameter: {actual_count} targets located. Expected baseline index parameter: {expected_count}")
    else:
        logger.info(f"Workspace Audit Verified: Perfect data structure match. All {expected_count} compliance manuals accounted for.")
        
    return found_pdfs

# ==========================================
# 11. COMMAND LINE INTERFACE (CLI) ENTRY POINT
# ==========================================
def main():
    """CLI execution wrapper tracking ingestion parameter profiles configurations adjustments."""
    logger_instance = setup_logging()
    
    cli_parser = argparse.ArgumentParser(
        description="Enterprise Ingestion Data Pipeline Matrix Utility Workspace Automation Script Framework Driver."
    )
    cli_parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Target folder directory destination containing compliance context document source PDFs."
    )
    cli_parser.add_argument(
        "--no-force",
        action="store_true",
        help="Disables system index wipe behaviors to reuse existing indexes."
    )
    
    parsed_arguments = cli_parser.parse_args()
    force_overwrite_flag = not parsed_arguments.no_force
    
    # Fire processing execution 
    ingest_regulatory_documents(
        data_dir=parsed_arguments.data_dir,
        force_re_ingest=force_overwrite_flag,
        logger=logger_instance
    )

if __name__ == "__main__":
    main()
