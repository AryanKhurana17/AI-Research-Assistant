"""
PDF Knowledge Base using DuckDB as the vector store.

Architecture:
    DuckDBVectorStore  -- Encapsulates all DuckDB vector operations (CRUD + search)
    DocumentProcessor  -- Handles PDF reading and text chunking
    KnowledgeBase      -- Orchestrates the full RAG pipeline (index + search)

Design Decision: We build the vector store manually on DuckDB rather than
using a wrapper library. This gives full control over the schema, search
logic, and lets us show retrieved context transparently.
"""
import os
import uuid
from typing import List, Dict, Any, Optional

import duckdb
from sentence_transformers import SentenceTransformer
from agno.knowledge.reader.pdf_reader import PDFReader

from src.config import (
    PDF_PATH, DUCKDB_PATH, DATA_DIR,
    EMBEDDING_MODEL_ID, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
)
from src.logging_config import get_logger

logger = get_logger("knowledge")


class DuckDBVectorStore:
    """Manages vector storage and similarity search in DuckDB.

    Responsibilities:
        - Table creation and schema management
        - Inserting document chunks with embeddings
        - Cosine similarity search over stored vectors
    """

    TABLE_NAME = "document_chunks"

    def __init__(self, db_path: str, embedding_dim: int = 384):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Create and return a new DuckDB connection."""
        return duckdb.connect(self.db_path)

    def init_table(self) -> None:
        """Create the document_chunks table if it does not exist."""
        conn = self._connect()
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                id          VARCHAR PRIMARY KEY,
                content     VARCHAR,
                page_num    INTEGER,
                source      VARCHAR,
                embedding   FLOAT[{self.embedding_dim}]
            )
        """)
        conn.close()

    def clear(self) -> None:
        """Drop and recreate the table (used during re-indexing)."""
        conn = self._connect()
        conn.execute(f"DROP TABLE IF EXISTS {self.TABLE_NAME}")
        conn.close()
        self.init_table()

    def insert_batch(self, chunks: List[Dict[str, Any]], embeddings: list) -> int:
        """Insert a batch of chunks with their embeddings.

        Args:
            chunks: List of dicts with keys: id, content, page_num, source.
            embeddings: Corresponding embedding vectors (numpy arrays or lists).

        Returns:
            Number of rows inserted.
        """
        conn = self._connect()
        for chunk, emb in zip(chunks, embeddings):
            emb_list = emb.tolist() if hasattr(emb, "tolist") else emb
            conn.execute(
                f"INSERT INTO {self.TABLE_NAME} (id, content, page_num, source, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                [chunk["id"], chunk["content"], chunk["page_num"], chunk["source"], emb_list],
            )
        count = conn.execute(f"SELECT COUNT(*) FROM {self.TABLE_NAME}").fetchone()[0]
        conn.close()
        return count

    def search(self, query_embedding: list, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find the top-k most similar chunks using cosine similarity.

        Args:
            query_embedding: The query vector as a list of floats.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: content, page_num, source, score.
        """
        conn = self._connect()
        self.init_table()  # ensure table exists even if empty

        results = conn.execute(
            f"""
            SELECT
                content,
                page_num,
                source,
                array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) AS score
            FROM {self.TABLE_NAME}
            ORDER BY score DESC
            LIMIT ?
            """,
            [query_embedding, top_k],
        ).fetchall()
        conn.close()

        return [
            {"content": row[0], "page_num": row[1], "source": row[2], "score": round(row[3], 4)}
            for row in results
        ]

    def count(self) -> int:
        """Return the number of stored chunks."""
        conn = self._connect()
        self.init_table()
        result = conn.execute(f"SELECT COUNT(*) FROM {self.TABLE_NAME}").fetchone()[0]
        conn.close()
        return result


class DocumentProcessor:
    """Reads PDFs and splits them into overlapping text chunks.

    Responsibilities:
        - PDF text extraction (via Agno's PDFReader)
        - Character-based chunking with configurable size and overlap
    """

    def __init__(self, pdf_path: str, chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._reader = PDFReader()

    def read_pdf(self) -> list:
        """Extract text from the PDF, returning one document per page."""
        return self._reader.read(str(self.pdf_path))

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks by character count.

        Uses a sliding window so each chunk stays within the embedding
        model's 256-token limit (~1000 characters).
        """
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def process(self) -> List[Dict[str, Any]]:
        """Read the PDF and return all chunks with metadata.

        Returns:
            List of dicts with keys: id, content, page_num, source.
        """
        docs = self.read_pdf()
        all_chunks: List[Dict[str, Any]] = []

        for page_idx, doc in enumerate(docs):
            page_chunks = self.chunk_text(doc.content)
            for chunk in page_chunks:
                all_chunks.append({
                    "id": str(uuid.uuid4()),
                    "content": chunk,
                    "page_num": page_idx + 1,
                    "source": os.path.basename(str(self.pdf_path)),
                })

        return all_chunks


class KnowledgeBase:
    """Orchestrates the full RAG pipeline: indexing and retrieval.

    Composes DuckDBVectorStore, DocumentProcessor, and SentenceTransformer
    into a single interface for the rest of the application.

    Usage:
        kb = KnowledgeBase()
        kb.index()                              # one-time indexing
        results = kb.search("cross-validation") # retrieval
    """

    def __init__(
        self,
        vector_store: Optional[DuckDBVectorStore] = None,
        document_processor: Optional[DocumentProcessor] = None,
        embedding_model_id: str = EMBEDDING_MODEL_ID,
    ):
        self.vector_store = vector_store or DuckDBVectorStore(
            db_path=str(DUCKDB_PATH),
            embedding_dim=384,
        )
        self.document_processor = document_processor or DocumentProcessor(
            pdf_path=str(PDF_PATH),
        )
        self._embedder: Optional[SentenceTransformer] = None
        self._embedding_model_id = embedding_model_id

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-load the embedding model (downloads ~90 MB on first use)."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(self._embedding_model_id)
        return self._embedder

    def index(self, recreate: bool = True) -> int:
        """Run the full indexing pipeline: read -> chunk -> embed -> store.

        Args:
            recreate: If True, clears existing data before indexing.

        Returns:
            Total number of chunks indexed.
        """
        # Step 1: Read and chunk
        logger.info("Reading PDF from %s", self.document_processor.pdf_path)
        chunks = self.document_processor.process()
        logger.info(
            "Created %d chunks (size=%d, overlap=%d)",
            len(chunks), self.document_processor.chunk_size,
            self.document_processor.chunk_overlap,
        )

        # Step 2: Embed
        logger.info("Generating embeddings...")
        texts = [c["content"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        # Step 3: Store
        logger.info("Storing in DuckDB...")
        if recreate:
            self.vector_store.clear()
        count = self.vector_store.insert_batch(chunks, embeddings)

        logger.info("Indexed %d chunks into %s", count, self.vector_store.db_path)
        return count

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Embed the query and retrieve the most relevant chunks.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of result dicts with keys: content, page_num, source, score.
        """
        logger.info("search() called with query: '%s' (top_k=%d)", query, top_k)
        if self.vector_store.count() == 0:
            logger.info("Vector database is empty or missing. Auto-indexing PDF document...")
            self.index(recreate=True)
        query_embedding = self.embedder.encode(query).tolist()
        results = self.vector_store.search(query_embedding, top_k)
        if results:
            logger.info(
                "search() returned %d results (top score: %.4f, lowest: %.4f)",
                len(results), results[0]["score"], results[-1]["score"],
            )
        else:
            logger.warning("search() returned no results for: '%s'", query)
        return results

    @staticmethod
    def format_results(results: List[Dict[str, Any]]) -> str:
        """Format search results into a readable string for the LLM and user."""
        if not results:
            return "No relevant context found in the knowledge base."

        sections = []
        for i, r in enumerate(results, 1):
            sections.append(
                f"[Chunk {i}] (Page {r['page_num']}, Score: {r['score']})\n"
                f"{r['content']}"
            )
        return "\n\n".join(sections)


# Global cache for the knowledge base
_cached_kb = None


def get_knowledge_base() -> KnowledgeBase:
    """Get or create the singleton KnowledgeBase instance.
    
    This implements lazy loading so the vector store and embedding models
    are only initialized once per application lifecycle, improving performance.
    """
    global _cached_kb
    if _cached_kb is None:
        logger.info("Initializing global KnowledgeBase instance (lazy load)")
        _cached_kb = KnowledgeBase()
    return _cached_kb
