"""
RAG Pipeline — Retrieval-Augmented Generation over local documents.

Supports: PDF, TXT, Markdown, DOCX
Vector store: FAISS (local, no server needed)
Embeddings: OpenAI text-embedding-3-small (or HuggingFace fallback)
"""

import os
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
}


class RAGPipeline:
    """
    Manages document ingestion and vector search.
    
    Workflow:
        1. Load documents from a directory
        2. Split into chunks
        3. Embed and store in FAISS
        4. Search via similarity

    Usage:
        rag = RAGPipeline()
        rag.ingest_directory("docs/")
        results = rag.search("What is LangChain?")
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self.vectorstore: FAISS | None = None
        self._load_or_init()

    def _load_or_init(self):
        """Load existing FAISS index or create a fresh one."""
        index_path = settings.FAISS_INDEX_PATH

        if os.path.exists(index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"Loaded existing FAISS index from '{index_path}'")
                return
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}. Starting fresh.")

        # Auto-ingest if docs folder exists
        docs_path = settings.DOCS_DIR
        if os.path.exists(docs_path):
            logger.info(f"Auto-ingesting documents from '{docs_path}'")
            self.ingest_directory(docs_path)
        else:
            logger.info("No documents directory found. Vector store is empty.")
            self.vectorstore = None

    def _load_file(self, filepath: str) -> list[Document]:
        """Load a single file based on its extension."""
        ext = Path(filepath).suffix.lower()
        loader_cls = SUPPORTED_EXTENSIONS.get(ext)

        if loader_cls is None:
            logger.warning(f"Unsupported file type: {filepath}")
            return []

        try:
            loader = loader_cls(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(filepath)
            return docs
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return []

    def ingest_directory(self, directory: str) -> int:
        """
        Ingest all supported documents from a directory.
        Returns the number of chunks added.
        """
        all_docs = []
        path = Path(directory)

        for filepath in path.rglob("*"):
            if filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs = self._load_file(str(filepath))
                all_docs.extend(docs)
                logger.info(f"Loaded: {filepath.name} ({len(docs)} pages)")

        if not all_docs:
            logger.warning("No documents found to ingest.")
            return 0

        chunks = self.splitter.split_documents(all_docs)
        logger.info(f"Split into {len(chunks)} chunks")

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        # Persist index
        self.vectorstore.save_local(settings.FAISS_INDEX_PATH)
        logger.info(f"FAISS index saved to '{settings.FAISS_INDEX_PATH}'")

        return len(chunks)

    def ingest_text(self, text: str, source: str = "manual") -> int:
        """Ingest raw text directly (e.g. from API, clipboard)."""
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self.splitter.split_documents([doc])

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        self.vectorstore.save_local(settings.FAISS_INDEX_PATH)
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[Document]:
        """Run a similarity search and return the top-k documents."""
        if self.vectorstore is None:
            logger.warning("Vector store is empty. Ingest documents first.")
            return []

        results = self.vectorstore.similarity_search(query, k=k)
        logger.debug(f"RAG search '{query[:50]}' → {len(results)} results")
        return results

    def search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Search and return documents with relevance scores."""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search_with_score(query, k=k)

    @property
    def doc_count(self) -> int:
        """Approximate number of chunks in the vector store."""
        if self.vectorstore is None:
            return 0
        return self.vectorstore.index.ntotal
