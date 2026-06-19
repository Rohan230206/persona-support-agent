import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Outdated SQLite override (essential for deployment environments like HF Spaces)
try:
    import sqlite3
    if sqlite3.sqlite_version_info < (3, 35, 0):
        try:
            import pysqlite3
            sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        except ImportError:
            pass
except ImportError:
    pass

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.utils import logger, retry_with_backoff, extract_text_from_pdf

class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function wrapper for ChromaDB using Gemini's text-embedding-004."""
    def __init__(self, client: genai.Client, model_name: str = config.EMBEDDING_MODEL):
        self.client = client
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=input
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings from Gemini: {e}")
            raise e

class LocalRAGPipeline:
    def __init__(self, db_dir: str = str(config.CHROMA_DB_PATH)):
        self.db_dir = db_dir
        self.offline_mode = not bool(config.GEMINI_API_KEY)
        
        if self.offline_mode:
            logger.warning("GEMINI_API_KEY is missing. RAG pipeline initialized in OFFLINE fallback mode.")
            self.offline_chunks: List[Dict[str, Any]] = []
            return
            
        logger.info(f"Initializing persistent ChromaDB client at: {db_dir}")
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.embedding_function = GeminiEmbeddingFunction(self.client)
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        
        # Setup collection with Cosine similarity distance metric
        self.collection = self.chroma_client.get_or_create_collection(
            name="support_kb",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_document(self, filepath: Path) -> bool:
        """
        Loads, splits, and embeds a document.
        Uses incremental indexing based on file modification times.
        Supports both online ChromaDB and offline fallback structures.
        """
        filename = filepath.name
        mtime = os.path.getmtime(filepath)
        
        # Load content depending on extension
        ext = filepath.suffix.lower()
        content = ""
        try:
            if ext == ".pdf":
                content = extract_text_from_pdf(str(filepath))
            elif ext in [".txt", ".md"]:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                logger.warning(f"Skipping file '{filename}': unsupported format.")
                return False
        except Exception as e:
            logger.error(f"Error reading file '{filename}': {e}")
            return False

        if not content.strip():
            logger.warning(f"File '{filename}' is empty. Skipping index.")
            return False

        # Split document
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=40,
            length_function=len
        )
        chunks = splitter.split_text(content)
        logger.info(f"Splitting '{filename}' into {len(chunks)} chunks.")

        if self.offline_mode:
            # In offline mode, clear old chunks of same file and load new ones
            self.offline_chunks = [c for c in self.offline_chunks if c["source"] != filename]
            for idx, chunk in enumerate(chunks):
                self.offline_chunks.append({
                    "text": chunk,
                    "source": filename,
                    "chunk_index": idx,
                    "last_modified": mtime
                })
            logger.info(f"Offline indexed '{filename}' with {len(chunks)} chunks.")
            return True

        # Online mode: Check if document is already indexed
        try:
            existing = self.collection.get(where={"source": filename})
            if existing and existing["ids"]:
                first_meta = existing["metadatas"][0]
                indexed_mtime = first_meta.get("last_modified", 0.0)
                
                if indexed_mtime == mtime:
                    logger.info(f"File '{filename}' already indexed and unchanged. Skipping.")
                    return False
                else:
                    logger.info(f"File '{filename}' has changed since last indexing. Re-indexing.")
                    self.collection.delete(where={"source": filename})
        except Exception as e:
            logger.warning(f"Error checking indexing ledger for '{filename}': {e}. Proceeding to re-ingest.")

        # Batch add to collection
        ids = [f"{filename}_chunk_{idx}" for idx in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": idx, "last_modified": mtime} for idx in range(len(chunks))]
        
        try:
            self.collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
            )
            logger.info(f"Successfully ingested and indexed file: '{filename}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add document chunks to Chroma for file '{filename}': {e}")
            return False

    def ingest_data_directory(self, data_dir: Path = config.DATA_DIR) -> int:
        """
        Scans directory for documents and indexes them.
        """
        logger.info(f"Scanning data directory for ingestion: {data_dir}")
        if not data_dir.exists():
            logger.warning(f"Data directory '{data_dir}' does not exist.")
            return 0
            
        supported_extensions = {".txt", ".md", ".pdf"}
        files_indexed = 0
        
        for file_path in data_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                success = self.ingest_document(file_path)
                if success:
                    files_indexed += 1
                    
        logger.info(f"Ingestion complete. Newly indexed/updated {files_indexed} documents.")
        return files_indexed

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search (Online) or word-overlap keyword matching (Offline).
        """
        logger.info(f"Retrieving top {top_k} contexts for query: '{query[:50]}'")
        
        if self.offline_mode:
            # Fallback to local string matching
            query_words = set(query.lower().split())
            stopwords = {"the", "a", "an", "is", "are", "of", "to", "in", "for", "on", "with", "at", "it", "this", "that", "your", "my"}
            query_words = {w for w in query_words if len(w) > 2 and w not in stopwords}
            
            matches = []
            for chunk in self.offline_chunks:
                chunk_words = set(chunk["text"].lower().split())
                intersection = query_words.intersection(chunk_words)
                
                score = 0.0
                if query_words:
                    overlap_ratio = len(intersection) / len(query_words)
                    # Scale to realistic similarity score (0.45 threshold check)
                    score = round(0.40 + 0.55 * overlap_ratio, 4) if overlap_ratio > 0 else 0.10
                else:
                    score = 0.10
                    
                matches.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "score": score
                })
                
            # Sort by similarity score descending
            matches.sort(key=lambda x: x["score"], reverse=True)
            retrieved = matches[:top_k]
            logger.info(f"[Offline] Retrieved {len(retrieved)} relevant contexts.")
            return retrieved

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            retrieved_chunks = []
            if results and results["documents"] and len(results["documents"][0]) > 0:
                for idx in range(len(results["documents"][0])):
                    doc_text = results["documents"][0][idx]
                    metadata = results["metadatas"][0][idx]
                    distance = results["distances"][0][idx] if results["distances"] else 1.0
                    similarity_score = round(max(0.0, 1.0 - distance), 4)
                    
                    retrieved_chunks.append({
                        "text": doc_text,
                        "source": metadata.get("source", "Unknown"),
                        "score": similarity_score
                    })
                    
            logger.info(f"Retrieved {len(retrieved_chunks)} relevant contexts.")
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Error querying Chroma DB: {e}")
            return []
