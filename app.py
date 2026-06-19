import os
import sys
import time
import random
import json
import logging
import functools
from pathlib import Path
from typing import List, Dict, Any, Literal, Tuple, TypeVar, Callable
from pydantic import BaseModel, Field
import gradio as gr
from pypdf import PdfReader

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
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. CONFIGURATION SECTION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SIMILARITY_THRESHOLD = 0.45
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"
LOG_FILE = PROJECT_ROOT / "app.log"

def validate_environment() -> bool:
    return bool(GEMINI_API_KEY)

# ==========================================
# 2. UTILITIES SECTION
# ==========================================
def setup_logger(name: str = "support_agent") -> logging.Logger:
    """Sets up console and file logging."""
    app_logger = logging.getLogger(name)
    if not app_logger.handlers:
        app_logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        app_logger.addHandler(console_handler)
        # File Handler
        try:
            file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
            file_handler.setFormatter(formatter)
            app_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file log handler: {e}")
    return app_logger

logger = setup_logger()
F = TypeVar('F', bound=Callable[..., Any])

def retry_with_backoff(max_retries: int = 5, initial_delay: float = 1.0, backoff_factor: float = 2.0) -> Callable[[F], F]:
    """Decorator that retries a function with exponential backoff on failure."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for function '{func.__name__}': {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay + random.uniform(0.0, 0.5))
                    delay *= backoff_factor
            logger.error(f"All {max_retries} attempts failed for function '{func.__name__}'.")
            raise last_exception  # type: ignore
        return wrapper  # type: ignore
    return decorator

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content page-by-page from a PDF document."""
    logger.info(f"Extracting text from PDF: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        extracted_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        full_text = "\n".join(extracted_text)
        logger.info(f"Successfully extracted {len(full_text)} characters from {pdf_path}")
        return full_text
    except Exception as e:
        logger.error(f"Failed to read PDF file {pdf_path}: {e}")
        raise RuntimeError(f"Error parsing PDF file: {e}")

# ==========================================
# 3. PERSONA CLASSIFIER SECTION
# ==========================================
class PersonaClassification(BaseModel):
    persona: Literal["Technical Expert", "Frustrated User", "Business Executive"] = Field(
        description="The classified customer persona. Must be exactly one of the three options."
    )
    confidence: float = Field(
        description="The confidence score of the classification, ranging from 0.0 to 1.0."
    )
    reasoning: str = Field(
        description="Brief explanation of the linguistic cues, sentiment, or tone that led to this classification."
    )

@retry_with_backoff(max_retries=3, initial_delay=1.0)
def classify_customer_persona(user_message: str) -> Dict[str, Any]:
    """Analyzes the user's message and classifies it into exactly one of three target personas."""
    logger.info(f"Classifying persona for message: {user_message[:60]}...")
    
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set. Using rule-based offline classification fallback.")
        query_lower = user_message.lower()
        if any(w in query_lower for w in ["clear cookies", "hour", "loading", "frozen", "fix this", "demand", "refund", "duplicate", "immediate"]):
            return {
                "persona": "Frustrated User",
                "confidence": 0.85,
                "reasoning": "[Offline Rule] Message contains high-urgency or negative sentiment key phrases."
            }
        elif any(w in query_lower for w in ["bearer token", "header", "database", "api", "config", "pool", "integration", "port"]):
            return {
                "persona": "Technical Expert",
                "confidence": 0.90,
                "reasoning": "[Offline Rule] Message contains technical protocols, configurations, or system terms."
            }
        elif any(w in query_lower for w in ["uptime", "timeline", "disputes", "executive", "roi", "sla", "operational"]):
            return {
                "persona": "Business Executive",
                "confidence": 0.88,
                "reasoning": "[Offline Rule] Message contains operational, timeline, or business impact metrics."
            }
        return {
            "persona": "Frustrated User",
            "confidence": 0.50,
            "reasoning": "[Offline Rule] Defaulting to Frustrated User."
        }
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    system_instruction = (
        "You are an advanced classification engine. Your task is to analyze the "
        "sentiment, vocabulary, and tone of an incoming support message and classify "
        "it into exactly one of three customer personas:\n"
        "1. 'Technical Expert': Uses jargon, asks about APIs, code, logs, configs, databases, "
        "detailed configuration steps, or specific error messages.\n"
        "2. 'Frustrated User': Uses emotional language, exclamation marks, expresses urgency, "
        "demands immediate resolution, or complains about issues wasting time.\n"
        "3. 'Business Executive': Focuses on business impact, SLA, ROI, timelines, "
        "or high-level resolution. Keeps responses brief and direct.\n\n"
        "Provide your evaluation strictly in the requested JSON structure."
    )
    
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=PersonaClassification,
                temperature=0.1
            )
        )
        if hasattr(response, 'parsed') and response.parsed:
            classification_data = response.parsed
            result = {
                "persona": classification_data.persona,
                "confidence": float(classification_data.confidence),
                "reasoning": classification_data.reasoning
            }
        else:
            data = json.loads(response.text)
            result = {
                "persona": data.get("persona", "Frustrated User"),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": data.get("reasoning", "")
            }
        logger.info(f"Successfully classified as: '{result['persona']}' (Confidence: {result['confidence']})")
        return result
    except Exception as e:
        logger.error(f"Error during customer persona classification: {e}")
        return {
            "persona": "Frustrated User",
            "confidence": 0.5,
            "reasoning": f"Classification system encountered error: {e}. Defaulting to Frustrated User."
        }

# ==========================================
# 4. RAG PIPELINE SECTION
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function wrapper for ChromaDB using Gemini's text-embedding-004."""
    def __init__(self, client: genai.Client, model_name: str = EMBEDDING_MODEL):
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
    def __init__(self, db_dir: str = str(CHROMA_DB_PATH)):
        self.db_dir = db_dir
        self.offline_mode = not bool(GEMINI_API_KEY)
        
        if self.offline_mode:
            logger.warning("GEMINI_API_KEY is missing. RAG pipeline initialized in OFFLINE fallback mode.")
            self.offline_chunks: List[Dict[str, Any]] = []
            return
            
        logger.info(f"Initializing persistent ChromaDB client at: {db_dir}")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.embedding_function = GeminiEmbeddingFunction(self.client)
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="support_kb",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_document(self, filepath: Path) -> bool:
        """Loads, splits, and embeds a document. Supports incremental indexing."""
        filename = filepath.name
        mtime = os.path.getmtime(filepath)
        
        ext = filepath.suffix.lower()
        content = ""
        try:
            if ext == ".pdf":
                content = extract_text_from_pdf(str(filepath))
            elif ext in [".txt", ".md"]:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                return False
        except Exception as e:
            logger.error(f"Error reading file '{filename}': {e}")
            return False

        if not content.strip():
            return False

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        chunks = splitter.split_text(content)

        if self.offline_mode:
            self.offline_chunks = [c for c in self.offline_chunks if c["source"] != filename]
            for idx, chunk in enumerate(chunks):
                self.offline_chunks.append({
                    "text": chunk,
                    "source": filename,
                    "chunk_index": idx,
                    "last_modified": mtime
                })
            return True

        try:
            existing = self.collection.get(where={"source": filename})
            if existing and existing["ids"]:
                first_meta = existing["metadatas"][0]
                indexed_mtime = first_meta.get("last_modified", 0.0)
                if indexed_mtime == mtime:
                    return False
                self.collection.delete(where={"source": filename})
        except Exception as e:
            logger.warning(f"Error checking indexing ledger: {e}")

        ids = [f"{filename}_chunk_{idx}" for idx in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": idx, "last_modified": mtime} for idx in range(len(chunks))]
        
        try:
            self.collection.add(ids=ids, documents=chunks, metadatas=metadatas)
            return True
        except Exception as e:
            logger.error(f"Failed to add document chunks to Chroma for file '{filename}': {e}")
            return False

    def ingest_data_directory(self, data_dir: Path = DATA_DIR) -> int:
        """Scans directory for documents and indexes them."""
        if not data_dir.exists():
            return 0
        supported_extensions = {".txt", ".md", ".pdf"}
        files_indexed = 0
        for file_path in data_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                success = self.ingest_document(file_path)
                if success:
                    files_indexed += 1
        return files_indexed

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs semantic similarity search (Online) or word-overlap keyword matching (Offline)."""
        if self.offline_mode:
            query_words = set(query.lower().split())
            stopwords = {"the", "a", "an", "is", "are", "of", "to", "in", "for", "on", "with", "at", "it", "this", "that", "your", "my"}
            query_words = {w for w in query_words if len(w) > 2 and w not in stopwords}
            matches = []
            for chunk in self.offline_chunks:
                chunk_words = set(chunk["text"].lower().split())
                intersection = query_words.intersection(chunk_words)
                score = 0.10
                if query_words:
                    overlap_ratio = len(intersection) / len(query_words)
                    score = round(0.40 + 0.55 * overlap_ratio, 4) if overlap_ratio > 0 else 0.10
                matches.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "score": score
                })
            matches.sort(key=lambda x: x["score"], reverse=True)
            return matches[:top_k]

        try:
            results = self.collection.query(query_texts=[query], n_results=top_k)
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
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Error querying Chroma DB: {e}")
            return []

# ==========================================
# 5. ESCALATION ENGINE SECTION
# ==========================================
class HandoffReport(BaseModel):
    persona: str = Field(description="The customer's communication persona.")
    issue_summary: str = Field(description="A clear, professional summary of the customer's issue.")
    retrieved_sources: List[str] = Field(description="List of file names of retrieved documents.")
    confidence_score: float = Field(description="The maximum RAG retrieval similarity score.")
    recommended_action: str = Field(description="Specific, actionable next steps for the human agent.")

def contains_sensitive_keywords(query: str) -> Tuple[bool, str]:
    sensitive_words = ["billing", "refund", "charge", "payment dispute", "legal", "account ownership"]
    query_lower = query.lower()
    for word in sensitive_words:
        if word in query_lower:
            return True, f"Contains sensitive keyword: '{word}'"
    return False, ""

def check_repeated_frustration(session_personas: List[str], current_persona: str) -> bool:
    if current_persona != "Frustrated User":
        return False
    if len(session_personas) > 0 and session_personas[-1] == "Frustrated User":
        return True
    return False

@retry_with_backoff(max_retries=3, initial_delay=1.0)
def generate_handoff_json(query: str, persona: str, retrieved_sources: List[str], confidence_score: float) -> Dict[str, Any]:
    """Calls Gemini to generate a structured JSON handoff report."""
    logger.info("Generating structured handoff report via Gemini...")
    if not GEMINI_API_KEY:
        return {
            "persona": persona,
            "issue_summary": f"User query: '{query[:100]}...'",
            "retrieved_sources": retrieved_sources,
            "confidence_score": confidence_score,
            "recommended_action": "Manually review the user request and contact customer."
        }
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    system_instruction = (
        "You are a customer support triage agent. Generate a detailed, professional "
        "handoff report for a human agent. Explain the customer's problem clearly in the "
        "issue_summary, list any retrieved sources, and recommend a specific, actionable step."
    )
    prompt = (
        f"Customer Message: {query}\nDetected Persona: {persona}\n"
        f"Retrieved Document Sources: {retrieved_sources}\nRetrieval Cosine Similarity Score: {confidence_score}\n"
    )
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=HandoffReport,
                temperature=0.1
            )
        )
        if hasattr(response, 'parsed') and response.parsed:
            handoff_data = response.parsed
            result = {
                "persona": handoff_data.persona,
                "issue_summary": handoff_data.issue_summary,
                "retrieved_sources": handoff_data.retrieved_sources,
                "confidence_score": float(handoff_data.confidence_score),
                "recommended_action": handoff_data.recommended_action
            }
        else:
            result = json.loads(response.text)
        return result
    except Exception as e:
        logger.error(f"Error generating handoff report: {e}")
        return {
            "persona": persona,
            "issue_summary": f"User is reporting issues. Original query: {query}",
            "retrieved_sources": retrieved_sources,
            "confidence_score": confidence_score,
            "recommended_action": "Triage immediately. Check database records or billing status for user."
        }

def evaluate_escalation(
    query: str,
    top_similarity_score: float,
    current_persona: str,
    session_personas: List[str],
    retrieved_sources: List[str]
) -> Tuple[bool, str, Dict[str, Any] | None]:
    """Evaluates all triggers for escalation."""
    if top_similarity_score < SIMILARITY_THRESHOLD:
        reason = f"Top retrieval similarity score ({top_similarity_score}) is below the threshold ({SIMILARITY_THRESHOLD})."
        handoff = generate_handoff_json(query, current_persona, retrieved_sources, top_similarity_score)
        return True, reason, handoff
        
    is_sensitive, kw_reason = contains_sensitive_keywords(query)
    if is_sensitive:
        reason = f"Query contains sensitive keywords. Triggered by: {kw_reason}."
        handoff = generate_handoff_json(query, current_persona, retrieved_sources, top_similarity_score)
        return True, reason, handoff
        
    if check_repeated_frustration(session_personas, current_persona):
        reason = "Repeated user frustration detected over consecutive interactions."
        handoff = generate_handoff_json(query, current_persona, retrieved_sources, top_similarity_score)
        return True, reason, handoff
        
    return False, "", None

# ==========================================
# 6. RESPONSE GENERATOR SECTION
# ==========================================
@retry_with_backoff(max_retries=3, initial_delay=1.0)
def generate_adapted_response(query: str, persona: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates response matching user persona, grounded in context."""
    if not context_chunks:
        return {
            "escalated": True,
            "response": "I apologize, but I couldn't find any documents in our knowledge base related to your query.",
            "reason": "Knowledge base returned empty context."
        }
        
    if not GEMINI_API_KEY:
        best_score = max([c["score"] for c in context_chunks]) if context_chunks else 0.0
        if best_score < SIMILARITY_THRESHOLD:
            return {
                "escalated": True,
                "response": "I apologize, but the available information in our database is insufficient to fully resolve your request.",
                "reason": f"Top similarity score ({best_score}) is below threshold ({SIMILARITY_THRESHOLD}) in offline mode."
            }
            
        top_chunk = context_chunks[0]
        snippet = top_chunk["text"].strip()
        source = top_chunk["source"]
        
        if persona == "Technical Expert":
            response = (
                f"[OFFLINE EXPERT ADAPTATION] - System Configuration Lead\n\n"
                f"Root-Cause Analysis:\n"
                f"Troubleshooting specifications retrieved from local document '{source}':\n\n"
                f"```\n{snippet}\n```\n\n"
                f"Configuration Steps:\n"
                f"1. Open connection settings file.\n"
                f"2. Ensure authentication parameters and request headers match the specifications.\n"
                f"3. Validate that your network client properly manages exceptions."
            )
        elif persona == "Frustrated User":
            short_detail = snippet.split('\n')[0] if '\n' in snippet else snippet[:200]
            response = (
                f"[OFFLINE EMPATHY ADAPTATION] - Customer Care Specialist\n\n"
                f"I completely understand how frustrating it is when the system doesn't load and you are stuck waiting. "
                f"I am here to guide you through this process and help get you resolved right away!\n\n"
                f"Based on our '{source}' guide, please try the following steps:\n"
                f"- **Step 1**: Check the details: {short_detail}\n"
                f"- **Step 2**: Clear your browser's history or refresh the screen (press Ctrl+F5 on Windows).\n"
                f"- **Step 3**: Try opening a private window to verify if it is a temporary cache error.\n\n"
                f"We are here to support you, so please let me know if these steps resolve the issue!"
            )
        else:  # Business Executive
            response = (
                f"[OFFLINE EXECUTIVE ADAPTATION] - Client Relations Director\n\n"
                f"Here is the high-level business summary regarding your request. The operational policy and timelines "
                f"defined in our documentation ('{source}') indicate:\n\n"
                f"- **Resolution Timeline**: Review and dispute investigations are resolved within 5-7 business days.\n"
                f"- **Operational Impact**: Access remains active during investigations as long as core usage fees are settled."
            )
        return {"escalated": False, "response": response, "reason": ""}

    context_blocks = [f"Document [{chunk['source']}] Chunk {idx + 1}:\n{chunk['text']}" for idx, chunk in enumerate(context_chunks)]
    context_text = "\n\n".join(context_blocks)
    
    if persona == "Technical Expert":
        persona_instructions = (
            "You are a Senior Systems Engineer and Technical Support Lead. Provide "
            "a highly detailed, technical, and structured response. Include root-cause explanations, "
            "configuration setups, precise API configurations, database pool parameters, or HTTP code blocks."
        )
    elif persona == "Frustrated User":
        persona_instructions = (
            "You are a reassuring, warm, and deeply empathetic Customer Care Specialist. "
            "Begin your response immediately with sincere, caring validation of their difficulty. "
            "Use simple, direct language, avoid technical jargon, and list steps using bullet points."
        )
    else:  # Business Executive
        persona_instructions = (
            "You are a concise, formal Client Relations Director. Provide a brief executive summary "
            "answering the query. Focus on business outcomes, timelines, SLAs, and organizational impact."
        )
        
    system_instruction = (
        f"{persona_instructions}\n\n"
        "CRITICAL SYSTEM RULES:\n"
        "1. You must answer the user's query using ONLY the facts explicitly provided in the 'FACTUAL CONTEXT DOCUMENTS' section below.\n"
        "2. Do not assume, extrapolate, or use outside knowledge. Do not hallucinate any details.\n"
        "3. If the provided context documents do not contain enough facts, output exactly '[INSUFFICIENT_CONTEXT]' and nothing else."
    )
    
    prompt = f"FACTUAL CONTEXT DOCUMENTS:\n{context_text}\n\nUSER QUERY: {query}\n"
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2
            )
        )
        response_text = response.text.strip()
        if "[INSUFFICIENT_CONTEXT]" in response_text or response_text == "[INSUFFICIENT_CONTEXT]":
            return {
                "escalated": True,
                "response": "I apologize, but the available information is insufficient to fully resolve your request.",
                "reason": "Insufficient factual context in knowledge base."
            }
        return {"escalated": False, "response": response_text, "reason": ""}
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        return {"escalated": True, "response": "An unexpected error occurred. Escalating to human support.", "reason": f"Generator exception: {e}"}

# ==========================================
# 7. GRADIO WORKSPACE ORCHESTRATOR
# ==========================================
from generate_kb import generate_all

# Startup Check & Ingest
try:
    existing_files = list(DATA_DIR.glob("*"))
    if len(existing_files) < 15:
        generate_all()
except Exception as e:
    logger.error(f"Error checking data: {e}")
    generate_all()

try:
    pipeline = LocalRAGPipeline()
    pipeline.ingest_data_directory()
except Exception as e:
    logger.error(f"Failed to run RAG pipeline ingestion: {e}")
    pipeline = None

def process_query(message: str, history: list, session_personas: list):
    if not message.strip():
        return "", history, session_personas, "N/A", 0.0, 0.0, "✅ Normal Operations", {}, "*No document chunks retrieved.*"

    if pipeline is None:
        err_msg = "System is running in offline mode. Please configure GEMINI_API_KEY."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": err_msg})
        return "", history, session_personas, "N/A", 0.0, 0.0, "🚨 Handoff Required", {"error": err_msg}, "*No document chunks retrieved.*"

    try:
        classification = classify_customer_persona(message)
        persona = classification.get("persona", "Frustrated User")
        conf = classification.get("confidence", 0.5)
        
        retrieved_chunks = pipeline.retrieve_context(message, top_k=3)
        top_score = max([chunk["score"] for chunk in retrieved_chunks]) if retrieved_chunks else 0.0
        retrieved_sources = list(set([chunk["source"] for chunk in retrieved_chunks]))
        
        sources_md = "### Top Retrieved Chunks\n\n"
        if retrieved_chunks:
            for idx, chunk in enumerate(retrieved_chunks):
                sources_md += f"**{idx+1}. {chunk['source']}** (Similarity Score: `{chunk['score']:.4f}`)\n"
                sources_md += f"> {chunk['text'].strip()}\n\n"
        else:
            sources_md += "*No document chunks retrieved.*"
            
        should_escalate, esc_reason, handoff_json = evaluate_escalation(
            query=message, top_similarity_score=top_score, current_persona=persona,
            session_personas=session_personas, retrieved_sources=retrieved_sources
        )
        
        final_response = ""
        escalation_status = "✅ Normal Operations"
        
        if should_escalate:
            escalation_status = "🚨 Handoff Required"
            final_response = "I apologize, but I am unable to locate the precise solution to your request. I am connecting you with a live human support specialist."
        else:
            gen_result = generate_adapted_response(message, persona, retrieved_chunks)
            if gen_result.get("escalated", False):
                should_escalate = True
                escalation_status = "🚨 Handoff Required"
                final_response = "I apologize, but I am unable to locate the precise solution to your request. I am connecting you with a live human support specialist."
                handoff_json = generate_handoff_json(message, persona, retrieved_sources, top_score)
            else:
                final_response = gen_result.get("response", "")

        session_personas.append(persona)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": final_response})
        handoff_display = handoff_json if handoff_json else {}
        
        return "", history, session_personas, persona, conf, top_score, escalation_status, handoff_display, sources_md
    except Exception as e:
        logger.error(f"Error executing user query processing: {e}")
        err_msg = f"System Error: {e}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "An unexpected system error occurred. We are routing you to support."})
        return "", history, session_personas, "N/A", 0.0, 0.0, "🚨 Handoff Required", {"error": err_msg}, "*No document chunks retrieved.*"

def reset_session():
    return [], [], "N/A", 0.0, 0.0, "✅ Normal Operations", {}, "*No document chunks retrieved.*"

with gr.Blocks(title="Persona-Adaptive Support Agent") as demo:
    session_personas_state = gr.State(value=[])
    gr.Markdown(
        """
        # 🤖 Persona-Adaptive Customer Support Agent
        ### Enterprise AI Support Assistant with Dynamic Persona Adaptation, Vector-guided RAG, and Human Handoff.
        """
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Support Session Log", height=500)
            query_input = gr.Textbox(
                label="Type your message...",
                placeholder="Ask about credentials, OAuth, billing charges, rate limits, webhooks, or database configs...",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Send Message", variant="primary")
                clear_btn = gr.Button("Clear Chat", variant="secondary")
            gr.Markdown(
                """
                > **Notice**: The agent uses Gemini RAG to query local helpdesk articles. 
                > Sensitive inquiries (payment details, refunds, duplicate charges) or queries matching low-confidence data are escalated to human agents immediately.
                """
            )
        with gr.Column(scale=2, elem_classes="diagnostic-container"):
            gr.Markdown("## 📊 Agent Diagnostics Dashboard")
            with gr.Row():
                persona_val = gr.Textbox(label="Detected User Persona", value="N/A", interactive=False)
                persona_conf = gr.Number(label="Classification Confidence", value=0.00, precision=2, interactive=False)
            with gr.Row():
                retrieval_conf = gr.Number(label="Retrieval Cosine Score", value=0.0000, precision=4, interactive=False)
                esc_status = gr.Textbox(label="System Escalation Status", value="✅ Normal Operations", interactive=False)
            with gr.Tab("Retrieved RAG Sources"):
                sources_display = gr.Markdown(value="*No document chunks retrieved.*", line_breaks=True)
            with gr.Tab("Escalation Handoff JSON"):
                handoff_display = gr.JSON(label="Structured JSON Handoff", value={})

    submit_btn.click(
        fn=process_query,
        inputs=[query_input, chatbot, session_personas_state],
        outputs=[query_input, chatbot, session_personas_state, persona_val, persona_conf, retrieval_conf, esc_status, handoff_display, sources_display]
    )
    query_input.submit(
        fn=process_query,
        inputs=[query_input, chatbot, session_personas_state],
        outputs=[query_input, chatbot, session_personas_state, persona_val, persona_conf, retrieval_conf, esc_status, handoff_display, sources_display]
    )
    clear_btn.click(
        fn=reset_session,
        inputs=None,
        outputs=[chatbot, session_personas_state, persona_val, persona_conf, retrieval_conf, esc_status, handoff_display, sources_display]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
