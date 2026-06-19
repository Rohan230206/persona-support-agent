import os
import sys
import json
from pathlib import Path

# Adjust path to import src modules
sys.path.append(str(Path(__file__).resolve().parent))

from src import config
from src.utils import logger, setup_logger
from src.classifier import classify_customer_persona
from src.rag_pipeline import LocalRAGPipeline
from src.escalator import evaluate_escalation, generate_handoff_json
from src.generator import generate_adapted_response
from generate_kb import generate_all

def run_test_suite():
    print("=" * 70)
    print(" PERSONA-ADAPTIVE CUSTOMER SUPPORT AGENT - TEST SUITE")
    print("=" * 70)
    
    # 1. Check Configuration
    if not config.validate_environment():
        print("WARNING: GEMINI_API_KEY environment variable is missing.")
        print("The test suite will execute in OFFLINE fallback mode with local string matching.")
        print("-" * 70)
    else:
        print("[1/4] Environment validation: PASSED")
    
    # 2. Check Data & Ingest
    existing_files = list(config.DATA_DIR.glob("*"))
    if len(existing_files) < 15:
        print("[2/4] Support files missing. Triggering auto-generation of knowledge base...")
        generate_all()
    else:
        print("[2/4] Support files check: PASSED (found 15+ documents)")
        
    print("[3/4] Initializing ChromaDB vector store and ingesting docs...")
    pipeline = LocalRAGPipeline()
    newly_indexed = pipeline.ingest_data_directory()
    print(f"      Chroma DB Ingestion: COMPLETE (Processed {newly_indexed} new/updated files)")
    
    # 3. Test Scenarios Definition
    scenarios = [
        {
            "id": 1,
            "name": "Cookie Clearing Issue (Frustrated User)",
            "query": "Where is the guide to clear cookies? It's been an hour and nothing is loading on your interface!",
            "expected_persona": "Frustrated User",
            "expected_escalate": False
        },
        {
            "id": 2,
            "name": "Bearer Token Authentication (Technical Expert)",
            "query": "What are the header parameter requirements for your bearer token auth implementation?",
            "expected_persona": "Technical Expert",
            "expected_escalate": False
        },
        {
            "id": 3,
            "name": "Billing Timeline Question (Business Executive)",
            "query": "Our operational uptime is decreasing. We need a timeline of when billing disputes are resolved.",
            "expected_persona": "Business Executive",
            "expected_escalate": False
        },
        {
            "id": 4,
            "name": "Database Integration Issue (Technical Expert)",
            "query": "I'm experiencing an issue with your database integration that's causing internal errors.",
            "expected_persona": "Technical Expert",
            "expected_escalate": False
        },
        {
            "id": 5,
            "name": "Billing & Refund Request (Escalation Trigger)",
            "query": "My billing statement has unexpected duplicate charges. I demand an immediate refund!",
            "expected_persona": "Frustrated User",
            "expected_escalate": True
        }
    ]
    
    print("\n[4/4] Executing scenarios verification loop...")
    print("-" * 70)
    
    session_personas = []
    
    for tc in scenarios:
        print(f"\nScenario #{tc['id']}: {tc['name']}")
        print(f"  Query: '{tc['query']}'")
        
        # A. Classification
        classification = classify_customer_persona(tc['query'])
        persona = classification.get("persona", "Frustrated User")
        conf = classification.get("confidence", 0.0)
        reasoning = classification.get("reasoning", "")
        
        print(f"  -> Detected Persona: {persona} (Confidence: {conf})")
        print(f"  -> Reasoning: {reasoning}")
        
        # B. Retrieval
        retrieved = pipeline.retrieve_context(tc['query'], top_k=3)
        top_score = max([chunk["score"] for chunk in retrieved]) if retrieved else 0.0
        sources = list(set([chunk["source"] for chunk in retrieved]))
        
        print(f"  -> Top Similarity Score: {top_score:.4f} (Sources: {sources})")
        
        # C. Escalation Evaluation
        should_escalate, reason, handoff = evaluate_escalation(
            query=tc['query'],
            top_similarity_score=top_score,
            current_persona=persona,
            session_personas=session_personas,
            retrieved_sources=sources
        )
        
        # D. Response Generation
        if should_escalate:
            print("  -> Escalation Status: [ESCALATED] (Handoff JSON Generated)")
            print(f"  -> Escalation Reason: {reason}")
            print(f"  -> Handoff JSON Report:\n{json.dumps(handoff, indent=4)}")
        else:
            print("  -> Escalation Status: [NORMAL]")
            gen_result = generate_adapted_response(tc['query'], persona, retrieved)
            if gen_result.get("escalated", False):
                print("  -> Escalation Status: [ESCALATED] (Triggered during generation)")
                print(f"     Reason: {gen_result.get('reason')}")
                handoff = generate_handoff_json(tc['query'], persona, sources, top_score)
                print(f"  -> Handoff JSON Report:\n{json.dumps(handoff, indent=4)}")
            else:
                print(f"  -> Adapted Response Output:\n{gen_result.get('response')}")
                
        # Record session persona
        session_personas.append(persona)
        print("-" * 70)
        
    print("\nVerification Test Suite Execution Completed.")
    print("=" * 70)

if __name__ == "__main__":
    run_test_suite()
