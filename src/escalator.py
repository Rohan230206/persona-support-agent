import json
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src import config
from src.utils import logger, retry_with_backoff

class HandoffReport(BaseModel):
    persona: str = Field(description="The customer's communication persona.")
    issue_summary: str = Field(description="A clear, professional summary of the customer's issue or question.")
    retrieved_sources: List[str] = Field(description="List of file names of retrieved documents.")
    confidence_score: float = Field(description="The maximum RAG retrieval similarity score.")
    recommended_action: str = Field(description="Specific, actionable next steps for the human customer support agent.")

def contains_sensitive_keywords(query: str) -> Tuple[bool, str]:
    """
    Checks if the query contains any of the specified sensitive keywords.
    Returns:
        (bool, reason)
    """
    sensitive_words = [
        "billing", "refund", "charge", "payment dispute", 
        "legal", "account ownership"
    ]
    query_lower = query.lower()
    
    # Exact keyword or substring matches
    for word in sensitive_words:
        if word in query_lower:
            return True, f"Contains sensitive keyword: '{word}'"
            
    return False, ""

def check_repeated_frustration(session_personas: List[str], current_persona: str) -> bool:
    """
    Checks if the user has shown repeated frustration.
    Returns True if the current persona is 'Frustrated User' and the previous
    persona was also 'Frustrated User'.
    """
    if current_persona != "Frustrated User":
        return False
        
    # Check if the last recorded persona in the active session history was also Frustrated User
    if len(session_personas) > 0 and session_personas[-1] == "Frustrated User":
        logger.info("Repeated frustration detected. Triggering escalation.")
        return True
        
    return False

@retry_with_backoff(max_retries=3, initial_delay=1.0)
def generate_handoff_json(
    query: str, 
    persona: str, 
    retrieved_sources: List[str], 
    confidence_score: float
) -> Dict[str, Any]:
    """
    Calls Gemini to generate a structured JSON handoff report.
    """
    logger.info("Generating structured handoff report via Gemini...")
    
    if not config.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY missing. Constructing static handoff JSON.")
        return {
            "persona": persona,
            "issue_summary": f"User query: '{query[:100]}...'",
            "retrieved_sources": retrieved_sources,
            "confidence_score": confidence_score,
            "recommended_action": "Manually review the user request and contact customer."
        }
        
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    system_instruction = (
        "You are a customer support triage agent. Generate a detailed, professional "
        "handoff report for a human agent. Explain the customer's problem clearly in the "
        "issue_summary, list any retrieved sources, and recommend a specific, actionable "
        "remediation step in the recommended_action field."
    )
    
    prompt = (
        f"Customer Message: {query}\n"
        f"Detected Persona: {persona}\n"
        f"Retrieved Document Sources: {retrieved_sources}\n"
        f"Retrieval Cosine Similarity Score: {confidence_score}\n"
    )
    
    try:
        response = client.models.generate_content(
            model=config.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=HandoffReport,
                temperature=0.1
            )
        )
        
        # Access the parsed model dump or fall back to json loading of text
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
            
        logger.info("Handoff report generated successfully.")
        return result
        
    except Exception as e:
        logger.error(f"Error generating handoff report via Gemini: {e}")
        # Programmatic fallback matching the required schema
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
    """
    Evaluates all triggers for escalation:
    1. Similarity score < 0.45
    2. Sensitive topic keyword detection
    3. Repeated frustration in session history
    
    Returns:
        (should_escalate, reason, handoff_json_or_none)
    """
    # 1. Similarity Check
    if top_similarity_score < config.SIMILARITY_THRESHOLD:
        reason = f"Top retrieval similarity score ({top_similarity_score}) is below the threshold ({config.SIMILARITY_THRESHOLD})."
        logger.info(f"Escalating: {reason}")
        handoff = generate_handoff_json(query, current_persona, retrieved_sources, top_similarity_score)
        return True, reason, handoff
        
    # 2. Keyword Check
    is_sensitive, kw_reason = contains_sensitive_keywords(query)
    if is_sensitive:
        reason = f"Query contains sensitive keywords. Triggered by: {kw_reason}."
        logger.info(f"Escalating: {reason}")
        handoff = generate_handoff_json(query, current_persona, retrieved_sources, top_similarity_score)
        return True, reason, handoff
        
    # 3. Repeated Frustration Check
    if check_repeated_frustration(session_personas, current_persona):
        reason = "Repeated user frustration detected over consecutive interactions."
        logger.info(f"Escalating: {reason}")
        handoff = generate_handoff_json(query, current_persona, retrieved_sources, top_similarity_score)
        return True, reason, handoff
        
    return False, "", None
