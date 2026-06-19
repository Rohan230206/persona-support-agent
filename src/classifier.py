import json
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src import config
from src.utils import logger, retry_with_backoff

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
    """
    Analyzes the user's message and classifies it into one of the three target personas.
    Uses Gemini structured JSON output. If GEMINI_API_KEY is missing, falls back
    to keyword/sentiment rules to support offline demonstration and testing.
    """
    logger.info(f"Classifying persona for message: {user_message[:60]}...")
    
    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set. Using rule-based offline classification fallback.")
        query_lower = user_message.lower()
        
        # Rule 1: Frustrated User cues
        if any(w in query_lower for w in ["clear cookies", "hour", "loading", "frozen", "fix this", "demand", "refund", "duplicate", "immediate"]):
            return {
                "persona": "Frustrated User",
                "confidence": 0.85,
                "reasoning": "[Offline Rule] Message contains high-urgency or negative sentiment key phrases."
            }
        # Rule 2: Technical Expert cues
        elif any(w in query_lower for w in ["bearer token", "header", "database", "api", "config", "pool", "integration", "port"]):
            return {
                "persona": "Technical Expert",
                "confidence": 0.90,
                "reasoning": "[Offline Rule] Message contains technical protocols, configurations, or system terms."
            }
        # Rule 3: Business Executive cues
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
        
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
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
            model=config.LLM_MODEL,
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
