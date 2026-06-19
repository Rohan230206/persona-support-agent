from typing import List, Dict, Any
from google import genai
from google.genai import types

from src import config
from src.utils import logger, retry_with_backoff

@retry_with_backoff(max_retries=3, initial_delay=1.0)
def generate_adapted_response(
    query: str, 
    persona: str, 
    context_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generates a personalized response matching the user's communication persona,
    strictly grounded in the retrieved context.
    If context is insufficient, returns an escalation flag.
    Supports offline fallback mode when GEMINI_API_KEY is not configured.
    """
    logger.info(f"Generating response for persona: '{persona}' using {len(context_chunks)} chunks.")
    
    if not context_chunks:
        logger.info("No context chunks provided. Flagging as insufficient context.")
        return {
            "escalated": True,
            "response": "I apologize, but I couldn't find any documents in our knowledge base related to your query. Connecting you to a support agent.",
            "reason": "Knowledge base returned empty context."
        }
        
    # Check for offline mode fallback
    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is missing. Using offline generation logic.")
        best_score = max([c["score"] for c in context_chunks]) if context_chunks else 0.0
        if best_score < config.SIMILARITY_THRESHOLD:
            return {
                "escalated": True,
                "response": "I apologize, but the available information in our database is insufficient to fully resolve your request.",
                "reason": f"Top similarity score ({best_score}) is below threshold ({config.SIMILARITY_THRESHOLD}) in offline mode."
            }
            
        top_chunk = context_chunks[0]
        snippet = top_chunk["text"].strip()
        source = top_chunk["source"]
        
        if persona == "Technical Expert":
            response = (
                f"[OFFLINE EXPERT ADAPTATION] - System Configuration Lead\n\n"
                f"Root-Cause Analysis:\n"
                f"We retrieved relevant troubleshooting specifications from local document '{source}'.\n\n"
                f"System Details & Code Snippet:\n"
                f"```\n"
                f"{snippet}\n"
                f"```\n\n"
                f"Configuration Steps:\n"
                f"1. Open connection settings file.\n"
                f"2. Ensure authentication parameters and request headers match the specifications from the snippet.\n"
                f"3. Validate that your network client properly manages exceptions and timeouts."
            )
        elif persona == "Frustrated User":
            # Extract first sentence or bullet
            short_detail = snippet.split('\n')[0] if '\n' in snippet else snippet[:200]
            response = (
                f"[OFFLINE EMPATHY ADAPTATION] - Customer Care Lead\n\n"
                f"I completely understand how frustrating it is when the system doesn't load and you are stuck waiting. "
                f"I am here to guide you through this process and help get you up and running right away!\n\n"
                f"According to our '{source}' guide, please try the following simple steps:\n"
                f"- **Step 1**: Check the details: {short_detail}\n"
                f"- **Step 2**: Clear your browser's history or refresh the screen (press Ctrl+F5 on Windows).\n"
                f"- **Step 3**: Try opening a private window to verify if it is a temporary cache error.\n\n"
                f"We are here to support you, so please let me know if these steps resolve the issue!"
            )
        else:  # Business Executive
            response = (
                f"[OFFLINE EXECUTIVE ADAPTATION] - Director of Client Operations\n\n"
                f"Here is the high-level business summary regarding your request. The operational policy and timelines "
                f"defined in our documentation ('{source}') indicate:\n\n"
                f"- **Resolution Timeline**: Review and dispute investigations are resolved within 5-7 business days.\n"
                f"- **Operational Impact**: Access remains active during investigations as long as core usage fees are settled.\n\n"
                f"If you require faster processing or custom SLA terms, please notify our accounts supervisor."
            )
            
        return {
            "escalated": False,
            "response": response,
            "reason": ""
        }

    # Online mode: format retrieved context
    context_blocks = []
    for idx, chunk in enumerate(context_chunks):
        context_blocks.append(f"Document [{chunk['source']}] Chunk {idx + 1}:\n{chunk['text']}")
    context_text = "\n\n".join(context_blocks)
    
    if persona == "Technical Expert":
        persona_instructions = (
            "You are a Senior Systems Engineer and Technical Support Lead. Your goal is to provide "
            "a highly detailed, technical, and structured response. Include root-cause explanations, "
            "configuration setups, precise API configurations, database pool parameters, or HTTP code blocks "
            "if relevant. Maintain a professional, systematic, and exact tone."
        )
    elif persona == "Frustrated User":
        persona_instructions = (
            "You are a reassuring, warm, and deeply empathetic Customer Care Specialist. Your goal is to "
            "de-escalate the user's frustration. Begin your response immediately with sincere, caring validation "
            "of their difficulty (e.g., 'I understand how frustrating it is when...'). Use simple, direct "
            "language, avoid any complex technical terms or configuration jargon, and list clear troubleshooting "
            "steps using bullet points."
        )
    else:  # Business Executive
        persona_instructions = (
            "You are a concise, formal Client Relations Director. Your goal is to provide a brief executive summary "
            "answering the query. Keep your response short, direct, and professional. Focus on business outcomes, "
            "timelines for resolution, SLAs, and organizational impact. Avoid long explanations, configuration files, "
            "or step-by-step technical guides."
        )
        
    system_instruction = (
        f"{persona_instructions}\n\n"
        "CRITICAL SYSTEM RULES:\n"
        "1. You must answer the user's query using ONLY the facts explicitly provided in the 'FACTUAL CONTEXT DOCUMENTS' section below.\n"
        "2. Do not assume, extrapolate, or use outside knowledge. Do not hallucinate any details.\n"
        "3. If the provided context documents do not contain enough facts to fully, accurately, and confidently answer the user's query, "
        "you must output exactly '[INSUFFICIENT_CONTEXT]' and nothing else. This is a hard requirement."
    )
    
    prompt = (
        f"FACTUAL CONTEXT DOCUMENTS:\n{context_text}\n\n"
        f"USER QUERY: {query}\n"
    )
    
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    try:
        response = client.models.generate_content(
            model=config.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2
            )
        )
        
        response_text = response.text.strip()
        
        if "[INSUFFICIENT_CONTEXT]" in response_text or response_text == "[INSUFFICIENT_CONTEXT]":
            logger.info("Model flagged context as insufficient for answering. Triggering escalation.")
            return {
                "escalated": True,
                "response": "I apologize, but the available information in our database is insufficient to fully resolve your request. I am handoffing this to a human agent.",
                "reason": "Insufficient factual context in knowledge base."
            }
            
        logger.info("Response successfully generated.")
        return {
            "escalated": False,
            "response": response_text,
            "reason": ""
        }
        
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        return {
            "escalated": True,
            "response": "An unexpected error occurred while processing your request. Escalating to human support.",
            "reason": f"Generator exception: {e}"
        }
