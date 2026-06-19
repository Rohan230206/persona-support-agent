import os
import sys
from pathlib import Path
import gradio as gr

# Ensure python path is aligned
sys.path.append(str(Path(__file__).resolve().parent))

from src import config
from src.utils import logger
from src.classifier import classify_customer_persona
from src.rag_pipeline import LocalRAGPipeline
from src.escalator import evaluate_escalation, generate_handoff_json
from src.generator import generate_adapted_response
from generate_kb import generate_all

# Startup Check: Ensure API key is configured
if not config.validate_environment():
    logger.warning("GEMINI_API_KEY is not configured. Please add it to your environment or .env file.")

# Startup Check: Check for sample documents in data/
try:
    existing_files = list(config.DATA_DIR.glob("*"))
    # We expect 16 files (15 text/md + 1 PDF)
    if len(existing_files) < 15:
        logger.info("Fewer than 15 documents found. Running knowledge base generation script...")
        generate_all()
except Exception as e:
    logger.error(f"Error checking data directory: {e}")
    generate_all()

# Startup Ingestion: Initialize RAG Pipeline and Ingest
try:
    pipeline = LocalRAGPipeline()
    pipeline.ingest_data_directory()
except Exception as e:
    logger.error(f"Failed to initialize and run initial RAG pipeline ingestion: {e}")
    pipeline = None

# Custom CSS for Premium Design
custom_css = """
body {
    font-family: 'Inter', sans-serif !important;
}
.diagnostic-container {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}
.badge-tech {
    background-color: #dbeafe !important;
    color: #1e40af !important;
    font-weight: 700 !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    border: 1px solid #bfdbfe !important;
}
.badge-frustrated {
    background-color: #fee2e2 !important;
    color: #991b1b !important;
    font-weight: 700 !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    border: 1px solid #fca5a5 !important;
}
.badge-exec {
    background-color: #fef3c7 !important;
    color: #92400e !important;
    font-weight: 700 !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    border: 1px solid #fde68a !important;
}
.normal-status {
    background-color: #ecfdf5 !important;
    color: #047857 !important;
    font-weight: 700 !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    border: 1px solid #a7f3d0 !important;
}
.escalated-status {
    background-color: #fff5f5 !important;
    color: #c53030 !important;
    font-weight: 700 !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    border: 1px solid #feb2b2 !important;
}
"""

def process_query(message: str, history: list, session_personas: list):
    """
    Main orchestrator for processing client query.
    """
    if not message.strip():
        return "", history, session_personas, "N/A", 0.0, 0.0, "✅ Normal Operations", {}, "*No document chunks retrieved.*"

    if pipeline is None:
        logger.error("RAG pipeline is uninitialized.")
        # Fallback response
        err_msg = "System is running in offline mode. Please configure GEMINI_API_KEY."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": err_msg})
        return "", history, session_personas, "N/A", 0.0, 0.0, "🚨 Handoff Required", {"error": err_msg}, "*No document chunks retrieved.*"

    try:
        # 1. Persona Classification
        classification = classify_customer_persona(message)
        persona = classification.get("persona", "Frustrated User")
        conf = classification.get("confidence", 0.5)
        
        # 2. Context Retrieval (top 3 chunks)
        retrieved_chunks = pipeline.retrieve_context(message, top_k=3)
        top_score = max([chunk["score"] for chunk in retrieved_chunks]) if retrieved_chunks else 0.0
        retrieved_sources = list(set([chunk["source"] for chunk in retrieved_chunks]))
        
        # Format sources markup for UI
        sources_md = "### Top Retrieved Chunks\n\n"
        if retrieved_chunks:
            for idx, chunk in enumerate(retrieved_chunks):
                sources_md += f"**{idx+1}. {chunk['source']}** (Similarity Score: `{chunk['score']:.4f}`)\n"
                sources_md += f"> {chunk['text'].strip()}\n\n"
        else:
            sources_md += "*No document chunks retrieved.*"
            
        # 3. Escalation Checking
        should_escalate, esc_reason, handoff_json = evaluate_escalation(
            query=message,
            top_similarity_score=top_score,
            current_persona=persona,
            session_personas=session_personas,
            retrieved_sources=retrieved_sources
        )
        
        final_response = ""
        escalation_status = "✅ Normal Operations"
        
        if should_escalate:
            escalation_status = "🚨 Handoff Required"
            final_response = "I apologize, but I am unable to locate the precise solution to your request. I am connecting you with a live human support specialist."
            
        else:
            # 4. Generate Persona-Adapted Response
            gen_result = generate_adapted_response(message, persona, retrieved_chunks)
            
            # Check if generator raised insufficient context exception
            if gen_result.get("escalated", False):
                should_escalate = True
                escalation_status = "🚨 Handoff Required"
                final_response = "I apologize, but I am unable to locate the precise solution to your request. I am connecting you with a live human support specialist."
                esc_reason = gen_result.get("reason", "Context insufficient")
                handoff_json = generate_handoff_json(message, persona, retrieved_sources, top_score)
            else:
                final_response = gen_result.get("response", "")

        # Store persona in history for repeated frustration checks
        session_personas.append(persona)
        
        # Update chat interface history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": final_response})
        
        handoff_display = handoff_json if handoff_json else {}
        
        return (
            "",  # Reset input textbox
            history,
            session_personas,
            persona,
            conf,
            top_score,
            escalation_status,
            handoff_display,
            sources_md
        )
        
    except Exception as e:
        logger.error(f"Error executing user query processing: {e}")
        err_msg = f"System Error: {e}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "An unexpected system error occurred. We are routing you to support."})
        return "", history, session_personas, "N/A", 0.0, 0.0, "🚨 Handoff Required", {"error": err_msg}, "*No document chunks retrieved.*"

def reset_session():
    """Clears chatbot and sidebar information."""
    return [], [], "N/A", 0.0, 0.0, "✅ Normal Operations", {}, "*No document chunks retrieved.*"

# Build Gradio Block Interface
with gr.Blocks(title="Persona-Adaptive Support Agent") as demo:
    # State components
    session_personas_state = gr.State(value=[])
    
    # Title Header
    gr.Markdown(
        """
        # 🤖 Persona-Adaptive Customer Support Agent
        ### Enterprise AI Support Assistant with Dynamic Persona Adaptation, Vector-guided RAG, and Human Handoff.
        """
    )
    
    with gr.Row():
        # Left Column: Conversation panel
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Support Session Log", 
                height=500
            )
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
            
        # Right Column: Diagnostics dashboard
        with gr.Column(scale=2, elem_classes="diagnostic-container"):
            gr.Markdown("## 📊 Agent Diagnostics Dashboard")
            
            with gr.Row():
                persona_val = gr.Textbox(
                    label="Detected User Persona",
                    value="N/A",
                    interactive=False
                )
                persona_conf = gr.Number(
                    label="Classification Confidence",
                    value=0.00,
                    precision=2,
                    interactive=False
                )
                
            with gr.Row():
                retrieval_conf = gr.Number(
                    label="Retrieval Cosine Score",
                    value=0.0000,
                    precision=4,
                    interactive=False
                )
                esc_status = gr.Textbox(
                    label="System Escalation Status",
                    value="✅ Normal Operations",
                    interactive=False
                )
                
            with gr.Tab("Retrieved RAG Sources"):
                sources_display = gr.Markdown(
                    value="*No document chunks retrieved.*",
                    line_breaks=True
                )
                
            with gr.Tab("Escalation Handoff JSON"):
                handoff_display = gr.JSON(
                    label="Structured JSON Handoff",
                    value={}
                )
                
    # Bind submit handlers
    submit_btn.click(
        fn=process_query,
        inputs=[query_input, chatbot, session_personas_state],
        outputs=[
            query_input, 
            chatbot, 
            session_personas_state, 
            persona_val, 
            persona_conf, 
            retrieval_conf, 
            esc_status, 
            handoff_display, 
            sources_display
        ]
    )
    
    query_input.submit(
        fn=process_query,
        inputs=[query_input, chatbot, session_personas_state],
        outputs=[
            query_input, 
            chatbot, 
            session_personas_state, 
            persona_val, 
            persona_conf, 
            retrieval_conf, 
            esc_status, 
            handoff_display, 
            sources_display
        ]
    )
    
    clear_btn.click(
        fn=reset_session,
        inputs=None,
        outputs=[
            chatbot, 
            session_personas_state, 
            persona_val, 
            persona_conf, 
            retrieval_conf, 
            esc_status, 
            handoff_display, 
            sources_display
        ]
    )

if __name__ == "__main__":
    # Bind to 0.0.0.0 for container deployments and resolve port dynamically
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, css=custom_css)
