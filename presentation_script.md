# 🎙️ Project Presentation Script & Code Walkthrough Guide

This document is your complete companion guide for recording your 3–5 minute project video. It contains a professional speech script, visual cue guidance, and a line-by-line code walkthrough tailored to your workspace structure.

You can open this file in **VS Code** side-by-side with your code or browser during your recording!

---

## 🎬 Video Overview & Timeline
*   **0:00 - 0:45**: Introduction & Key Features Overview
*   **0:45 - 2:00**: Live Application Walkthrough & Scenario Demos
*   **2:00 - 4:15**: Code Walkthrough in VS Code (`app.py` & Project Files)
*   **4:15 - 4:45**: Deployment (Hugging Face Spaces & GitHub) & Wrap-Up

---

## 🎤 PART 1: Introduction & Features (0:00 - 0:45)
*(Visual: Show the running Gradio App in your browser at `http://127.0.0.1:7860` or your Hugging Face Space URL. Have the browser screen clean and ready.)*

**What to Say:**
> "Hello everyone! Today, I am excited to present my project: the **Persona-Adaptive Customer Support Agent**.
> 
> The core objective of this project is to build an intelligent, enterprise-grade customer support assistant. The system leverages state-of-the-art Generative AI and Vector Retrieval to achieve three main features:
> 1. **Dynamic Persona Adaptation**: It automatically detects if a customer is a *Technical Expert*, *Frustrated User*, or *Business Executive*, and adjusts its communication tone accordingly.
> 2. **Context-Grounded Retrieval (RAG)**: It indexes a local knowledge base of support documents using a vector database, ensuring answers are 100% factual and grounded.
> 3. **Automated Escalation Policy**: If a query is outside the database scope, contains sensitive billing/refund requests, or exhibits repeated customer frustration, the system flags the session and generates a structured handoff report for a human agent.
> 
> Let's start with a quick live demonstration of the system."

---

## 🎤 PART 2: Live App Demonstration (0:45 - 2:00)
*(Visual: Type queries into the Gradio UI text box and click "Send Message" or press Enter. Point to the diagnostic values on the right-hand panel as they update.)*

**What to Say:**
> "Here is our Gradio dashboard. On the left is our support chat interface, and on the right is our real-time **Agent Diagnostics Dashboard**. Let's run a few scenarios:
> 
> **Scenario 1: Technical Query**
> Let's type: *'What are the header parameter requirements for Bearer token auth?'*
> 
> *(Pause briefly for response)*
> 
> As you can see, the agent classifies the user as a **Technical Expert** with a high confidence score. It retrieved the `bearer_tokens.md` guide from our knowledge base with a similarity score above 0.60. Notice how the AI response is structured: it uses technical jargon, explains the exact protocol, and includes formatted code blocks.
> 
> **Scenario 2: Empathy for a Frustrated User**
> Now, let's test a frustrated customer: *'Where is the guide to clear cookies? It's been an hour and nothing is loading on your interface!'*
> 
> *(Pause briefly for response)*
> 
> The classifier correctly flags the persona as **Frustrated User** due to the urgency and emotional cues. Notice how the AI adapts: it bypasses technical jargon and begins with sincere, caring validation: *'I completely understand how frustrating it is when the system doesn't load...'*, followed by clear, bulleted steps.
> 
> **Scenario 3: Automatic Human Handoff**
> Finally, let's trigger an escalation with: *'I demand a refund for the duplicate charges on my billing statement.'*
> 
> *(Pause briefly for response)*
> 
> Because this contains the sensitive keywords 'billing' and 'refund', the system immediately sets the Escalation Status to **Handoff Required**. It displays a professional fallback response to the customer, and generates this structured **Handoff JSON** on the right, providing a summary, source files, and recommended next steps for a human support agent."

---

## 🎤 PART 3: Code Walkthrough in VS Code (2:00 - 4:15)
*(Visual: Switch to VS Code. Show the file explorer sidebar on the left containing `app.py`, `generate_kb.py`, `test_scenarios.py`, and `requirements.txt`. Then open the files as you talk about them.)*

**What to Say:**
> "Now let's dive into the implementation. 
> 
> To ensure high robustness during deployment on containerized platforms like Hugging Face Spaces—and to prevent module resolution errors—the backend services and frontend Gradio UI are consolidated into a clean, self-contained `app.py` file. Let's look at the key sections:
> 
> **1. SQLite Portability Workaround (Lines 14-24)**
> Hugging Face containers run older versions of SQLite by default, which can cause ChromaDB to fail. At the very top of `app.py`, we implement a dynamic override that hot-swaps the standard library `sqlite3` module with `pysqlite3-binary` if the version is older than `3.35.0`.
> 
> **2. The Vector Database & Ingestion (Lines 234-311)**
> In the `LocalRAGPipeline` class, we initialize a persistent `chromadb` client using the Cosine distance space. 
> To split document formats like `.txt`, `.md`, and `.pdf`, we use LangChain's `RecursiveCharacterTextSplitter` configured with a chunk size of 400 characters and an overlap of 40 characters.
> 
> Crucially, we implement **Incremental Indexing**. We extract the file modification time (`mtime`) and check it against the indexed metadata. If a file is unmodified, we skip it to avoid wasting API tokens and indexing duplicate entries. If it has changed, we delete the old chunks and re-index.
> 
> **3. Fallback / Offline Mode (Lines 138-164 & Lines 327-345)**
> If no Gemini API key is present in the environment, the application does not crash. It automatically falls back to an offline mode: using rule-based keyword matching for persona classification, and a custom word-overlap similarity scorer that mirrors our RAG database behavior.
> 
> **4. Persona Classification & Structured Outputs (Lines 120-212)**
> The `classify_customer_persona` function utilizes the new `google-genai` SDK and the `gemini-2.5-flash` model. We pass a Pydantic schema `PersonaClassification` into the `response_schema` configuration, ensuring Gemini outputs structured JSON reliably.
> 
> **5. Escalation Policy (Lines 376-471)**
> The `evaluate_escalation` function checks three criteria:
> - If the RAG retrieval similarity score falls below `0.45`.
> - If the user query contains sensitive terms like `billing`, `refund`, or `legal`.
> - Or if the user has shown repeated frustration (two consecutive interactions flagged as 'Frustrated User').
> If any condition is met, it runs `generate_handoff_json` via Gemini to construct our structured triage handoff schema.
> 
> **6. Response Generation & Safeguards (Lines 476-583)**
> In `generate_adapted_response`, we dynamically set system instructions matching the user's detected persona. To enforce strict grounding, we instruct the model to answer *only* using the retrieved facts. If the facts are missing, the prompt instructs the model to output a special `[INSUFFICIENT_CONTEXT]` token, which we catch programmatically to trigger a human handoff."

---

## 🎤 PART 4: Conclusion & Project Wrap-Up (4:15 - 4:45)
*(Visual: Show the GitHub repository page or Hugging Face Space settings, then show the test results file or readme in VS Code.)*

**What to Say:**
> "To wrap up, the codebase includes:
> - `generate_kb.py`, which populates our knowledge base with 15 markdown/text guides and compiles a PDF layout using ReportLab.
> - `test_scenarios.py`, containing automated tests for all assignment specifications.
> - A comprehensive `README.md` containing Hugging Face spaces configuration frontmatter.
> 
> The project has been successfully built, fully committed in Git, published to GitHub, and is running live on Hugging Face Spaces!
> 
> Thank you for your time, and I look forward to your feedback!"

---

## 💡 Quick Tips for Recording Your Video:
1. **Resolution**: Record in 1080p, and zoom in slightly on VS Code (`Ctrl + +` / `Cmd + +`) so your text is clear and readable.
2. **Audio**: Use a quiet room and a decent microphone. Speak at a steady, natural pace.
3. **Practice**: Do a quick 1-minute dry-run of typing the test queries and navigating VS Code before hitting record.
4. **Tools**: You can use free tools like **Loom**, **OBS Studio**, or the built-in Windows Game Bar (`Win + G`) to record your screen and microphone.
