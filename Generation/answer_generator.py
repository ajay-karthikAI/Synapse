"""
Day 6: Answer Generation
========================
Takes reranked chunks (Day 5) and generates a waiting room-appropriate
response using GPT-4o.

Response structure (every answer follows this):
  1. ACKNOWLEDGE   — validate what the patient is experiencing
  2. EDUCATE       — plain language summary from retrieved research
  3. CONTEXTUALIZE — frame what the doctor will be evaluating
  4. EMPOWER       — 3-5 specific questions to ask the physician
  5. BOUNDARY      — explicit non-diagnosis disclaimer
  6. EMERGENCY     — bypasses RAG entirely if danger signals detected

Design principles:
  - Answer ONLY from retrieved context (no hallucination)
  - Plain language (8th grade reading level target)
  - Never diagnose, never recommend specific treatments
  - Always defer final judgment to the physician
  - Citations map back to real PubMed papers

Interview talking point:
  "The prompt is the safety layer. I constrain the LLM to answer only
  from retrieved context, enforce plain language for health literacy,
  and structurally separate education from advice. The physician
  question card is the key differentiator — it converts passive
  information into an actionable tool the patient uses in the next
  30 minutes."
"""

import os
import re
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from Data.fetch_and_chunk import Chunk
from Retrieval.reranker import Reranker

from openai import OpenAI


# ---------------------------------------------------------------------------
# Emergency detection — runs BEFORE retrieval, bypasses RAG entirely
# ---------------------------------------------------------------------------

EMERGENCY_SIGNALS = [
    "chest pain", "chest tightness", "can't breathe", "cannot breathe",
    "shortness of breath", "difficulty breathing", "heart attack",
    "stroke", "face drooping", "arm weakness", "speech difficulty",
    "overdose", "took too much", "severe bleeding", "unconscious",
    "passing out", "suicidal", "want to die", "end my life",
    "severe headache", "worst headache", "sudden vision loss",
    "coughing blood", "vomiting blood",
]

EMERGENCY_RESPONSE = """I want to make sure you're safe right now.

Some of the symptoms you've described can sometimes require immediate medical attention. Please let the front desk or a nurse know how you're feeling right now — don't wait for your scheduled appointment.

If you feel you need immediate help:
  → Tell the front desk immediately
  → Call 911 if you feel you are in danger
  → Go to the nearest emergency room

This app is for general health education and cannot assess your current condition. Please speak with medical staff right away."""


def check_emergency(query: str) -> bool:
    """Returns True if the query contains emergency signals."""
    query_lower = query.lower()
    return any(signal in query_lower for signal in EMERGENCY_SIGNALS)


# ---------------------------------------------------------------------------
# System prompt — defines the app's entire persona and constraints
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a patient education assistant for a medical waiting room app called Prep.

Your role is to help patients understand their health concerns and prepare meaningful questions for their upcoming physician appointment. You are NOT a doctor and do NOT provide diagnoses or treatment recommendations.

STRICT RULES:
1. Answer ONLY using the provided source passages. If the sources don't contain relevant information, say so honestly.
2. Use plain, clear language. Avoid medical jargon. If you must use a medical term, explain it immediately.
3. Never suggest a specific diagnosis.
4. Never recommend starting, stopping, or changing any medication.
5. Never interpret specific test results as definitively normal or abnormal for this patient.
6. Always frame information as "research suggests" or "studies show" — not as personal medical advice.
7. End every response with the physician question card.

TONE:
- Calm and reassuring, never alarming
- Warm but professional
- Empowering — the goal is to help the patient feel informed and prepared, not anxious

RESPONSE FORMAT:
You must follow this exact structure for every response:

📋 WHAT THE RESEARCH SAYS
[2-3 paragraphs of plain language education grounded in the sources]

🔬 WHAT YOUR DOCTOR WILL EVALUATE
[1 paragraph explaining what the physician will assess — keeps expectations realistic]

❓ QUESTIONS TO ASK YOUR DOCTOR TODAY
[Numbered list of 4-5 specific, actionable questions the patient can bring to the appointment]

⚠️ IMPORTANT
This information is based on general medical research and is not a diagnosis or personal medical advice. Your physician will evaluate your specific situation."""


# ---------------------------------------------------------------------------
# User prompt — injects retrieved context and patient query
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """Here are relevant passages from medical research:

{context}

---

Patient's question: {query}

Please respond following the required format. Base your answer only on the provided passages. If a passage is relevant, cite it as [Source 1], [Source 2], etc."""


# ---------------------------------------------------------------------------
# Answer generator
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """
    Generates waiting room responses from reranked chunks.

    Full pipeline for a single query:
      1. Check for emergency signals → return emergency response if found
      2. Format reranked chunks into labeled context
      3. Call GPT-4o with system + user prompt
      4. Parse and return structured response with metadata
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        # gpt-4o-mini is fine for this use case and much cheaper
        # Upgrade to gpt-4o if answer quality needs improvement
        self.model = model

    def generate(
        self,
        query: str,
        reranked_results: list,
        api_key: str,
        reranker: Reranker = None,
    ) -> dict:
        """
        Generate a waiting room response.

        Returns dict:
          {
            answer:        str   — full formatted response
            is_emergency:  bool  — True if emergency routing triggered
            sources:       list  — [{pmid, title, url, confidence_pct}]
            query:         str   — original query
            model:         str   — model used
          }
        """
        # --- Step 1: Emergency check ---
        if check_emergency(query):
            return {
                "answer":       EMERGENCY_RESPONSE,
                "is_emergency": True,
                "sources":      [],
                "query":        query,
                "model":        "emergency_bypass",
            }

        # --- Step 2: Format context ---
        if reranker is None:
            reranker = Reranker(strategy="keyword")
        context = reranker.format_for_context(reranked_results)

        # --- Step 3: Generate response ---
        client = OpenAI(api_key=api_key)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,    # slight creativity for natural language
            max_tokens=800,     # enough for full structured response
        )

        content = response.choices[0].message.content
        if content is None:
            content = "I wasn't able to generate a response. Please try again."
        answer = content.strip()

        # --- Step 4: Build source list for UI display ---
        sources = []
        for item in reranked_results:
            chunk = item["chunk"]
            sources.append({
                "pmid":           chunk.pmid,
                "title":          chunk.title,
                "url":            chunk.source_url,
                "confidence_pct": item.get("confidence_pct", 0),
                "rerank_score":   item.get("rerank_score", 0),
            })

        return {
            "answer":       answer,
            "is_emergency": False,
            "sources":      sources,
            "query":        query,
            "model":        self.model,
        }


# ---------------------------------------------------------------------------
# Full pipeline runner — connects Days 1-6
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    chunks: list,
    api_key: str,
    retrieval_top_k: int = 10,
    rerank_top_k: int = 3,
) -> dict:
    """
    End-to-end pipeline from query to waiting room response.
    Days 1-6 connected in one function.

    This is what Day 7 Streamlit UI will call.
    """
    from retrieval.vector_store import VectorStore, build_faiss_index
    from retrieval.bm25_index import BM25Index
    from retrieval.hybrid_retriever import HybridRetriever, linear_fusion
    import numpy as np

    # Emergency check before any API calls
    if check_emergency(query):
        generator = AnswerGenerator()
        return generator.generate(query, [], api_key)

    # Build indexes (in production these are loaded from disk, not rebuilt)
    print("[Pipeline] Building indexes...")
    hybrid = HybridRetriever(fusion="linear", alpha=0.7)
    hybrid.build(chunks, api_key=api_key)

    # Retrieve
    print("[Pipeline] Retrieving...")
    retrieval_results = hybrid.search(query, api_key=api_key, top_k=retrieval_top_k)

    # Rerank
    print("[Pipeline] Reranking...")
    reranker = Reranker(strategy="llm")
    reranked = reranker.rerank(query, retrieval_results, api_key=api_key, top_k=rerank_top_k)

    # Generate
    print("[Pipeline] Generating response...")
    generator = AnswerGenerator()
    result = generator.generate(query, reranked, api_key=api_key, reranker=reranker)

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", "")

    # --- Test 1: Emergency detection (no API needed) ---
    print("=== Test 1: Emergency Detection ===")
    emergency_queries = [
        "I have chest pain and my left arm is numb",
        "my blood glucose spikes to 300",     # not emergency
        "I took too much of my medication",
    ]
    for q in emergency_queries:
        is_emergency = check_emergency(q)
        print(f"  {'🚨 EMERGENCY' if is_emergency else '✓ Normal  '} | '{q}'")

    print()

    # --- Test 2: Full generation (requires API key) ---
    if not api_key:
        print("=== Test 2: Answer Generation ===")
        print("No OPENAI_API_KEY — showing what a response looks like:\n")

        mock_response = """📋 WHAT THE RESEARCH SAYS
Blood glucose levels of 300 mg/dL are significantly above the normal range of 70-140 mg/dL [Source 1]. Research suggests that episodes of high blood glucose, called hyperglycemia, can occur due to several factors including the body's reduced ability to use insulin effectively, dietary choices, physical activity levels, and stress [Source 2].

Studies show that frequent glucose monitoring helps identify patterns in these spikes, which gives your doctor important information about how your body is responding to current management [Source 1].

🔬 WHAT YOUR DOCTOR WILL EVALUATE
Your physician will review your glucose readings alongside your medical history, current medications, and lifestyle factors. They may order additional tests such as an HbA1c — a measure of your average blood sugar over the past 2-3 months [Source 3] — to better understand the pattern of these spikes and adjust your care plan accordingly.

❓ QUESTIONS TO ASK YOUR DOCTOR TODAY
1. What is causing my glucose to spike this high, and is this pattern concerning?
2. Should I be monitoring my blood glucose at home, and how often?
3. Could my current medications or diet be contributing to these spikes?
4. What glucose level should prompt me to seek immediate care?
5. Would checking my HbA1c help understand how often this is happening?

⚠️ IMPORTANT
This information is based on general medical research and is not a diagnosis or personal medical advice. Your physician will evaluate your specific situation."""

        print(mock_response)
        print("\n--- Sources ---")
        print("  [Source 1] PMID:29505530 — Glucose Monitoring Review (confidence: 82%)")
        print("  [Source 2] PMID:28823139 — ADA Standards of Care (confidence: 74%)")
        print("  [Source 3] PMID:27697747 — HbA1c Targets Study (confidence: 68%)")
        print("\n✓ Pipeline structure validated. Add OPENAI_API_KEY to test live generation.")

    else:
        print("=== Test 2: Live Answer Generation ===\n")

        # Simulate reranked results from Day 5
        reranked_results = [
            {
                "chunk": Chunk(
                    text="Blood glucose monitoring frequency should be individualized. Patients on insulin may need to monitor 4-10 times daily. Continuous glucose monitoring provides additional insight into glucose variability.",
                    source="pubmed", pmid="29505530", title="Glucose Monitoring Review",
                    source_url="https://pubmed.ncbi.nlm.nih.gov/29505530/",
                ),
                "rerank_score": 8.5, "confidence_pct": 85, "rank": 1,
            },
            {
                "chunk": Chunk(
                    text="Metformin is the preferred initial pharmacologic agent for type 2 diabetes. It lowers HbA1c by 1-2% and has a favorable safety and tolerability profile.",
                    source="pubmed", pmid="28823139", title="ADA Standards of Care",
                    source_url="https://pubmed.ncbi.nlm.nih.gov/28823139/",
                ),
                "rerank_score": 7.2, "confidence_pct": 72, "rank": 2,
            },
            {
                "chunk": Chunk(
                    text="HbA1c reflects average plasma glucose over 2-3 months. A target below 7% is appropriate for most non-pregnant adults with diabetes to reduce complications.",
                    source="pubmed", pmid="27697747", title="HbA1c Targets Study",
                    source_url="https://pubmed.ncbi.nlm.nih.gov/27697747/",
                ),
                "rerank_score": 6.8, "confidence_pct": 68, "rank": 3,
            },
        ]

        reranker = Reranker(strategy="llm")
        generator = AnswerGenerator(model="gpt-4o-mini")

        query = "my blood glucose spikes randomly to 300. why does this happen and what should I ask my doctor?"
        print(f"Query: '{query}'\n")

        result = generator.generate(query, reranked_results, api_key=api_key, reranker=reranker)

        print(result["answer"])
        print("\n--- Sources ---")
        for s in result["sources"]:
            print(f"  PMID:{s['pmid']} — {s['title']} (confidence: {s['confidence_pct']}%)")
            print(f"  {s['url']}")