"""
Day 5: Reranking
================
Takes the top-k results from hybrid retrieval (Day 4) and re-scores
them using an LLM to improve precision before answer generation.
 
Why reranking?
  Retrieval optimizes for recall — get everything potentially relevant.
  Reranking optimizes for precision — surface the most useful chunks.
  Without reranking, the LLM sees noisy context and either hallucinates
  to fill gaps or produces a diluted answer averaging across irrelevant chunks.
 
Two strategies implemented:
  1. LLM reranking   — GPT scores each chunk's relevance (0-10)
  2. Cross-encoder   — dedicated model scores query-chunk pairs
                       (included as upgrade path, not default)
 
For a waiting room patient tool, LLM reranking is the right default:
  - No extra model download
  - Scores are interpretable (maps to confidence %)
  - Can be prompted to weight patient-friendliness alongside relevance
 
Interview talking point:
  "Reranking separates retrieval from precision. I retrieve broadly
  with hybrid search to maximize recall, then rerank to maximize
  precision before the LLM sees the context. This reduces hallucination
  because the model isn't trying to reconcile irrelevant chunks."
"""
 
import os
import json #Parse LLM responses and converts Python <->> JSON
from pathlib import Path
 
import sys #Mod Python's import path at run time
sys.path.append(str(Path(__file__).parent.parent)) #__file__ is current file path .parent is folder that its in .parent.parent is go up one more directory sys.path.append(..) tells python to also look here when importing modules
from Data.fetch_and_chunk import Chunk #
 
from openai import OpenAI
 
# ---------------------------------------------------------------------------
# LLM Reranker
# ---------------------------------------------------------------------------
#Defining multi-string template for the LLM 
RERANK_PROMPT = """\ 
You are evaluating how useful a medical text passage is for answering a patient's question.
The patient is in a waiting room and wants to understand their condition and prepare questions for their doctor.
 
Score the passage from 0 to 10 based on:
  - Relevance: Does it directly address the patient's question? (most important)
  - Clarity: Is the information specific enough to be useful?
  - Safety: Does it stay educational without making specific diagnoses?
 
Question: {query}
 
Passage: {passage}
 
Respond with ONLY a JSON object in this exact format:
{{"score": <number 0-10>, "reason": "<one sentence>"}}
"""
 
 
def rerank_with_llm(
    query: str, #User question
    results: list, #Retreived chunks
    api_key: str, #OpenAI Key
    model: str = "gpt-4o-mini", #default model; can be easily upgraded without changing logic
    top_k: int = 3, #How many results to keep
) -> list: #Explains purpose and design decisions
    """
    Re-score retrieval results using GPT.
 
    Why gpt-4o-mini?
      Fast, cheap (~$0.0001 per rerank call), good enough for relevance
      scoring. Reserve gpt-4o for the final answer generation in Day 6.
 
    Args:
        results: list of {chunk, score, rank} from HybridRetriever.search()
        top_k:   how many to keep after reranking
 
    Returns:
        Reranked list of {chunk, original_score, rerank_score,
                          rerank_reason, rank, confidence_pct}
    """
    client = OpenAI(api_key=api_key) #Create OpenAI client instance; Pass API key explicitly (good practice)
    scored = [] #Empty list to store reranked outputs
 
    for item in results: #Iterating over hybrid retrieval output
        chunk = item["chunk"] #Pulls out actual text container
        prompt = RERANK_PROMPT.format( #Injects query and passage into template
            query=query,
            passage=chunk.text #Dynamically constructed query-passage pair prompt for each chunk
        )
 
        try: #Try block for robustness; Failures expected: bad JSON, API issues, formatting errors
            response = client.chat.completions.create( #Sending request to OpenAI
                model=model, #Uses configurable model
                messages=[{"role": "user", "content": prompt}], #Chat format: Single turn interaction; no system prompt (embedding in string)
                temperature=0,       # deterministic scoring; makes output deterministic, critical for scoring consistency
                max_tokens=100, #Limits response size to prevent cost blowups and ensures short JSON output
            )
            raw = response.choices[0].message.content #Pulls text from response
            if raw is None:
                raise ValueError("Empty response from API")
            raw = raw.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw) #Converts string -> Python dict
            rerank_score = float(parsed.get("score", 5)) #Extract score safely, default = 5 if missing
            reason = parsed.get("reason", "") #Extract explanation
        except Exception as e: #Catches bad JSON, missing fields, API errors
            rerank_score = item.get("score", 0.5)* 10 #Convert 0-1 score to 0-10 scale
            reason = f"Fallback scoring (parse error: {e})" #Keeps debugging trace
 
        scored.append({ #Store enriched result
            "chunk":          chunk, #Original data 
            "original_score": item["score"], #From hybrid retrieval
            "rerank_score":   rerank_score, #From LLM
            "rerank_reason":  reason, #Explainability
            "confidence_pct": int(rerank_score * 10),  # 0-100% for UI display
        })
 
    # Sort by rerank score descending
    scored.sort(key=lambda x: x["rerank_score"], reverse=True) 
 
    # Add final rank
    for i, item in enumerate(scored): 
        item["rank"] = i + 1 #i starts at 0 -> add 1 for human friendly rank
 
    return scored[:top_k] #Only keep best results to reduce token usage and improve answer quality
 

# ---------------------------------------------------------------------------
# Offline reranker (no API — used for testing and fallback)
# ---------------------------------------------------------------------------
 
def rerank_by_keyword_overlap(
    query: str, #User question
    results: list, #Retrieved chunks from hybrid search
    top_k: int = 3, #How many results to return
) -> list: #Reranked list of chunks
    """
    Simple keyword overlap reranker — no API needed.
    Used for offline testing and as a fast fallback.
 
    Not as good as LLM reranking but validates the pipeline structure.
    """
    query_tokens = set(query.lower().split()) #query.lower() -> normalize case .split() -> break into words set(..) -> remove duplicates
 
    scored = [] #Empty list for results
    for item in results: #Lopp retrieved chunks
        chunk_tokens = set(item["chunk"].text.lower().split()) #Same process as query
        overlap = len(query_tokens & chunk_tokens) #Compute overlap
        overlap_ratio = overlap / max(len(query_tokens), 1) #Normalize overlap so that long queries dont unfairly infale score
  
        # Combine original retrieval score with keyword overlap 
        combined = 0.6 * item["score"] + 0.4 * overlap_ratio
 
        scored.append({ 
            "chunk":          item["chunk"],
            "original_score": item["score"],
            "rerank_score":   combined * 10,
            "rerank_reason":  f"Keyword overlap: {overlap}/{len(query_tokens)} query terms matched",
            "confidence_pct": int(combined * 100),
            "rank":           0,  # assigned below
        })
 
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    for i, item in enumerate(scored):
        item["rank"] = i + 1
 
    return scored[:top_k]

# ---------------------------------------------------------------------------
# Reranker class — wraps both strategies
# ---------------------------------------------------------------------------
 
class Reranker:
    """
    Drop-in reranking layer between HybridRetriever and answer generation.
 
    Usage:
        retrieval_results = hybrid_retriever.search(query, api_key, top_k=10)
        reranked = reranker.rerank(query, retrieval_results, api_key, top_k=3)
        # Pass reranked to Day 6 answer generator
    """
 
    def __init__(self, strategy: str = "llm"):
        """
        strategy: "llm" | "keyword"
          "llm"     — GPT scoring, best quality, costs ~$0.0001/query
          "keyword" — overlap scoring, free, weaker, good for testing
        """
        assert strategy in ("llm", "keyword"), f"Unknown strategy: {strategy}"
        self.strategy = strategy
 
    def rerank(
        self,
        query: str,
        results: list,
        api_key: str = "",
        top_k: int = 3,
    ) -> list:
        if self.strategy == "llm":
            assert api_key, "api_key required for LLM reranking"
            return rerank_with_llm(query, results, api_key, top_k=top_k)
        else:
            return rerank_by_keyword_overlap(query, results, top_k=top_k)
 
    def format_for_context(self, reranked_results: list) -> str:
        """
        Format reranked chunks into a context string for the Day 6 LLM prompt.
        Each chunk is labeled so the LLM can reference it by number in citations.
        """
        parts = []
        for item in reranked_results:
            chunk = item["chunk"]
            label = f"[Source {item['rank']}]"
            citation = chunk.citation()
            parts.append(f"{label} {citation}\n{chunk.text}")
        return "\n\n".join(parts)
 

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", "")
 
    # Simulate retrieval results from Day 4
    sample_results = [
        {
            "chunk": Chunk(
                text="Metformin is the preferred initial pharmacologic agent for type 2 diabetes. It lowers HbA1c by 1-2% and has a favorable safety profile.",
                source="pubmed", pmid="28823139", title="ADA Standards of Care",
            ),
            "score": 0.82, "rank": 1,
        },
        {
            "chunk": Chunk(
                text="Blood glucose monitoring frequency should be individualized. Patients on insulin may need to monitor 4-10 times daily.",
                source="pubmed", pmid="29505530", title="Glucose Monitoring Review",
            ),
            "score": 0.74, "rank": 2,
        },
        {
            "chunk": Chunk(
                text="HbA1c reflects average plasma glucose over 2-3 months. Target below 7% for most non-pregnant adults with diabetes.",
                source="pubmed", pmid="27697747", title="HbA1c Targets Study",
            ),
            "score": 0.71, "rank": 3,
        },
        {
            "chunk": Chunk(
                text="Cardiovascular risk reduction is a primary treatment goal in type 2 diabetes. SGLT2 inhibitors and GLP-1 agonists show cardiovascular benefit.",
                source="pubmed", pmid="31580749", title="CV Risk in Diabetes",
            ),
            "score": 0.65, "rank": 4,
        },
        {
            "chunk": Chunk(
                text="Hypertension affects over 70% of patients with type 2 diabetes and significantly increases cardiovascular risk.",
                source="pubmed", pmid="30291960", title="Diabetes Comorbidities",
            ),
            "score": 0.58, "rank": 5,
        },
    ]
 
    query = "my blood glucose spikes to 300, what should I ask my doctor?"
 
    reranker = Reranker(strategy="keyword" if not api_key else "llm")
 
    print(f"Query: '{query}'")
    print(f"Strategy: {reranker.strategy}")
    print(f"Input: {len(sample_results)} chunks from hybrid retrieval\n")
 
    reranked = reranker.rerank(query, sample_results, api_key=api_key, top_k=3)
 
    print("=== Reranked Results ===")
    for r in reranked:
        print(f"\n#{r['rank']} | rerank={r['rerank_score']:.2f}/10 | confidence={r['confidence_pct']}%")
        print(f"   Was rank #{next(x['rank'] for x in sample_results if x['chunk'].pmid == r['chunk'].pmid)}")
        print(f"   Reason: {r['rerank_reason']}")
        print(f"   Text: {r['chunk'].text[:100]}...")
 
    print("\n=== Formatted Context for Day 6 LLM ===")
    print(reranker.format_for_context(reranked))
 
    # Verify rank changes happened (reranking is doing something)
    original_order = [r["chunk"].pmid for r in sample_results[:3]]
    reranked_order = [r["chunk"].pmid for r in reranked]
    if original_order != reranked_order:
        print("\n✓ Reranking changed the order — working correctly")
    else:
        print("\n~ Order unchanged (may happen with keyword strategy on this data)")
