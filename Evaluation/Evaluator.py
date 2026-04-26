"""
Day 7: Evaluation Metrics
=========================
Measures retrieval quality so you can actually validate the pipeline
is working — not just assume it is.

Metrics implemented:
  1. Recall@k     — of the relevant chunks, how many did we retrieve?
  2. Precision@k  — of the retrieved chunks, how many are actually relevant?
  3. MRR          — Mean Reciprocal Rank (how high does the first relevant
                    chunk appear?)
  4. Failure log  — tracks bad retrievals for manual review

Interview talking point:
  "I added evaluation because retrieval quality isn't visible from the
  UI. A system can look like it's working while silently returning
  irrelevant chunks. Recall@k tells me if my hybrid retrieval is
  actually capturing the right documents, and MRR tells me if the
  most relevant chunk is appearing near the top where the LLM
  will weight it most."
"""

import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Evaluation data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """
    A single evaluation test case.
    relevant_pmids: PMIDs that SHOULD appear in top-k results for this query.
    """
    query: str
    relevant_pmids: list
    notes: str = ""


@dataclass
class RetrievalResult:
    query: str
    retrieved_pmids: list       # in rank order
    relevant_pmids: list
    recall_at_k: float
    precision_at_k: float
    mrr: float
    k: int
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    """
    Of the relevant documents, what fraction did we retrieve in top-k?

    recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Example: relevant=[A,B,C], retrieved=[A,D,B] → recall@3 = 2/3 = 0.67

    Why this matters:
      Low recall means the LLM is answering without seeing the best
      evidence. The answer will be incomplete or wrong.
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)


def precision_at_k(retrieved: list, relevant: list, k: int) -> float:
    """
    Of the top-k retrieved documents, what fraction are actually relevant?

    precision@k = |retrieved[:k] ∩ relevant| / k

    Why this matters:
      Low precision means the LLM context is diluted with irrelevant
      chunks, increasing hallucination risk.
    """
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / k


def mean_reciprocal_rank(retrieved: list, relevant: list) -> float:
    """
    What rank position is the FIRST relevant document?
    MRR = 1/rank of first relevant hit (0 if none found)

    Why this matters:
      Reranking should push the most relevant chunk to position 1.
      MRR measures how well that's working. MRR of 1.0 = perfect.
      MRR of 0.5 = first relevant chunk is at rank 2.
    """
    relevant_set = set(relevant)
    for rank, pmid in enumerate(retrieved, start=1):
        if pmid in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs evaluation queries against your retrieval pipeline and
    tracks results over time for regression detection.
    """

    # Built-in eval set for the waiting room use case
    # These are example queries — in production you'd label these
    # manually by running queries and marking which results were correct
    DEFAULT_EVAL_SET = [
        EvalQuery(
            query="what causes blood glucose spikes",
            relevant_pmids=["29505530", "28823139", "27697747"],
            notes="Core diabetes query"
        ),
        EvalQuery(
            query="first line treatment for type 2 diabetes",
            relevant_pmids=["28823139", "31580749"],
            notes="Treatment query — should retrieve metformin content"
        ),
        EvalQuery(
            query="what is HbA1c and why does it matter",
            relevant_pmids=["27697747", "29505530"],
            notes="Lab value explanation query"
        ),
        EvalQuery(
            query="symptoms of high blood pressure",
            relevant_pmids=[],   # empty = no ground truth yet
            notes="Hypertension query — needs labeling"
        ),
    ]

    def __init__(self, log_path: str = "eval_log.json"):
        self.log_path = Path(log_path)
        self.results: list = []
        self._load_log()

    def _load_log(self):
        if self.log_path.exists():
            with open(self.log_path) as f:
                self.results = json.load(f)

    def _save_log(self):
        with open(self.log_path, "w") as f:
            json.dump(self.results, f, indent=2)

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_results: list,    # from HybridRetriever.search()
        relevant_pmids: list,
        k: int = 5,
    ) -> RetrievalResult:
        """
        Score a single retrieval against known-relevant PMIDs.
        Logs result for regression tracking.
        """
        retrieved_pmids = [r["chunk"].pmid for r in retrieved_results[:k]]

        result = RetrievalResult(
            query=query,
            retrieved_pmids=retrieved_pmids,
            relevant_pmids=relevant_pmids,
            recall_at_k=recall_at_k(retrieved_pmids, relevant_pmids, k),
            precision_at_k=precision_at_k(retrieved_pmids, relevant_pmids, k),
            mrr=mean_reciprocal_rank(retrieved_pmids, relevant_pmids),
            k=k,
            timestamp=datetime.datetime.now().isoformat(),
        )

        self.results.append(asdict(result))
        self._save_log()
        return result

    def log_failure(
        self,
        query: str,
        retrieved_results: list,
        reason: str,
    ) -> None:
        """
        Manually log a bad retrieval for later analysis.
        Call this when you notice the app returning irrelevant results.
        """
        failure = {
            "type": "failure",
            "query": query,
            "reason": reason,
            "retrieved": [r["chunk"].text[:100] for r in retrieved_results[:3]],
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.results.append(failure)
        self._save_log()
        print(f"[Evaluator] Failure logged: {reason}")

    def summary(self) -> dict:
        """Aggregate metrics across all logged evaluations."""
        eval_results = [r for r in self.results if "recall_at_k" in r]
        failures = [r for r in self.results if r.get("type") == "failure"]

        if not eval_results:
            return {"message": "No evaluations logged yet"}

        avg_recall    = sum(r["recall_at_k"] for r in eval_results) / len(eval_results)
        avg_precision = sum(r["precision_at_k"] for r in eval_results) / len(eval_results)
        avg_mrr       = sum(r["mrr"] for r in eval_results) / len(eval_results)

        return {
            "total_evaluations": len(eval_results),
            "total_failures":    len(failures),
            "avg_recall_at_k":   round(avg_recall, 3),
            "avg_precision_at_k": round(avg_precision, 3),
            "avg_mrr":           round(avg_mrr, 3),
            "recent_results":    eval_results[-5:],
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluator = Evaluator(log_path="eval_log_test.json")

    # Simulate retrieval results
    from Data.fetch_and_chunk import Chunk

    mock_retrieved = [
        {"chunk": Chunk(text="...", source="pubmed", pmid="29505530", title="Glucose Monitoring"), "score": 0.9, "rank": 1},
        {"chunk": Chunk(text="...", source="pubmed", pmid="28823139", title="ADA Standards"),      "score": 0.8, "rank": 2},
        {"chunk": Chunk(text="...", source="pubmed", pmid="99999999", title="Irrelevant Paper"),   "score": 0.7, "rank": 3},
        {"chunk": Chunk(text="...", source="pubmed", pmid="27697747", title="HbA1c Targets"),      "score": 0.6, "rank": 4},
        {"chunk": Chunk(text="...", source="pubmed", pmid="88888888", title="Another Irrelevant"), "score": 0.5, "rank": 5},
    ]

    relevant = ["29505530", "28823139", "27697747"]
    query = "what causes blood glucose spikes"

    result = evaluator.evaluate_retrieval(query, mock_retrieved, relevant, k=5)

    print(f"Query: '{query}'")
    print(f"Recall@5:    {result.recall_at_k:.2f}  (retrieved {sum(1 for p in result.retrieved_pmids if p in relevant)}/{len(relevant)} relevant)")
    print(f"Precision@5: {result.precision_at_k:.2f} ({sum(1 for p in result.retrieved_pmids if p in relevant)} relevant in top 5)")
    print(f"MRR:         {result.mrr:.2f}  (first relevant at rank {int(1/result.mrr) if result.mrr > 0 else 'N/A'})")

    # Log a failure
    evaluator.log_failure(query, mock_retrieved[:2], "Top result missing key glucose variability content")

    print(f"\nSummary: {evaluator.summary()}")
    print("\n✓ Evaluation module working")