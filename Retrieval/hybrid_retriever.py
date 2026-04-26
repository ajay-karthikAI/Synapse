"""
Day 4: Hybrid Retrieval
========================
Combines BM25 (Day 3) and FAISS vector search (Day 2) into a single
ranked list using two fusion strategies:
 
  1. Weighted Linear Fusion  — normalize scores to [0,1], weighted sum
  2. Reciprocal Rank Fusion  — rank-based, score-agnostic, more robust
 
Interview talking point:
  "I implemented both fusion strategies. Linear fusion is more intuitive
  and lets you tune alpha based on query type — higher alpha favors
  semantic search, lower alpha favors keyword precision. RRF is more
  robust when score distributions shift across different corpora, which
  matters in production where you can't retune alpha for every dataset."
"""
 
import os 
import numpy as np
from pathlib import Path
 
import sys
sys.path.append(str(Path(__file__).parent.parent)) #Adds parent directory of this file's parent to Python's modue searach path
from Data.fetch_and_chunk import Chunk 
from Retrieval.vector_store import VectorStore, get_embeddings, build_faiss_index #Imports semantic retrieval pipeline; VectorStore: abstraction for vetor search; get_embeddings: converts text to embeddings; build_faiss_index: builds FAISS index
from Retrieval.bm25_index import BM25Index
 
from openai import OpenAI


 
# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------
 
def min_max_normalize(scores: list) -> list: #Define fucntion that takes a list of numbers (scores) and return a normalized version
    """
    Scale scores to [0, 1].
    Handles edge case where all scores are equal (returns all 0.5).
    """
    arr = np.array(scores, dtype=np.float32) #Converts python list to numpy array; uses float32 for efficiency
    mn, mx = arr.min(), arr.max() #Find min score (mn) and max score (mx)
    if mx - mn < 1e-10: #Checks if all values are basically the same; 1e-10 avoids division by zero
        return [0.5] * len(scores) #If all scores are equal -> return neutral value 0.5 (middle of [0,1] and doesnt bias ranking)
    return ((arr - mn) / (mx - mn)).tolist() #minmax normalization (x-min/max-min) to preserve relative ranking while aligning score ranges
 
 
def invert_distances(distances: list) -> list: #Takes FAISS distances as input and converts them to similarity scores
    """
    FAISS returns L2 distances — lower is better.
    Convert to similarity scores — higher is better — before normalizing.
    Simple inversion: similarity = 1 / (1 + distance)
    """
    return [1.0 / (1.0 + d) for d in distances] #Similarity = 1/(1+d); if distance = 0. then simalirity = 1 (perfect match)
 

# ---------------------------------------------------------------------------
# Fusion Strategy 1: Weighted Linear Fusion
# ---------------------------------------------------------------------------
 
def linear_fusion(
    vector_scores_all: list,   # score for every chunk from vector search #FAISS distances (not yet similarities)
    bm25_scores_all: list,     # score for every chunk from BM25 
    chunks: list,              # actual data objects
    alpha: float = 0.7,        # weight for vector search (1-alpha for BM25) #Controls balance between methods
    top_k: int = 5,            # How many results to return
) -> list:                     # Implemented weighted linear fusion where alpha controls the contribution of semantic vs keyword retrieval
    """
    Normalize both score arrays to [0,1] then compute weighted sum.
 
    alpha=0.7 means 70% vector, 30% BM25.
    Tune this based on your query mix:
      - More clinical/semantic queries → higher alpha
      - More code/drug/exact queries  → lower alpha
 
    Returns top_k results as {chunk, score, rank, vector_score, bm25_score}
    """
    # Convert FAISS distances to similarities (higher = better)
    vector_sim = invert_distances(vector_scores_all) #Flips distance to similarity becase you cant combine lower is better with higher is better
 
    # Normalize both to [0, 1]; This ensures both retreial methods contribute proportionally
    norm_vector = min_max_normalize(vector_sim)
    norm_bm25   = min_max_normalize(bm25_scores_all)
 
    # Weighted sum
    combined = [
        alpha * v + (1 - alpha) * b  # score = alpha x vector + (1-alpha) x BM25; If alpha = 0.7 -> 70% semantic meaning 30% keyword search
        for v, b in zip(norm_vector, norm_bm25) #This approach assumes both score arrays correspond to the same document ordering
    ]
 
    # Rank by combined score
    top_indices = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:top_k] #Create list of indices [0,1,2,3...] then sorth by combined_score; takes top k
 
    return [ #Returns list of dictionaries containing
        {
            "chunk":        chunks[i],
            "score":        combined[i], #Final score
            "vector_score": norm_vector[i], #Individual component scores
            "bm25_score":   norm_bm25[i], #Individual component scores
            "rank":         rank + 1,
            "method":       "linear_fusion", #Method used
        }
        for rank, i in enumerate(top_indices) #Return both component scores for interpretability and debugging, which is important in production systems
    ]

# ---------------------------------------------------------------------------
# Fusion Strategy 2: Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
 
def reciprocal_rank_fusion( #Rank based Fusion Function uses ranks ONLY as opposed to scorebased linear fusion
    vector_results: list,   # top-k results from VectorStore.search() 
    bm25_results: list,     # top-k results from BM25Index.search()
    chunks: list,
    k: int = 60,            # RRF constant — 60 is the standard default # k = 60 is large enough to smooth differnces between ranks and small enough to reward top ranked items
    top_k: int = 5,
) -> list:
    """
    RRF score for a chunk = sum of 1/(k + rank) across all result lists. ### Higher ranked itesm contribute more than lower ranked items but all still contribute
 
    Why k=60? It's the empirically validated default from the original
    RRF paper (Cormack et al., 2009) that balances high/low rank influence.
 
    Why RRF over linear fusion?
    - Score distributions vary wildly across queries and corpora
    - RRF only cares about rank order, not raw score values
    - More stable across domains without retuning
 
    Downside: you lose the ability to weight one source over the other
    for specific query types.
    """
    # Build chunk_id → chunk lookup
    chunk_lookup = {c.chunk_id(): c for c in chunks} #Ensures you can uniquely ID chunks (same chunk can appear in both result lists); use chunk IDs to deduplicate and merge results across retrieval methods
 
    # Accumulate RRF scores keyed by chunk_id
    rrf_scores: dict = {} #Initialize score dictionary
 
    for result_list in [vector_results, bm25_results]: #Iterates over vector results and BM25 results to treat both retreival methods equally
        for item in result_list: #Process each result;
            cid = item["chunk"].chunk_id() # Extract unique ID for the chunk to merge duplicates across lists
            if cid not in rrf_scores: 
                rrf_scores[cid] = {"score": 0.0, "chunk": item["chunk"]} #Initializes score if chunk is seen for the first time
            rrf_scores[cid]["score"] += 1.0 / (k + item["rank"]) #Adds RRF contribution; Rank 1 -> high contribution Rank 10 low contribution #Example: k = 60; Rank 1 -> 1/61 = 0.016 Rank 5 -> 1/65 = 0.015 Rank 20 -> 1/80 = 0.012
 
    # Sort by RRF score descending
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k] #Sorts chunks by accumulated RRF score; takes top_k
 
    return [ #Returns clean structured output
        {
            "chunk":  r["chunk"], #Chunk
            "score":  r["score"], #RRF score
            "rank":   rank + 1, #Final rank
            "method": "rrf", #Method label
        }
        for rank, r in enumerate(sorted_results)
    ]
 

 
# ---------------------------------------------------------------------------
# HybridRetriever — the main class Days 5-7 will use
# ---------------------------------------------------------------------------
 
class HybridRetriever: #Unified retrieval system #Wrap both retrieval systems (FAISS and RRF fusion), Expose one interface, Choose fusion strategy (Linear or RRF), Handle persistence (save/load)
    """
    Unified retriever that wraps both indexes and exposes a single
    .search() method. Days 5, 6, 7 only need to call this. #Hide complexity and provide clean API, basically allowing later components to treat retrieval as a black box
    """
 
    def __init__(self, fusion: str = "linear", alpha: float = 0.7):  #Constructor #Initializes retriever with fusion strategy and alpha weight (only for linear fusion)
        """ 
        fusion: "linear" or "rrf"
        alpha:  vector weight for linear fusion (ignored for rrf)
        """
        self.vector_store = VectorStore() #Dense retriever
        self.bm25_index   = BM25Index() #Sparse retriever
        self.fusion       = fusion #Stores fusion method choice
        self.alpha        = alpha #Stores tuning parameter
        self.chunks: list = [] #Stores dataset reference
 
    def build(self, chunks: list, api_key: str) -> None: #Builds both indexes from raw data
        """Build both indexes from the same chunk list."""
        self.chunks = chunks #Stores original dataset for later alignment
        print(f"[HybridRetriever] Building indexes ({len(chunks)} chunks)...") #Debug visibility (useful in pipelines)
        self.vector_store.build(chunks, api_key=api_key) #Embeddings + FAISS index
        self.bm25_index.build(chunks) #Token based inverted index
        print(f"[HybridRetriever] Ready. Fusion={self.fusion}, alpha={self.alpha}\n") #Confirms system is ready and shows configuration
 
    def search(self, query: str, api_key: str, top_k: int = 5) -> list: #Main API Search Function
        """
        Run hybrid retrieval and return top_k fused results.
        Each result: {chunk, score, rank, method, vector_score*, bm25_score*}
        (* only present for linear fusion)
        """
        if self.fusion == "linear": #Branch 1: Linear Fusion
            # Need full score arrays for all chunks
            vector_results_topk = self.vector_store.search(query, api_key, top_k=len(self.chunks)) #Vector search (full corpus); need scores for every chunk, not just top-k
            # Reconstruct full distance array aligned to self.chunks
            score_map = {r["chunk"].chunk_id(): r["score"] for r in vector_results_topk} #Build score map; converts list -> dict, enables fast lookup by chunk ID
            vector_scores_all = [score_map.get(c.chunk_id(), 2.0) for c in self.chunks] #Reconstruct full aligned array; every chunk has a score and missing chunks get default value of 2.0
            bm25_scores_all   = self.bm25_index.get_all_scores(query) #Gets BM25 score for every chunk
 
            return linear_fusion( #Call fusion
                vector_scores_all, bm25_scores_all,
                self.chunks, self.alpha, top_k
            )
  
        elif self.fusion == "rrf": #Branch 2: RRF
            vector_results = self.vector_store.search(query, api_key, top_k=top_k * 2) #Vector top-k retrieval; *2 improves recall RRF needs mulptiple candidates not just top results
            bm25_results   = self.bm25_index.search(query, top_k=top_k * 2) #BM25 top-k retrieval
            return reciprocal_rank_fusion( #Fuse
                vector_results, bm25_results,
                self.chunks, top_k=top_k
            )
 
        else: #Error Handling
            raise ValueError(f"Unknown fusion method: {self.fusion}") #Prevents silent failures; forces correct configuration
 
    def save(self, dir_path: str = "hybrid_index") -> None: #SAVE METHOD Persists both systems
        self.vector_store.save(f"{dir_path}/vector") #Saves FAISS index
        self.bm25_index.save(f"{dir_path}/bm25") #Saves BM25 index
        print(f"[HybridRetriever] Saved to {dir_path}/")
 
    @classmethod #LOAD METHOD
    def load(cls, dir_path: str = "hybrid_index", fusion: str = "linear", alpha: float = 0.7) -> "HybridRetriever": #Factory method to restore system state
        hr = cls(fusion=fusion, alpha=alpha)  #Create new instance
        hr.vector_store = VectorStore.load(f"{dir_path}/vector") #Load saved index
        hr.bm25_index   = BM25Index.load(f"{dir_path}/bm25") #Load saved index
        hr.chunks       = hr.bm25_index.chunks #BM25 is used as source of truth for chunk ordering
        return hr
 
 
# ---------------------------------------------------------------------------
# Validation (offline — no API key needed)
# ---------------------------------------------------------------------------
 
if __name__ == "__main__": #Runs this code when file is executed directly to prevent running during imports
    api_key = os.getenv("OPENAI_API_KEY", "") #Read API key from .env
 
    sample_chunks = [ #Small test corpus that coveres different query types (drug names, medical concepts, codes)
        Chunk(text="Metformin 500mg is the first-line oral treatment for type 2 diabetes mellitus.", source="pubmed", pmid="1", title="Diabetes Treatment"),
        Chunk(text="Insulin resistance leads to elevated fasting glucose and HbA1c levels above 6.5%.", source="pubmed", pmid="2", title="Insulin Resistance"),
        Chunk(text="SGLT2 inhibitors such as empagliflozin reduce cardiovascular mortality in diabetic patients.", source="pubmed", pmid="3", title="SGLT2 Review"),
        Chunk(text="ICD-10 code E11.9 refers to type 2 diabetes mellitus without complications.", source="pubmed", pmid="4", title="ICD Coding"),
        Chunk(text="GLP-1 receptor agonists promote weight loss and improve glycemic control in obese patients.", source="pubmed", pmid="5", title="GLP-1 Agents"),
        Chunk(text="HbA1c greater than 6.5% on two separate tests confirms a diagnosis of diabetes.", source="pubmed", pmid="6", title="Diagnosis Criteria"),
        Chunk(text="Hypertension affects over 1 billion adults and is a leading cause of cardiovascular disease.", source="pubmed", pmid="7", title="Hypertension"),
    ]
 
    if not api_key: #If no API, simulate everything
        # ------------------------------------------------------------------
        # Offline test: use random vectors to validate fusion logic
        # ------------------------------------------------------------------
        print("No OPENAI_API_KEY — running offline fusion logic test\n") #Debug message
 
        import numpy as np
        dim = 1536 #Embedding dimension (matches OPENAI)
        np.random.seed(42) #Reproducibility
 
        # Build fake vector store
        vecs = np.random.randn(len(sample_chunks), dim).astype(np.float32) #Generate random embeddings
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) #Normalizes vectors unit length
        vs = VectorStore() #Manually constructs vector store without API by bypassing embedding generation
        vs.chunks = sample_chunks
        vs.embeddings = vecs
        vs.index = build_faiss_index(vecs)
 
        # Build BM25 index
        bm25 = BM25Index()
        bm25.build(sample_chunks)
 
        # Test linear fusion (Semantic query should favor vectors)
        query = "treatments for high blood sugar"
        print(f"Query: '{query}'")
        print(f"\n--- Linear Fusion (alpha=0.7) ---")
 
        query_vec = np.random.randn(1, dim).astype(np.float32) #Fake query embedding
        query_vec /= np.linalg.norm(query_vec)
        distances, indices = vs.index.search(query_vec, len(sample_chunks)) #Gets distances from FAISS
 
        # Reconstruct full distance array
        dist_map = {int(idx): float(dist) for idx, dist in zip(indices[0], distances[0])} #Align scores with chunks
        vector_scores_all = [dist_map.get(i, 2.0) for i in range(len(sample_chunks))] #Fill missing with penalty
        bm25_scores_all = bm25.get_all_scores(query)
 
        linear_results = linear_fusion(vector_scores_all, bm25_scores_all, sample_chunks, alpha=0.7, top_k=4) #Run fusion
        for r in linear_results:
            print(f"  #{r['rank']} | combined={r['score']:.3f} vec={r['vector_score']:.3f} bm25={r['bm25_score']:.3f} | {r['chunk'].text[:65]}...")
 
        # Test RRF
        print(f"\n--- Reciprocal Rank Fusion ---") 
        # Fake vector results
        vec_results = [{"chunk": sample_chunks[int(indices[0][i])], "score": float(distances[0][i]), "rank": i+1} for i in range(len(sample_chunks))] #Construct fake ranked list
        bm25_results = bm25.search(query, top_k=len(sample_chunks))
        rrf_results = reciprocal_rank_fusion(vec_results, bm25_results, sample_chunks, top_k=4)
        for r in rrf_results:
            print(f"  #{r['rank']} | rrf_score={r['score']:.4f} | {r['chunk'].text[:65]}...")
 
        print("\n✓ Fusion logic validated offline.")
        print("  Set OPENAI_API_KEY to test with real semantic vectors.")
 
    else:
        # ------------------------------------------------------------------
        # Live test with real embeddings
        # ------------------------------------------------------------------
        hr = HybridRetriever(fusion="linear", alpha=0.7)
        hr.build(sample_chunks, api_key=api_key)
 
        queries = [
            ("metformin 500mg",                  "exact — BM25 should boost this"),
            ("treatments for high blood sugar",   "semantic — vectors should lead"),
            ("ICD-10 E11.9",                      "exact code — BM25 critical"),
        ]
 
        for query, note in queries:
            print(f"\nQuery: '{query}'  [{note}]")
            results = hr.search(query, api_key=api_key, top_k=3)
            for r in results:
                print(f"  #{r['rank']} | combined={r['score']:.3f} | {r['chunk'].text[:70]}...")
 
        # Compare linear vs RRF on one query
        hr_rrf = HybridRetriever(fusion="rrf")
        hr_rrf.vector_store = hr.vector_store
        hr_rrf.bm25_index   = hr.bm25_index
        hr_rrf.chunks       = hr.chunks
 
        print(f"\n--- Linear vs RRF comparison ---")
        q = "diabetes treatment options"
        linear_r = hr.search(q, api_key=api_key, top_k=3)
        rrf_r    = hr_rrf.search(q, api_key=api_key, top_k=3)
 
        print(f"Linear: {[r['chunk'].text[:45] for r in linear_r]}")
        print(f"RRF:    {[r['chunk'].text[:45] for r in rrf_r]}")