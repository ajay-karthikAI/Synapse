"""
Day 2: Embeddings + FAISS Vector Store
=======================================
Turns text chunks into searchable vectors using OpenAI embeddings,
stored in a FAISS index for fast nearest-neighbor retrieval.
 
Key decisions documented here (know these for interviews):
 
1. Model: text-embedding-3-small
   - 1536 dimensions, ~$0.02/million tokens
   - Outperforms ada-002 on MTEB retrieval benchmarks
   - text-embedding-3-large exists but costs 10x more with ~5% gain
 
2. Index type: IndexFlatL2
   - Exact nearest neighbor search (no approximation)
   - Fine up to ~100k chunks; switch to IndexIVFFlat for larger corpora
   - L2 distance == cosine similarity on normalized vectors (we normalize)
 
3. Batching:
   - OpenAI API has a token limit per request
   - We batch at 100 chunks to stay safely under the limit

NOTE: PubMed API calls the content of this RAGbot; other files will use OPENAI API to convert content into vectors
"""


import os
import pickle #Serialization library; used to save/load Python objects (dicts, lists, embeddings)
import numpy as np #Numerical computing library
from pathlib import Path #Modern way to handle file paths; more robust than os.path
from typing import Optional #For type hints; helps with code readability and debugging
 
import faiss #Facebook AI Similarity Seatch; used fro fast vector similarity search; stores embeddings
from openai import OpenAI 
 
import sys
sys.path.append(str(Path(__file__).parent.parent)) #parent.parent = Go up 2 directories; sys.path.append adds that directory to Python's search path for imports; allows us to import from Data folder
from Data.fetch_and_chunk import Chunk


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
 
def get_embeddings(
    texts: list, #List of strings to embed
    client: OpenAI,
    model: str = "text-embedding-3-small", #OpenAI embedding model; text-embedding-3-small is a good balance of performance and cost for retrieval tasks
    batch_size: int = 100, #Texts per API call
) -> np.ndarray:
    """
    Embed a list of strings, batched to respect API limits.
    Returns float32 array of shape (n_texts, embedding_dim).
    """
    all_embeddings = [] #Initialize list to store embeddings; will store all vectors from all batches
 
    for i in range(0, len(texts), batch_size): #Loop over texts in steps of batch_size; texts/batch_size =. # of API calls needed
        batch = texts[i : i + batch_size] #Slice out the current batch of texts
        response = client.embeddings.create(input=batch, model=model) #Sends batch to OpenAI and returns embeddings
        batch_vecs = [item.embedding for item in response.data] #Extract embeddings from API response; respnse.data = list of results; each has .embedding 
        all_embeddings.extend(batch_vecs) #Add batch results to full lists
        if i > 0 and i % (batch_size * 5) == 0: #Print every 5 batches
            print(f"  Embedded {i}/{len(texts)} chunks...")
 
    arr = np.array(all_embeddings, dtype=np.float32) #Convert to NumPy; FAISS requires float32 arrays
 
    # L2-normalize so cosine similarity == dot product
    # This makes IndexFlatL2 behave like cosine similarity search
    norms = np.linalg.norm(arr, axis=1, keepdims=True) #Computes L2 norm (vector length) for each embedding
    arr = arr / np.maximum(norms, 1e-10) #Divides each vector by its magnitutde -> unite vector; 1e-10 prevents division by 0
 
    return arr
 

# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------
 
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2: #
    """
    Build an exact L2 index from an embedding matrix.
 
    IndexFlatL2 vs alternatives:
      - IndexFlatL2:    exact, no training, best for <100k chunks
      - IndexIVFFlat:   approximate, faster at scale, needs training
      - IndexHNSWFlat:  graph-based ANN, faster queries, more RAM
 
    For medical RAG where accuracy > speed, exact search is the right call.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index
 
 
# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------
 
class VectorStore:
    """
    Thin wrapper around FAISS that keeps chunks and their vectors in sync.
 
    Core invariant: chunks[i] corresponds to the vector at index position i.
    Never shuffle or delete — it breaks the alignment.
    """
 
    def __init__(self): #
        self.index: Optional[faiss.IndexFlatL2] = None #FAISS index (initially empty) IndexFlatL2: uses L2 distance for exact search
        self.chunks: list = [] #Stores original text chunks
        self.embeddings: Optional[np.ndarray] = None #Stores embedding matrix
        self.model: str = "text-embedding-3-small" #Tracks which model was used
 
    def build(self, chunks: list, api_key: str, model: str = "text-embedding-3-small") -> None: #Index Creation
        """Embed all chunks and build FAISS index."""
        self.chunks = chunks #Store inputs
        self.model = model #Store inputs
        client = OpenAI(api_key=api_key)
 
        print(f"[VectorStore] Embedding {len(chunks)} chunks with {model}...") #Logging
        texts = [c.text for c in chunks] #Extract raw text from chunk objects; assumes Chunk has .text attribute
        self.embeddings = get_embeddings(texts, client, model)
 
        print(f"[VectorStore] Building FAISS index (dim={self.embeddings.shape[1]})...") #Embedding dimension
        self.index = build_faiss_index(self.embeddings) #Build the FAISS index with the embeddings
        print(f"[VectorStore] Done. {self.index.ntotal} vectors indexed.\n") #ntotal = number of vectors in the index; should match len(chunks)
 
    def search(self, query: str, api_key: str, top_k: int = 5) -> list: 
        """
        Embed a query and return top-k most similar chunks.
        Returns list of {chunk, score, rank} dicts.
        Score is L2 distance — lower means more similar.
        """
        assert self.index is not None, "Call build() before search()" #Safety check toesure the index is initialized before querying
 
        client = OpenAI(api_key=api_key)
        query_vec = get_embeddings([query], client, self.model) #convert query to embedding; list inpute because function expects batch
        distances, indices = self.index.search(query_vec, top_k) #Returns 2 arrays: distances (similarity scores) and indices (positions of nearest neighbors)
                                                                 #Note: Score is L2 - lower means more similarr; embeddings were normalized earlier, so L2 distance is effectively cosine similarity
        results = [] 
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])): #iterate through results
            if idx == -1: #FAISS returns -1 if no matches found; skip these
                continue
            results.append({
                "chunk": self.chunks[idx], #idx -> lookup in self.chunks
                "score": float(dist),
                "rank": rank + 1,
            })
        return results
 
    def save(self, dir_path: str = "vector_store") -> None: #Save method (Persistence)
        path = Path(dir_path) #create directory if needed
        path.mkdir(exist_ok=True) 
        faiss.write_index(self.index, str(path / "index.faiss")) #Saves FAISS index to disk; index.faiss is the filename
        with open(path / "chunks.pkl", "wb") as f: #Saves chunk objects using pickle; chunks.pkl is the filename
            pickle.dump(self.chunks, f)
        np.save(str(path / "embeddings.npy"), self.embeddings) #Saves embeddings as a NumPy array; embeddings.npy is the filename
        print(f"[VectorStore] Saved to {dir_path}/")
 
    @classmethod
    def load(cls, dir_path: str = "vector_store") -> "VectorStore": #Createsinstance from disk
        path = Path(dir_path) 
        vs = cls() #New object
        vs.index = faiss.read_index(str(path / "index.faiss")) #Load FAISS index from disk
        with open(path / "chunks.pkl", "rb") as f: #Load chunks using pickle
            vs.chunks = pickle.load(f)
        vs.embeddings = np.load(str(path / "embeddings.npy")) #Load embeddings from NumPy file
        print(f"[VectorStore] Loaded {vs.index.ntotal} vectors from {dir_path}/")
        return vs
 

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
 
if __name__ == "__main__": #local testing and validation
    api_key = os.getenv("OPENAI_API_KEY", "") #Use environment variable to manage API keys securely; Avoids hardcoding secrets. Reads environment variable
 
    if not api_key:
        # -----------------------------------------------------------
        # Offline smoke test — verifies FAISS logic, no API cost
        # -----------------------------------------------------------
        print("No OPENAI_API_KEY found — running offline smoke test\n")
 
        fake_chunks = [ 
            Chunk(text=t, source="pubmed", pmid=str(i), title=f"Article {i}") #Dummy chunk objects
            for i, t in enumerate([ #Add realistic medical sentences to simulate real data using synthetich chunks to test FAISS logic without API calls
                "Insulin resistance causes elevated blood glucose levels.",
                "Metformin is first-line therapy for type 2 diabetes.",
                "HbA1c reflects average blood glucose over 3 months.",
                "Beta cell dysfunction precedes overt type 2 diabetes.",
                "SGLT2 inhibitors reduce cardiovascular mortality in diabetics.",
            ])
        ]
 
        dim = 1536 #Matches real embedding size
        np.random.seed(42) #Setting a seed ensures that the random vectors are the same every time we run the test, making it deterministic and easier to debug if something goes wrong.
        vecs = np.random.randn(len(fake_chunks), dim).astype(np.float32) #Generate random vectors to simulate embeddings; shape = (5, 1536)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) #Normalize to keep behaviour consistne with real pipeline; ensures L2 distance is meaningful
 
        vs = VectorStore() #Build Vectors Manually #Bypass API and directtly inject embeddings and chunks to test retrieval independently
        vs.chunks = fake_chunks 
        vs.embeddings = vecs
        vs.index = build_faiss_index(vecs)
 
        print(f"Index built: {vs.index.ntotal} vectors, dim={dim}")
 
        query_vec = np.random.randn(1, dim).astype(np.float32) #Test query vector
        query_vec /= np.linalg.norm(query_vec) #Fake query vector; normalized for consistency with indexed vectors
        distances, indices = vs.index.search(query_vec, 3) #FAISS search for top-3 nearest neighbors; returns distances and indices of closest vectors
 
        print("\nTop-3 results (random vectors, order is arbitrary):") 
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])): #Print Results
            print(f"  #{rank+1} | score={dist:.4f} | {fake_chunks[idx].text}")
 
        print("\n✓ Offline smoke test passed.")
        print("  Set OPENAI_API_KEY env var to test real semantic search.")
 
    else:
        # -----------------------------------------------------------
        # Live test — real embeddings, costs ~$0.0001
        # -----------------------------------------------------------
        sample_chunks = [
            Chunk(text=t, source="pubmed", pmid=str(i), title=f"Article {i}")
            for i, t in enumerate([
                "Insulin resistance causes elevated blood glucose levels.",
                "Metformin is first-line therapy for type 2 diabetes.",
                "HbA1c reflects average blood glucose over 3 months.",
                "Beta cell dysfunction precedes overt type 2 diabetes.",
                "SGLT2 inhibitors reduce cardiovascular mortality in diabetics.",
                "Hypertension is a major risk factor for cardiovascular disease.",
            ])
        ]
 
        vs = VectorStore() #Text -> Embeddings -> FAISS Index; This is the real test of the full pipeline, including OpenAI embeddings and FAISS search
        vs.build(sample_chunks, api_key=api_key)
        vs.save("vector_store_test") #Tests persistence
 
        query = "What medication treats high blood sugar?" #Realistic user query
        print(f"Query: '{query}'\n")
        results = vs.search(query, api_key=api_key, top_k=3) #Semantic retrieval
 
        for r in results:
            print(f"  #{r['rank']} | score={r['score']:.4f} | {r['chunk'].text}") #Print results
            print(f"       {r['chunk'].citation()}") #Print citations
 
        # Verify save/load round-trip
        vs2 = VectorStore.load("vector_store_test") #Reload from disk
        r2 = vs2.search(query, api_key=api_key, top_k=1) #Run same query
        print(f"\n✓ Persistence check passed: '{r2[0]['chunk'].text}'") #Provessave/load works correctly
 