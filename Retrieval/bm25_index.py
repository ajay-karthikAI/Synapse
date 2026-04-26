"""
Day 3: BM25 Keyword Search
==========================
Adds keyword-based retrieval alongside the vector search from Day 2.
Together they form the "hybrid" in Hybrid RAG.

Why BM25 specifically?
  BM25 (Best Match 25) is a probabilistic ranking function that improves
  on raw TF-IDF by accounting for:
    1. Term saturation — seeing "diabetes" 10x isn't 10x more relevant than 1x
    2. Document length normalization — short chunks shouldn't be unfairly penalized

  For medical text this matters because:
    - Drug names ("metformin", "sitagliptin") are exact tokens — semantic
      similarity won't help if the embedding space conflates them
    - ICD codes ("E11.9", "I10") are opaque to embedding models
    - Lab values ("HbA1c > 7%") need exact token matching

Where BM25 beats vectors:   exact terms, codes, drug names, numeric values
Where vectors beat BM25:    synonyms, paraphrases, conceptual queries
Combined (Day 4):           best of both
"""

import math
import pickle
from pathlib import Path
from collections import Counter #Counts frequencies; used from term frequency (TF) calculation in BM25

import sys
sys.path.append(str(Path(__file__).parent.parent)) #Adds project root directory to Python's import parth, allowing us to import from Data/ and other modules without worrying about relative paths.
from Data.fetch_and_chunk import Chunk 

from rank_bm25 import BM25Okapi #BMM25 is a ranking function used in search engies; scores documents based on TF, inverse doc frequency (IDF), document length normalizatio ; Works well for keyword search, fast and interpretable, enables hybrid retrieval


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """
    Simple whitespace + punctuation tokenizer.

    Why not use a more sophisticated tokenizer?
    - For BM25, simple tokenization is standard and works well
    - Lowercasing ensures "Metformin" matches "metformin"
    - Keeping hyphens intact preserves terms like "HbA1c", "SGLT2"

    What you'd add in production:
    - Medical stopword removal ("the", "a", "is" are already low-IDF)
    - UMLS concept normalization (maps synonyms to canonical terms)
    - Stemming/lemmatization (optional — BM25 handles this reasonably without)
    """
    text = text.lower()
    # Remove punctuation except hyphens (preserve drug names, codes); makes matching case-insensitive
    import re #Regular expression module; used for searching and replacing patterns in text; here we use it to remove punctuation while keeping hyphens
    text = re.sub(r'[^\w\s\-]', ' ', text) #Remove Punctuation but keep hyphens; []: character set; ^: NOT; \w: letters,numbers,underscores; \s: whitespace; \-: hyphens; AKA "Match anything that is NOT a letter/number, whitespace, or hyphen"; '  ' -> replace those characters with a space; NOTE: keep hyphens for medical speficie reasons i.e. COVID-19
    tokens = text.split() #Split string by whitespace into words
    # Filter very short tokens (usually noise)
    tokens = [t for t in tokens if len(t) > 1] #Keeps only tokens longer than 1 character; removes "a","i" (usually noise) as they dont help with retrieval
    return tokens #outputs clean token list


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Wraps rank_bm25.BM25Okapi with the same interface as VectorStore
    so Day 4 can treat both identically when merging results. Enables hybrid retrieval without downstream code beeding to know the underlying method
    """

    def __init__(self): #BM25 returns indices and scores, not text -> map back to chunks
        self.bm25 = None #Holds BM25 model
        self.chunks: list = [] #Stores original documents

    def build(self, chunks: list) -> None: #System learns the corpus 
        """
        Tokenize all chunks and build the BM25 index.
        No API calls, no cost — pure local computation.
        """
        self.chunks = chunks #Store chunks; keep reference to original docs
        print(f"[BM25] Tokenizing {len(chunks)} chunks...") #Logging; debugs visibility and shows pipeline progress

        tokenized = [tokenize(c.text) for c in chunks]  #Tokenization of corpus BM25 does NOT take raw text, but token lists so pretokenize them before indexing
        self.bm25 = BM25Okapi(tokenized) #Builds BM25 inverted index, allowing fast scoring at query time (builds TF< IDF< doc legnth stats)

        # Quick sanity check: avg tokens per chunk; basic corpus statistics to monitor prepreocessing quality
        avg_tokens = sum(len(t) for t in tokenized) / len(tokenized) #average # of tokens per chunk 
        print(f"[BM25] Index built. Avg tokens/chunk: {avg_tokens:.1f}\n") #Detects overly large or small chunks

    def search(self, query: str, top_k: int = 5) -> list: #Search Method; Retrieval step
        """
        Score all chunks against the query and return top-k.

        Returns list of {chunk, score, rank} dicts — same shape as
        VectorStore.search() so Day 4 merging code is clean.

        Note: BM25 scores are NOT normalized (unlike L2 distances from FAISS).
        A score of 0 means no term overlap at all.
        Day 4 will handle normalization before combining with vector scores.
        """
        assert self.bm25 is not None, "Call build() before search()" #Prevents runtime errors if index is not built

        query_tokens = tokenize(query) #Tokenize query
        scores = self.bm25.get_scores(query_tokens) #score documents by relevence to document BM25 is not normalized or probabilistic

        # Get top-k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = [] #Format results
        for rank, idx in enumerate(top_indices): #rank = position in results (0,1,2) idx = doc index
            results.append({ 
                "chunk": self.chunks[idx],
                "score": float(scores[idx]),
                "rank": rank + 1,
            })

        return results

    def get_all_scores(self, query: str) -> list: #Returns scores for every document
        """
        Return BM25 scores for ALL chunks (needed for Day 4 hybrid fusion).
        Length matches self.chunks exactly — position i = score for chunk i.
        """
        assert self.bm25 is not None, "Call build() before search()" #safety check
        query_tokens = tokenize(query)
        return self.bm25.get_scores(query_tokens).tolist() #Return full score vector; .tolist() converts numpy array to Python list; expose full score vectors to enavle hybrid fusion with dense retrieval methods

    def save(self, dir_path: str = "bm25_index") -> None: #Save Method (Persistence)
        path = Path(dir_path) #Create folder
        path.mkdir(exist_ok=True)
        with open(path / "bm25.pkl", "wb") as f: #Save BM25 model
            pickle.dump(self.bm25, f)
        with open(path / "chunks.pkl", "wb") as f: #Save chunks
            pickle.dump(self.chunks, f)
        print(f"[BM25] Saved to {dir_path}/") #log

    @classmethod #Load Method (Factory Method)
    def load(cls, dir_path: str = "bm25_index") -> "BM25Index":  #Created a fully reconstructed object
        path = Path(dir_path) 
        idx = cls() #Create instance
        with open(path / "bm25.pkl", "rb") as f: #Load BM25 model
            idx.bm25 = pickle.load(f)
        with open(path / "chunks.pkl", "rb") as f: #Load chunks
            idx.chunks = pickle.load(f)
        print(f"[BM25] Loaded index ({len(idx.chunks)} chunks) from {dir_path}/") #Log
        return idx


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Corpus designed to show WHERE BM25 beats vectors
    # (exact drug names, codes, numeric values)
    sample_chunks = [ #Test corpus design; 
        Chunk(text="Metformin 500mg is the first-line oral treatment for type 2 diabetes mellitus.", source="pubmed", pmid="1", title="Diabetes Treatment"),
        Chunk(text="Insulin resistance leads to elevated fasting glucose and HbA1c levels.", source="pubmed", pmid="2", title="Insulin Resistance"),
        Chunk(text="SGLT2 inhibitors such as empagliflozin reduce cardiovascular mortality.", source="pubmed", pmid="3", title="SGLT2 Review"),
        Chunk(text="ICD-10 code E11.9 refers to type 2 diabetes without complications.", source="pubmed", pmid="4", title="ICD Coding"),
        Chunk(text="Hypertension affects approximately 1.28 billion adults worldwide.", source="pubmed", pmid="5", title="Hypertension Epidemiology"),
        Chunk(text="GLP-1 receptor agonists promote weight loss and glycemic control.", source="pubmed", pmid="6", title="GLP-1 Agents"),
        Chunk(text="HbA1c greater than 6.5% on two separate tests confirms diabetes diagnosis.", source="pubmed", pmid="7", title="Diagnosis Criteria"),
    ]

    idx = BM25Index() #Index building (tokenization, BM25 indexing, doc frequency computation)
    idx.build(sample_chunks)

    # Three queries that show BM25's strengths vs weaknesses
    test_queries = [ #Testing system behavior
        ("metformin 500mg",         "exact drug + dose — BM25 should dominate"),
        ("ICD-10 E11.9",            "exact code — vectors would struggle here"),
        ("treatments for high blood sugar", "semantic query — vectors would do better"),
    ]

    for query, note in test_queries:
        print(f"Query: '{query}'")
        print(f"Note:   {note}")
        results = idx.search(query, top_k=3)
        for r in results:
            score_str = f"{r['score']:.3f}" if r['score'] > 0 else "0.000 (no match)"
            print(f"  #{r['rank']} | score={score_str} | {r['chunk'].text[:80]}...")
        print()

    # Show get_all_scores() — used in Day 4
    print("All scores for 'metformin' query (used in Day 4 fusion):")
    all_scores = idx.get_all_scores("metformin")
    for i, (chunk, score) in enumerate(zip(sample_chunks, all_scores)):
        bar = "█" * int(score * 3)
        print(f"  [{i}] {score:.3f} {bar} | {chunk.text[:60]}...")

    # Save/load test
    idx.save("bm25_test")
    idx2 = BM25Index.load("bm25_test")
    r = idx2.search("metformin", top_k=1)
    print(f"\n✓ Persistence check passed: '{r[0]['chunk'].text[:60]}...'")