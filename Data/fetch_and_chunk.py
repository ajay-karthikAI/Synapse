"""
Day 1: Data Ingestion + Chunking
================================================================
Data sources supported:
  - PubMed abstracts via NCBI E-utilities (no API key needed)
  - Local PDF files via PyPDFLoader
  - Raw .txt files
 
Chunking:
  - RecursiveCharacterTextSplitter (LangChain) — respects sentence/paragraph
    boundaries before falling back to character splits. Better than naive
    sliding-window for preserving semantic units.
 
Metadata:
  - Custom Chunk dataclass keeps citation data (PMID / filename / page)
    first-class rather than buried in a generic dict.
 
Persistence:
  - Chunks saved to .pkl so you don't re-fetch on every run.

NOTE: PubMed API calls the content of this RAGbot; other files will use OPENAI API to convert content into vectors
"""
 
from ast import If
import time
import os
import pickle # For saving/loading chunked data
import re # For cleaning text and extracting patterns (PMIDs, citations, etc)
import xml.etree.ElementTree as ET # For parsing PubMed XML responses; Extracting IDs, titles, authors, etc.
from dataclasses import dataclass #Creates and defines datastructures 
from pathlib import Path # For handling file paths in a cross-platform way (Mac/Linux/Windows)
import urllib.request #Calling PubMed API via HTTP requests; Encoding query parameters; Handling rate limits with time.sleep
import urllib.parse 
import json # For parsing JSON responses from APIs (if needed in future extensions)
 
from langchain_text_splitters import RecursiveCharacterTextSplitter # For chunking text while respecting semantic boundaries (sentences, paragraphs)
 
# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
 
@dataclass #Automatically generates init (constructor), repr (nice printing), eq (comparison); makes classs behave like a structured data container
class Chunk:
    text: str #Chunked text content that will go into embeddings and LLM prompts
    source: str #Tracks where the chunk came from; can branch logic later (format citations, filter sources, weight trust)         # "pubmed" | "pdf" | "txt"
    pmid: str = "" #For PubMed chunks, store the PMID for citation and traceability; empty for non-PubMed sources
    title: str = "" #Title of the source document (PubMed article title, PDF filename, or txt filename) for citation and display in UI
    chunk_index: int = 0 #Position of chunk within document; chunk 0 = first chunk, chunk 1 = 2nd chunk, etc. Useful for reconstructing order and for PDF page tracking.
    total_chunks: int = 1 #Total number of chunks from the same source; reconstructs context later, useful for UI like "Chunk 2 of 8"
    source_url: str = "" #Link to original source (PubMed article URL, local file path for PDFs/txts) for traceability and potential future retrieval
    page: int = -1 #Page number; -1 means "not applicable" Keeps type consistent (int always); avoids None checks       # PDF only
 
    def citation(self) -> str: #Generaes a human-readable citation string
        if self.source == "pubmed":
            return f"[PMID:{self.pmid}] {self.title}"
        elif self.source == "pdf": #Adds page number only if it exists (>=0); avoids cluttering citation with "p.-1"
            pg = f" p.{self.page}" if self.page >= 0 else ""
            return f"[{self.title}{pg}]"
        return f"[{self.title}]"
 
    def chunk_id(self) -> str: #Creates a unique identifier for aeach chunk; useful for debugging, logging, and potential future extensions (e.g., storing in a database with unique keys)
        return f"{self.pmid or self.title}_chunk{self.chunk_index}" #If PMID exists, use it; otherwise fallback to title; appends chunk index for uniqueness within the same source
    
#   If chunk_id breaks because of long titles/special characters, run this instead
#     def chunk_id(self) -> str:
#     base = self.pmid if self.pmid else re.sub(r"\W+", "_", self.title)
#     return f"{base}_chunk{self.chunk_index}"


    
    print("WORKS")
 
 
# ---------------------------------------------------------------------------
# Splitter
# Why RecursiveCharacterTextSplitter?
#   Tries \n\n → \n → "." → " " in order, so chunks break at natural
#   boundaries instead of mid-sentence. Mid-sentence chunks retrieve poorly
#   because the embedding vector is semantically ambiguous.
# ---------------------------------------------------------------------------
 
def make_splitter(chunk_size: int = 500, overlap: int = 100): #Configurable function; chunk size = 500 overlap = 100; overlap preserves continuity and improves retrieval quality
    return RecursiveCharacterTextSplitter( #Langchain's hierarchal text splitter; tries multiple separators in order to preserve semantic units; falls back to character splits if needed
        chunk_size=chunk_size, #Maximum size per chunk; characters over tokens because characters compute faster and are model agnostic; 500 chars ~ 100 tokens, which is a good balance of context and retrieval performance
        chunk_overlap=overlap, #Repeats part of previous chunk in the next chunk; 100 characters is enough to preserve context but not too large ot dupllicate noise
        separators=["\n\n", "\n", ".", " "], #Priority hierarchy for splitting; tries to split at double newlines first (paragraphs), then single newlines (line breaks), then periods (sentences), then spaeces (words) as a last resort; preserves semantic coherence in chunks; progressive degradation
        length_function=len, #Measures length of chunk size with character count
    )

print("WORKS")

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
 
def clean_text(text: str) -> str: #Takes in raw text, returns clean text
    text = re.sub(r'\s+', ' ', text) #Collapses white space (spaces, tabs, newlines) into single spaces; normalizes formatting and prevents weird spacing issues in chunks
    text = re.sub(r'\n+', '\n', text) #Normalize newlines to single \n; preserves intentional line breaks while preventing excessive blank lines
    return text.strip() #Removes leading/trailing whitespace; ensures clean start and end of text; prevents chunks with just spaces
 
print("WORKS")

# ---------------------------------------------------------------------------
# Source A: PubMed
# ---------------------------------------------------------------------------
 
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils" #Base URL for NCBI E-utilities API; used for both searching and fetching; no API key needed
 
def search_pubmed(query: str, max_results: int = 20) -> list: #Convert a natural language query into a list of PMIDs; uses ESearch endpoint; returns list of PMIDs as strings
    params = urllib.parse.urlencode({ #Turns Python dict into URL query string
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    })

    url = f"{EUTILS_BASE}/esearch.fcgi?{params}" #Constructs full URL for ESearch request; e.g., https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=cancer&retmax=20&retmode=json
    with urllib.request.urlopen(url) as resp: #Sends HTTP GET request to PubMed API; handles response as a context manager to ensure proper resource cleanup; resp is the HTTP response object
        data = json.loads(resp.read()) #Parses JSON; converts raw bytes from response into Python dict
    return data["esearchresult"]["idlist"] #Extract PMIDs from the response; returns list of PMIDs as strings
 
 
def fetch_abstracts(pmids: list) -> list:
    if not pmids:
        return []
    ids = ",".join(pmids)
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ids,
        "retmode": "xml",
    })
    url = f"{EUTILS_BASE}/efetch.fcgi?{params}"
    with urllib.request.urlopen(url) as resp:
        root = ET.fromstring(resp.read())
    results = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el     = article.find(".//PMID")
        title_el    = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")

        pmid     = pmid_el.text     if pmid_el     is not None and pmid_el.text     else "unknown"
        title    = title_el.text    if title_el    is not None and title_el.text    else "No title"
        abstract = abstract_el.text if abstract_el is not None and abstract_el.text else ""

        # Skip articles with no abstract
        if abstract.strip():
            results.append({
                "pmid":       pmid,
                "title":      title,
                "abstract":   clean_text(abstract),
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
    time.sleep(0.4)
    return results
 
 
def chunks_from_pubmed(query, max_articles=20, chunk_size=500, overlap=100): #Full pipelinez: Query -> PMIDs -> Abstracts -> Chunks -> Chunk objects with metadata; returns list of article dicts and list of Chunk dataclass instances
    print(f"[PubMed] Searching: '{query}'") 
    pmids = search_pubmed(query, max_articles) #Step 1: Search for PMIDs matching the query; returns list of PMIDs as strings
    print(f"         {len(pmids)} PMIDs found")
    articles = fetch_abstracts(pmids) #Step 2: Fetch abstracts and metadata for the PMIDs; returns list of dicts with keys: pmid, title, abstract, source_url
    print(f"         {len(articles)} abstracts fetched")
 
    splitter = make_splitter(chunk_size, overlap) #Create text splitter with specified chunk size and overlap; this will be used to split abstracts into chunks while respecting semantic boundaries
    all_chunks = []
 
    for art in articles: #Iterate articles
        raw = splitter.split_text(art["abstract"]) #Split text of the abstract into chunks
        for i, text in enumerate(raw): #Convert each raw chunk into a Chunk object; attaching PMID, URL, chunk index, and title
            all_chunks.append(Chunk(
                text=text, source="pubmed",
                pmid=art["pmid"], title=art["title"],
                chunk_index=i, total_chunks=len(raw),
                source_url=art["source_url"],
            ))
 
    print(f"         {len(all_chunks)} chunks created\n") #Logging; gives visinility into pipeline scale
    return articles, all_chunks #Returns both the original article metadata (for potential future use) and the list of Chunk objects that will be used for embeddings and retrieval


# ---------------------------------------------------------------------------
# Source B: PDF
# ---------------------------------------------------------------------------
 
def chunks_from_pdf(pdf_path, chunk_size=500, overlap=100): #Turn pdf into clean structured chunk objects
    from langchain_community.document_loaders import PyPDFLoader #Loads PDF files and extracts texts; avoids loading the dependanct unless needed; useful for keeping startup time fast if user is only using PubMed or text sources
    print(f"[PDF] Loading: {pdf_path}") #Visibility into pipeline; shows which file is being processed
    loader = PyPDFLoader(pdf_path) #Load PDF; returns a lost of LangChain Documents
    documents = loader.load() 
    splitter = make_splitter(chunk_size, overlap) #Same splitter as PubMed; ensures consistent chunking strategy across sources; can be customized per source if needed in future extensions
    lc_chunks = splitter.split_documents(documents) #Takes each page and splits it into smaller chunks; preserves page-level metadata in the resulting chunks; returns a list of LangChain Documents with chunked text and metadata
    title = Path(pdf_path).stem #Title extraction
    chunks = [ #Build chunk objects
        Chunk(text=clean_text(doc.page_content), source="pdf", title=title, #Raw text of the chunk; comes from LangChain; cleaned to remove noise before embedding; source is "pdf" for later citation formating and filtering; title is derived from filename for citations
              chunk_index=i, total_chunks=len(lc_chunks), #chunk_index = 1 is global index across entire PDF; total_chunks is total number of chunks from the PDF; both are used for reconstructing order and providing context in UI (e.g., "Chunk 2 of 8")
              page=doc.metadata.get("page", -1)) #Extracts page numbers
        for i, doc in enumerate(lc_chunks)
    ] 
    print(f"       {len(chunks)} chunks created\n") #Confirms pipeline success; helps detect empty PDFs or parsing failures
    return chunks #Returns chunk-level data
 

# ---------------------------------------------------------------------------
# Source C: Plain text
# ---------------------------------------------------------------------------
 
def chunks_from_txt(txt_path, chunk_size=500, overlap=100): #Take .txt file, clean, split, wrap into chunk objects
    print(f"[TXT] Loading: {txt_path}") #Helps trace what file is being processed; useful when batching multiple files; confirms file is found and read successfully
    text = clean_text(Path(txt_path).read_text()) #Reads entire file into memory as a string; cleans it to normalize whitespace/formatting
    splitter = make_splitter(chunk_size, overlap) #Same logic as other sources; keeps chunking behavior consistent across the system
    raw = splitter.split_text(text) #Split text
    title = Path(txt_path).stem #Title extraction
    chunks = [ #Build chunk objects
        Chunk(text=t, source="txt", title=title, chunk_index=i, total_chunks=len(raw)) #text = t already cleaned and chunked; source = "txt" distinguises from pdf and pubmed; title from filename; chunk_index = i is the position of the chunk in the document, total_chunks is the total number of chunks from this txt file; page is not applicable for txt, so we leave it at the default -1
        for i, t in enumerate(raw)
    ]
    print(f"       {len(chunks)} chunks created\n") #Confirms pipeline success; helps detect empty files or parsing issues
    return chunks
 
# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
 
def save_chunks(chunks, path="processed_chunks.pkl"): #Saves your chunk objects to disk
    with open(path, "wb") as f: #Open file in binary write mode; "w" for write, "b" for binary; creates the file if it doesnt exist, overwrites it if it does
        pickle.dump(chunks, f) #Converts Python objects into a byte stream and saves entire list of chunk objects to disk; pickle is a simple way to persist complex Python data structures without needing a full database; good for prototyping and small datasets; not recommended for production or large-scale data due to security and performance concerns
    print(f"[Saved] {len(chunks)} chunks → {path}") #Confirms success and helps you verify dataset size
 
 
def load_chunks(path="processed_chunks.pkl"): #Load saved chunks instead of recomputing; useful for iterative development and testing; avoids redundant API calls and processing; returns list of Chunk objects
    with open(path, "rb") as f: #Open file in binary read mode; "r" for read, "b" for binary; ensures we read the exact byte stream that was written by pickle
        chunks = pickle.load(f) #COnverts byte stream to original Python objects
    print(f"[Loaded] {len(chunks)} chunks from {path}") #Confirms success and helps you verify dataset size; ensures you know where your data is coming from
    return chunks
 

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
 
def build_corpus( #Combine all daa sources into 1 list of chunk objects
    query="", pdf_paths=None, txt_paths=None, #Query: triggers PubMed ingestion (if provided) or skps Pubmed entirely (if empty); pdf_paths, txt_paths: lists of file paths to ingest; if None, that source is skipped; allows flexible combinations of sources
    max_articles=20, chunk_size=500, overlap=100, #Max_articles limits the number of PubMed results to prevent huge API pulls; chunk_size and overlap configure the text splitter for all sources to ensure consistent chunking strategy
    cache_path="processed_chunks.pkl", use_cache=True, #Cache_path: Where chunks are saved/loaded; use_cache: Controls whether to reuse saved data
):
    if use_cache and Path(cache_path).exists(): #If cache exists and use_cache is true, skip everything. Immediately reutn saved chunks. Avoids API calls, PDF parsing, chunking. makes reruns instant
        return load_chunks(cache_path)
 
    all_chunks = []#Initialize storage. Master list for everything
    if query: #PubMed ingestion; only runs if query is not empy; ignores article level output (_); keeps only chunks; allows user to skip PubMed entirely by leaving query blank
        _, pubmed_chunks = chunks_from_pubmed(query, max_articles, chunk_size, overlap)
        all_chunks.extend(pubmed_chunks)
    for path in (pdf_paths or []): #PDF ingestion: Prevents crash if None
        all_chunks.extend(chunks_from_pdf(path, chunk_size, overlap))
    for path in (txt_paths or []): #TXT ingestion: Prevents crash if None
        all_chunks.extend(chunks_from_txt(path, chunk_size, overlap))
 
    save_chunks(all_chunks, cache_path) #Save everything; stores full corpus; enables fast relod next time
    return all_chunks
 
# ---------------------------------------------------------------------------
# Validation (no network needed)
# ---------------------------------------------------------------------------
 
if __name__ == "__main__": #Runs this block only when you execute script directly; wont run if imported elsewhere; good for testing and validation
    sample = clean_text(""" 
        Insulin resistance is a pathological condition in which cells fail to respond
        normally to the hormone insulin. When the body produces insulin under conditions
        of insulin resistance, the cells are resistant to the insulin and are unable to
        use it as effectively, leading to high blood sugar. Beta cells in the pancreas
        subsequently increase their production of insulin, further contributing to a high
        blood insulin level. This often remains undetected and can contribute to a
        diagnosis of prediabetes and type 2 diabetes. Insulin resistance is associated
        with various conditions such as obesity, hypertension, dyslipidemia,
        cardiovascular disease, non-alcoholic fatty liver disease, and polycystic ovary
        syndrome.
    """) #Defined sample text and pass it through clean_text() 
    splitter = make_splitter(chunk_size=500, overlap=100) #Same config as main pipleine
    raw = splitter.split_text(sample) #Splits text and outputs list of strings (chunks)
    chunks = [ #Simulating PubMed chunks without API calls; attaching realistic metadata
        Chunk(text=t, source="pubmed", pmid="00000001",
              title="Insulin Resistance — local test",
              chunk_index=i, total_chunks=len(raw))
        for i, t in enumerate(raw) 
    ]
 
    print(f"Chunks created: {len(chunks)}") #Sanity check; If this prints 0 something went wrong with the splitter or the sample text; should print 2-3 chunks for this sample
    for c in chunks: #Inspect each chunk
        print(f"\n  {c.citation()} [{c.chunk_index+1}/{c.total_chunks}]") #Print citation and position
        print(f"  '{c.text[:200]}...'")  #Print preview of text; first 200 characters; confirms text is present and cleaned; helps you visually verify the chunking strategy; should show coherent sentences without weird breaks or excessive whitespace
 