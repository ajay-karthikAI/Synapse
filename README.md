# Synapse: A Medical Retrieval-Augmented Generation System

Synapse is a hybrid retrieval + LLM-powered system designed to answer medical questions with **relevant, trustworthy, and patient-friendly explanations**. It combines keyword search, semantic search, and LLM-based reranking to surface the most useful medical information—helping users better understand their conditions and prepare for conversations with healthcare providers.

---

## 🚀 Features

* **Hybrid Retrieval**

  * Combines BM25 (keyword search) + vector similarity (semantic search)
  * Captures both exact matches (e.g., drug names, dosages) and conceptual meaning

* **LLM Reranking**

  * Scores retrieved passages based on:

    * Relevance to the patient’s question
    * Clarity and usefulness
  * Ensures the most helpful context is prioritized

* **Chunked Medical Knowledge Base**

  * Processes and splits long medical texts into structured chunks
  * Supports sources like PubMed-style abstracts and clinical explanations

* **Patient-Centered Responses**

  * Designed for real-world usability (waiting room use case)
  * Focuses on clarity over jargon

* **Offline + Online Modes**

  * FAISS-based vector search works without API access
  * Optional OpenAI API enhances semantic understanding and reranking

---

## 🏗️ System Architecture

```text
User Query
    ↓
Hybrid Retriever
   ├── BM25 (keyword match)
   └── Vector Search (FAISS embeddings)
    ↓
Merged Candidates
    ↓
LLM Reranker
    ↓
Top-K Relevant Chunks
    ↓
Final Answer Generation
```

---

## 📂 Project Structure

```bash
medrag/
│
├── retriever/
│   ├── bm25.py              # Keyword-based retrieval
│   ├── vector_store.py     # FAISS vector search
│   └── hybrid.py           # Combines BM25 + vector results
│
├── processing/
│   ├── clean.py            # Text cleaning utilities
│   └── chunk.py            # Text chunking logic
│
├── reranker/
│   └── llm_reranker.py     # LLM scoring + ranking
│
├── data/
│   └── sample_chunks.json  # Example dataset
│
├── main.py                 # Entry point / pipeline execution
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/medrag.git
cd medrag
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Set your OpenAI API key (optional but recommended):

```bash
export OPENAI_API_KEY="your_api_key_here"
```

If no API key is provided, the system will run in **offline mode** using only local retrieval.

---

## 🧪 Usage

### Run the pipeline

```bash
python main.py
```

### Example Query

```text
"What causes insulin resistance?"
```

### Example Output

* Ranked relevant medical passages
* Clear explanation of the condition
* Context suitable for patient understanding

---

## 🧠 Key Concepts

### Hybrid Retrieval

* **BM25** excels at:

  * Exact matches (e.g., "Metformin 500mg")
* **Vector Search** excels at:

  * Semantic meaning (e.g., "high blood sugar causes")

Together → higher recall + precision

---

### LLM Reranking

Each passage is scored from **0–10** based on:

* Relevance (most important)
* Clarity
* Patient usefulness

This ensures the final answer is grounded in the *best* available evidence.

---

## 📊 Example Use Cases

* Patient symptom education
* Pre-appointment preparation
* Medical chatbot backend
* Clinical knowledge retrieval
* Health-tech startup MVP (e.g., symptom checker)

---

## ⚠️ Disclaimer

Synapse is for **educational purposes only** and is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 🔮 Future Improvements

* Structured medical ontologies (e.g., SNOMED, ICD)
* Personalization based on patient history
* Source attribution + citation highlighting
* UI dashboard for real-time querying
* Integration with hospital systems (e.g., EHRs)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.

---

## 📜 License

MIT License

---

## 💡 Inspiration

Built to bridge the gap between **dense medical literature** and **real patient understanding**—bringing clarity to complex health questions.

---

## 👨‍💻 Author

Ajay Karthikeyan
Medical AI | Data Science | Health Tech

---
