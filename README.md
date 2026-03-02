# Genomic-RAG (RARS1)

A Retrieval-Augmented Generation (RAG) system that dynamically queries PubMed for the latest RARS1 abstracts, indexes them in a vector database (ChromaDB), and uses a Large Language Model (LLM) to extract variants, phenotypes, and associated diseases with PMID/DOI citations.

---

## PROJECT OVERVIEW

### A) Ingest
The system fetches the latest PubMed abstracts for the query "RARS1" using the NCBI Entrez API (Biopython).  
Results are sorted by publication date and limited to the most recent records.

### B) Index
Abstracts are safely chunked and embedded using a local SentenceTransformer model (`all-MiniLM-L6-v2`).  
Embeddings are stored persistently in ChromaDB.

### C) Query
When a user asks a question, the system:
1. Retrieves the most relevant evidence snippets from ChromaDB.
2. Builds a structured context using PMID and DOI metadata.
3. Sends the context to the LLM.
4. Returns structured JSON output.

### D) Guardrails
Two guardrail layers are implemented:

- Guardrail v1: Ensures every answer item cites a PMID/DOI that exists in retrieved evidence.
- Guardrail v2: Verifies that each extracted claim text appears inside the cited evidence text (reduces hallucinations).

Trick questions (e.g., unrelated diseases) return no unsupported answers.

---

## REPOSITORY STRUCTURE

- `main.py` — Entry point for interactive querying  
- `ingest.py` — Fetches PubMed abstracts and writes `data/pubmed_raw.jsonl`  
- `rag_query.py` — Retrieval + LLM structured extraction + guardrails  
- `evaluate.py` — Evaluation runner that outputs `data/eval_results.json`  
- `requirements.txt` — Dependencies  
- `data/` — Generated artifacts (`pubmed_raw.jsonl`, `eval_results.json`)  
- `chroma_db/` — Persistent Chroma vector store (generated)  

---

## SETUP

### 1) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Set your OpenAI API key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_key_here
```

---

## RUN THE PIPELINE

### A) Ingest latest PubMed abstracts

```bash
python ingest.py
```

Output:
- `data/pubmed_raw.jsonl`

---

### B) Ask questions interactively

```bash
python main.py
```

You will be prompted to enter a question and receive structured JSON output with citations.

---

### C) Run evaluation

```bash
python evaluate.py
```

Output:
- `data/eval_results.json`

---

## NCBI ENTIREZ RATE LIMIT HANDLING

This demo keeps API usage lightweight by:
- Limiting fetched records (`retmax=30`)
- Using retry with exponential backoff via `tenacity`

For stricter compliance, a small delay between API calls can be added.

---

## WHY THIS EMBEDDING MODEL?

This project uses the local SentenceTransformer model:

`all-MiniLM-L6-v2`

Reasons:
- Lightweight and fast
- Good semantic retrieval performance for scientific abstracts
- Avoids external embedding API costs

---

## ENSURING PHENOTYPE VS VARIANT CORRECTNESS

- The LLM outputs structured JSON with `type: phenotype | disease | variant`.
- Variants must appear exactly in the evidence text.
- Guardrail v2 verifies each extracted claim exists in cited evidence text.
- Trick questions return no unsupported answers.

---

## LIMITATIONS

- Results may change as new PubMed papers are published.
- The system relies on abstracts (not full text), so some details may be missing.