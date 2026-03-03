# Genomic-RAG (RARS1)

A Retrieval-Augmented Generation (RAG) system that dynamically queries PubMed for the latest RARS1 literature, indexes abstracts in a persistent vector database (ChromaDB), and uses a Large Language Model (LLM) to extract structured clinical information (variants, phenotypes, diseases) with strict citation guardrails.

---

# 1. Project Overview

This project implements a dynamic Genomic RAG pipeline focused on the **RARS1** gene.

The system:

A) Dynamically queries PubMed using the NCBI Entrez API (Biopython)  
B) Safely chunks medical abstracts without breaking variant names  
C) Stores embeddings in a persistent Chroma vector database  
D) Uses an LLM to extract structured genomic knowledge  
E) Applies hallucination guardrails to ensure citation correctness  

The goal is to ensure accuracy over speed — in medical domains, unsupported claims are rejected.

---

# 2. Technical Architecture

## A) Dynamic Data Ingestion

- Uses NCBI Entrez API via Biopython
- Searches for `"RARS1"`
- Sorts results by publication date
- Fetches the most recent abstracts (default: 30)
- Writes results to `data/pubmed_raw.jsonl`
- Implements retry with exponential backoff (`tenacity`)

This ensures the system does not rely on static or pre-downloaded PDFs.

---

## B) Knowledge Processing & Storage

### Safe Chunking Strategy

Abstracts are split at sentence boundaries to prevent genetic variant names (e.g., `c.2T>C (p.Met1Thr)`) from being split across chunks.

This avoids breaking clinically meaningful tokens.

### Vector Database

- Uses persistent ChromaDB (`chroma_db/`)
- Embeddings generated locally with:
  
  `all-MiniLM-L6-v2`

Reasons for choosing this model:
- Lightweight and fast
- Strong semantic retrieval for scientific abstracts
- No external embedding API cost
- Fully local inference

---

## C) LLM Extraction Layer

When a user submits a question:

1. Relevant chunks are retrieved from ChromaDB.
2. Evidence blocks are constructed including PMID and DOI metadata.
3. The LLM generates structured JSON output.
4. Each claim must include a valid citation.

Output JSON schema:

{
  "gene": "RARS1",
  "answers": [
    {
      "type": "phenotype | disease | variant",
      "text": "extracted item",
      "notes": "short evidence-grounded explanation",
      "evidence": [
        { "pmid": "...", "doi": "..." }
      ]
    }
  ],
  "limitations": ["optional notes"]
}

---

## D) Hallucination Guardrails

Two layers of validation are implemented:

### Guardrail v1
Ensures each extracted item cites a PMID/DOI that exists in the retrieved evidence.

### Guardrail v2
Ensures that the extracted claim text appears inside the cited evidence text.

If validation fails, the item is removed.

This reduces fabricated variants or unsupported clinical claims.

Trick questions (e.g., diseases unrelated to RARS1) return no unsupported outputs.

---

# 3. Repository Structure

- `main.py` — Entry point for interactive querying  
- `ingest.py` — Fetches PubMed abstracts  
- `rag_query.py` — Retrieval + structured extraction + guardrails  
- `evaluate.py` — Evaluation runner  
- `requirements.txt` — Dependencies  
- `data/` — Generated artifacts (`pubmed_raw.jsonl`, `eval_results.json`)  
- `chroma_db/` — Persistent Chroma vector store (generated locally)  

---

# 4. Setup

## A) Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## B) Install Dependencies

```bash
pip install -r requirements.txt
```

## C) Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_key_here
ENTREZ_EMAIL=your_email@example.com
```

The system reads these values securely at runtime.

---

# 5. Running the Pipeline

## A) Ingest Latest PubMed Data

```bash
python ingest.py
```

Output:
- `data/pubmed_raw.jsonl`

---

## B) Ask Questions

```bash
python main.py
```

You will be prompted to enter a query.
The system returns structured JSON with citations.

---

## C) Run Evaluation

```bash
python evaluate.py
```

Output:
- `data/eval_results.json`

This includes trick-question testing to demonstrate hallucination resistance.

---

# 6. NCBI API Rate Limit Handling

- Limits fetched records (`retmax=30`)
- Uses retry with exponential backoff (`tenacity`)
- Designed for lightweight academic/demo usage

A fixed delay between calls can be added for stricter compliance.

---

# 7. Accuracy Over Speed

Medical systems must prioritize correctness.

If evidence does not support a claim, the system returns no result rather than hallucinating information.

---

# 8. Limitations

- Uses abstracts only (not full text articles)
- Results evolve as new PubMed papers are published
- Strict guardrails may occasionally remove borderline-valid paraphrased claims

---

# 9. Evaluation Criteria Alignment

This implementation addresses:

- Clean API handling
- Smart chunking of medical text
- Persistent vector indexing
- Structured LLM outputs with citations
- Hallucination guardrails
- Modular and readable code

---

End of README.