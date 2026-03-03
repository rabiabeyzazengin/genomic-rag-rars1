import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# =========================
# Helpers
# =========================

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    fence = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
    m = fence.match(t)
    if m:
        return m.group(1).strip()
    t = re.sub(r"```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```", "", t).strip()
    return t


def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    s = text
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1].strip()
    return None


def _parse_llm_json(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if raw is None:
        return None, "empty LLM output"
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned), None
    except Exception:
        pass
    candidate = _extract_first_json_object(cleaned)
    if candidate:
        try:
            return json.loads(candidate), None
        except Exception as e:
            return None, f"json parse failed after extraction: {e}"
    return None, "LLM output did not contain valid JSON"


def _build_evidence_blocks(docs: List[Dict[str, Any]]) -> str:
    blocks = []
    for d in docs:
        pmid = _safe_str(d.get("pmid"))
        doi = _safe_str(d.get("doi"))
        text = _safe_str(d.get("text"))
        blocks.append(f"[PMID: {pmid} | DOI: {doi}]\n{text}\n")
    return "\n".join(blocks).strip()


def _chunk_abstract(text: str, max_chars: int = 900) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    sents = re.split(r"(?<=[.!?])\s+", t)
    chunks: List[str] = []
    cur = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = cur + " " + s
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def _is_definition_question(q: str) -> bool:
    t = (q or "").strip().lower()
    if not t:
        return False
    triggers = [
        "nedir", "ne demek", "what is", "what does", "gene function", "function of", "tanımla", "açıkla",
        "rars1 ne", "rars1 nedir", "rars1 ne demek", "rars1 explain"
    ]
    return any(x in t for x in triggers)


# -------------------------
# NEW: Question-target gate
# -------------------------

_STOPWORDS = {
    "rars1", "cause", "causes", "caused", "associated", "association", "with", "is", "are", "does", "do",
    "a", "an", "the", "of", "to", "and", "or", "provide", "evidence", "pubmed", "pmid", "doi",
    "what", "which", "list", "tell", "me", "about", "in", "on", "for", "related"
}

def _extract_target_terms(question: str) -> List[str]:
    """
    Target gate sadece spesifik "X ile Y ilişkili mi?" tarzı sorularda çalışmalı.
    (cause/associated with gibi pattern yoksa, [] döner ve gate devreye girmez.)
    """
    q = (question or "").strip().lower()
    if not q:
        return []

    # Sadece bu patternlerde hedef çıkar:
    m = re.search(r"(cause|causes|caused by|associated with|association with)\s+(.+?)\??$", q)
    if not m:
        return []

    target = m.group(2).strip()
    target = re.sub(r"^(a|an|the)\s+", "", target).strip()
    target = re.sub(r"\s+", " ", target).strip()

    # çok genel hedefleri sayma
    if not target:
        return []
    if target in {"disease", "diseases", "phenotype", "phenotypes", "variant", "variants", "symptom", "symptoms"}:
        return []

    terms = [target]
    toks = [t for t in re.split(r"[^a-z0-9]+", target) if t and t not in _STOPWORDS]
    for t in toks:
        if t not in terms:
            terms.append(t)
    return terms[:5]


def _evidence_mentions_terms(evidence_docs: List[Dict[str, Any]], terms: List[str]) -> bool:
    """
    terms boşsa: gate uygulama (normal RARS1 soruları)
    terms doluysa: evidence'da en az bir term geçmeli.
    """
    if not terms:
        return True

    blob = " ".join(
        ((d.get("title") or "") + " " + (d.get("text") or ""))
        for d in evidence_docs
        if isinstance(d, dict)
    ).lower()

    return any(t in blob for t in terms)


# =========================
# Core: index + query
# =========================

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rars1"
PUBMED_JSONL = "data/pubmed_raw.jsonl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# NEW: embedder cache (hız)
_EMBEDDER: Optional[SentenceTransformer] = None

def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


def build_or_load_collection() -> Any:
    _ensure_dir(CHROMA_DIR)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name=COLLECTION_NAME)


def maybe_index_pubmed(collection: Any) -> None:
    try:
        count = collection.count()
    except Exception:
        count = 0

    if count and count > 0:
        print(f"[index] Collection dolu görünüyor (count={count}). Yeniden indexlemiyorum.")
        return

    rows = _read_jsonl(PUBMED_JSONL)
    if not rows:
        print(f"[index] Uyarı: {PUBMED_JSONL} boş veya yok.")
        return

    print(f"[index] Embedding modeli yükleniyor: {EMBED_MODEL}")
    model = _get_embedder()

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for r in rows:
        pmid = _safe_str(r.get("pmid"))
        doi = _safe_str(r.get("doi"))
        title = _safe_str(r.get("title"))
        abstract = _safe_str(r.get("abstract"))

        chunks = _chunk_abstract(abstract)
        for i, ch in enumerate(chunks):
            cid = f"{pmid}:{i}"
            ids.append(cid)
            docs.append(ch)
            metas.append({"pmid": pmid, "doi": doi, "title": title})

    if not docs:
        print("[index] Uyarı: chunk üretilemedi (abstract boş olabilir).")
        return

    embs = model.encode(docs, show_progress_bar=True).tolist()
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print(f"[index] Index tamam: added={len(ids)}")


def retrieve(collection: Any, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Relevance-gated retrieval:
    - Chroma her zaman top_k döndürür; alakasız sorularda da "en yakın" sonuçlar gelir.
    - Distance threshold ile alakasızları eliyoruz.
    """
    model = _get_embedder()
    q_emb = model.encode([query]).tolist()[0]

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    # düşük distance = daha yakın (pratik)
    max_dist = float(os.getenv("RETRIEVE_MAX_DISTANCE", "1.0"))  # DEFAULT daha katı!

    out: List[Dict[str, Any]] = []
    for d, m, dist in zip(docs, metas, dists):
        if dist is None:
            continue
        if float(dist) > max_dist:
            continue

        out.append({
            "pmid": m.get("pmid"),
            "doi": m.get("doi"),
            "title": m.get("title"),
            "text": d,
            "distance": float(dist),
        })

    return out


def call_llm(question: str, evidence_docs: List[Dict[str, Any]], model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    IMPORTANT:
    - Kanıt yoksa LLM çağırma -> direkt "bilmiyorum" JSON dön.
    - Soru spesifik bir hedef (hastalık/konu) içeriyorsa ama evidence'da o hedef yoksa -> yine "bilmiyorum".
      (Keyword listesi değil; hedefi sorudan çıkarıyoruz.)
    """
    if not evidence_docs:
        return {
            "gene": "RARS1",
            "answers": [],
            "limitations": [
                "No relevant evidence retrieved from the indexed abstracts for this question."
            ]
        }

    # NEW: target gate
    target_terms = _extract_target_terms(question)
    if target_terms and not _evidence_mentions_terms(evidence_docs, target_terms):
        return {
            "gene": "RARS1",
            "answers": [],
            "limitations": [
                f"No evidence mentioning the target topic/condition in retrieved abstracts: {target_terms}"
            ]
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY missing in environment"}

    client = OpenAI(api_key=api_key)
    evidence = _build_evidence_blocks(evidence_docs)

    is_def = _is_definition_question(question)

    if is_def:
        schema = """{
  "gene": "RARS1",
  "answers": [
    {
      "type": "gene_summary",
      "text": "short definition of RARS1 (what it encodes / role)",
      "notes": "1-2 sentence evidence-grounded explanation",
      "evidence": [{"pmid":"...","doi":"..."}]
    }
  ],
  "limitations": ["optional notes"]
}"""
        task = (
            "Task: The user asks for a definition/overview. "
            "Extract a short gene summary grounded ONLY in the evidence provided. "
            "If evidence does not clearly define the gene, return answers as an empty list."
        )
    else:
        schema = """{
  "gene": "RARS1",
  "answers": [
    {
      "type": "phenotype | disease | variant",
      "text": "extracted item",
      "notes": "short evidence-grounded explanation",
      "evidence": [{"pmid":"...","doi":"..."}]
    }
  ],
  "limitations": ["optional notes"]
}"""
        task = (
            "Task: Extract clinically relevant phenotypes, diseases, and variants related to RARS1 "
            "grounded ONLY in the evidence provided. "
            "If there is no evidence, return answers as an empty list."
        )

    system = (
        "You are a biomedical information extraction assistant.\n"
        "Return ONLY valid JSON (no markdown, no code fences).\n"
        "Do NOT invent facts. Each item must cite PMID/DOI from the evidence blocks.\n"
    )

    user = f"""
Question: {question}

Evidence:
{evidence}

{task}

Output JSON schema:
{schema}
""".strip()

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content or ""
    obj, err = _parse_llm_json(raw)
    if err:
        return {"error": "LLM returned non-JSON output", "raw": raw}
    return obj


def build_or_refresh_index() -> Any:
    collection = build_or_load_collection()
    maybe_index_pubmed(collection)
    return collection


def main():
    collection = build_or_refresh_index()

    while True:
        q = input("Soru: ").strip()
        if not q:
            break

        evid = retrieve(collection, q, top_k=10)
        print(f"\n[retrieve] Bulunan kanıt parçası: {len(evid)}\n")

        out = call_llm(q, evid)
        print("\n--- OUTPUT (JSON) ---")
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()