import os
import json
import re
from typing import List, Optional, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Şu an projede ne yapıyoruz?
# Retrieval (Chroma + embedding) + LLM (OpenAI) ile kaynaklı (PMID/DOI) çıktı üretiyoruz.
# -----------------------------

load_dotenv()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rars1_collection"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = OpenAI()


def chunk_text_safely(text: str, target_chars: int = 1200) -> List[str]:
    """
    Şu an projede ne yapıyoruz?
    Metni 'chunk'lara bölüyoruz. Variant ifadeleri bölünmesin diye cümle bazlı ilerliyoruz.

    Basit ama güvenli yaklaşım:
    - Cümlelere böl
    - Cümleleri target_chars civarı biriktir
    - Variant içeren cümleyi ortadan kesme
    """
    if not text:
        return []

    # Basit cümle bölme
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    buf = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # buffer doluysa ve ekleme aşırı büyütecekse yeni chunk başlat
        if buf and (len(buf) + 1 + len(s) > target_chars):
            chunks.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip()

    if buf:
        chunks.append(buf.strip())

    return chunks


def get_chroma_collection() -> chromadb.api.models.Collection.Collection:
    """
    Şu an projede ne yapıyoruz?
    ChromaDB'yi disk üzerinde kalıcı (persistent) kullanıyoruz.
    Böylece her çalıştırmada embedding'i yeniden üretmek zorunda kalmıyoruz.
    """
    chroma = chromadb.PersistentClient(path=PERSIST_DIR)

    # collection varsa al, yoksa oluştur
    try:
        return chroma.get_collection(name=COLLECTION_NAME)
    except Exception:
        return chroma.create_collection(name=COLLECTION_NAME)


def build_or_refresh_index(jsonl_path: str = "data/pubmed_raw.jsonl") -> None:
    """
    Şu an projede ne yapıyoruz?
    PubMed'den çektiğimiz abstract'ları:
    - chunk'lıyoruz
    - embedding üretiyoruz
    - Chroma'ya kaydediyoruz
    """
    collection = get_chroma_collection()

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"{jsonl_path} bulunamadı. Önce ingest.py çalıştır.")

    # Zaten veri var mı? (küçük projede hızlı check)
    existing_count = collection.count()
    if existing_count > 0:
        print(f"[index] Collection dolu görünüyor (count={existing_count}). Yeniden indexlemiyorum.")
        return

    print("[index] Collection boş. Index oluşturuyorum...")

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    embs: List[List[float]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            art = json.loads(line)
            pmid = str(art.get("pmid", "")).strip()
            title = (art.get("title") or "").strip()
            abstract = (art.get("abstract") or "").strip()
            doi = art.get("doi")

            if not pmid or not abstract:
                continue

            chunks = chunk_text_safely(abstract)

            for i, ch in enumerate(chunks):
                chunk_id = f"{pmid}_{i}"
                ids.append(chunk_id)
                docs.append(ch)
                metas.append(
                    {
                        "pmid": pmid,
                        "doi": doi,
                        "title": title,
                    }
                )

    # Embeddingleri batch üretelim
    if docs:
        vecs = embedding_model.encode(docs, batch_size=32, show_progress_bar=True)
        embs = [v.tolist() for v in vecs]

        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    print(f"[index] Bitti. Eklenen chunk sayısı: {len(docs)}")


def retrieve(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Şu an projede ne yapıyoruz?
    Kullanıcı sorusuna en yakın metin parçalarını (chunk) buluyoruz.
    """
    collection = get_chroma_collection()

    q_emb = embedding_model.encode([query])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])

    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(
            {
                "text": doc,
                "pmid": meta.get("pmid"),
                "doi": meta.get("doi"),
                "title": meta.get("title"),
                "distance": dist,
            }
        )
    return hits


def build_context(hits: List[Dict[str, Any]]) -> str:
    """
    Şu an projede ne yapıyoruz?
    LLM'e 'sadece bu kanıtlara dayan' diyebilmek için bağlamı (context) hazırlıyoruz.
    """
    blocks = []
    for i, h in enumerate(hits, start=1):
        cite = f"PMID:{h.get('pmid')}"
        if h.get("doi"):
            cite += f" DOI:{h.get('doi')}"
        title = h.get("title") or ""
        blocks.append(
            f"[EVIDENCE {i}] {cite}\nTITLE: {title}\nTEXT: {h.get('text')}\n"
        )
    return "\n".join(blocks)


def ask_llm_structured(question: str, context: str) -> Dict[str, Any]:
    """
    Şu an projede ne yapıyoruz?
    LLM'den 'JSON' formatında, kanıta bağlı cevap istiyoruz.
    Her madde için PMID/DOI zorunlu.
    """
    system = (
        "You are a biomedical information extraction assistant. "
        "You MUST answer only using the provided EVIDENCE blocks. "
        "Every clinical claim MUST include supporting PMID or DOI from the evidence. "
        "If you cannot find evidence, say so explicitly and do NOT guess."
    )

    user = f"""
QUESTION:
{question}

EVIDENCE:
{context}

Return ONLY valid JSON with this schema:
{{
  "gene": "RARS1",
  "answers": [
    {{
      "type": "phenotype|disease|variant",
      "text": "the extracted item (phenotype name OR disease name OR variant string like c.2T>C)",
      "notes": "short explanation grounded in evidence",
      "evidence": [
        {{"pmid": "....", "doi": ".... or null"}}
      ]
    }}
  ],
  "limitations": ["..."]
}}

Rules:
- Only include items that are explicitly supported by the EVIDENCE.
- For variants: include ONLY if the exact variant string appears in EVIDENCE.
- evidence array must not be empty for any answer item.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content.strip()

    # JSON parse
    try:
        return json.loads(content)
    except Exception:
        # Eğer LLM nadiren JSON dışı dönerse, debug için ham içeriği sakla
        return {
            "error": "LLM returned non-JSON output",
            "raw": content
        }


def validate_evidence(output: Dict[str, Any], hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Şu an projede ne yapıyoruz?
    'Hallucination guardrail' için ilk kontrol:
    - Her item evidence taşıyor mu?
    - Evidence'daki pmid/doi bizim retrieval sonucunda var mı?
    """
    if "answers" not in output or not isinstance(output.get("answers"), list):
        return output

    allowed_pmids = set([h.get("pmid") for h in hits if h.get("pmid")])
    allowed_dois = set([h.get("doi") for h in hits if h.get("doi")])

    cleaned = []
    removed = []

    for item in output["answers"]:
        ev = item.get("evidence") or []
        if not ev:
            removed.append({"reason": "missing_evidence", "item": item})
            continue

        ok = False
        for e in ev:
            pmid_ok = e.get("pmid") in allowed_pmids if e.get("pmid") else False
            doi_ok = e.get("doi") in allowed_dois if e.get("doi") else False
            if pmid_ok or doi_ok:
                ok = True
                break

        if ok:
            cleaned.append(item)
        else:
            removed.append({"reason": "evidence_not_in_retrieved_context", "item": item})

    output["answers"] = cleaned
    if removed:
        output["guardrail_removed"] = removed

    return output


def main():
    print("Şu an projede ne yapıyoruz? -> RARS1 için RAG sorgu hattını çalıştırıyoruz.\n")

    # 1) Index var mı? Yoksa oluştur.
    build_or_refresh_index()

    # 2) Kullanıcıdan soru al
    question = input("Soru: ").strip()
    if not question:
        print("Boş soru olmaz.")
        return

    # 3) Retrieve
    hits = retrieve(question, top_k=10)
    print(f"\n[retrieve] Bulunan kanıt parçası: {len(hits)}")

    # 4) Context hazırla
    context = build_context(hits)

    # 5) LLM ile structured cevap
    out = ask_llm_structured(question, context)

    # 6) Guardrail doğrulama
    out = validate_evidence(out, hits)

    # 7) Yazdır
    print("\n--- OUTPUT (JSON) ---")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()