import json
import time
from datetime import datetime
from typing import Dict, Any, List

# rag_query.py'den fonksiyonları kullanacağız
from rag_query import (
    build_or_refresh_index,
    retrieve,
    build_context,
    ask_llm_structured,
    validate_evidence,
)

# -----------------------------
# Şu an projede ne yapıyoruz?
# Otomatik test soruları çalıştırıp eval_results.json üretiyoruz.
# -----------------------------

OUTPUT_PATH = "data/eval_results.json"


def run_one_test(question: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Tek bir soruyu uçtan uca koşturur:
    - retrieve
    - context build
    - llm structured output
    - guardrail validation
    """
    hits = retrieve(question, top_k=top_k)
    context = build_context(hits)

    raw = ask_llm_structured(question, context)
    guarded = validate_evidence(raw, hits)

    return {
        "question": question,
        "retrieval": {
            "top_k": top_k,
            "num_hits": len(hits),
            "hit_pmids": list({h.get("pmid") for h in hits if h.get("pmid")}),
        },
        "raw_output": raw,
        "guarded_output": guarded,
    }


def is_trick_question(q: str) -> bool:
    """
    Trick questions:
    - açık alakasız hastalıklar / ilişkiler
    - guardrail v2 kanıt testi: 'including autism'
    """
    q_lower = q.lower()
    trick_keywords = [
        "not associated",
        "unrelated",
        "breast cancer",
        "diabetes",
        "alzheimer",
        "including autism",
        "autism",
    ]
    return any(k in q_lower for k in trick_keywords)


def simple_judge(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basit değerlendirme:

    Trick sorularda:
      - ideal: answers boş (uydurmadı)
      - answers doluysa ama guardrail_removed varsa PARTIAL_PASS (guardrail yakalamış)
      - answers dolu ve guardrail_removed yoksa FAIL

    Normal sorularda:
      - answers doluysa PASS
      - boşsa PARTIAL_PASS (veri az/çok katı guardrail)
    """
    q = result["question"]
    out = result.get("guarded_output", {})
    answers = out.get("answers", [])
    removed = out.get("guardrail_removed", [])
    limitations = out.get("limitations", [])

    if is_trick_question(q):
        # autism guardrail testi için özel ufak kontrol:
        # Eğer cevaplar doluysa bile "autism" geçmiyorsa bu kötü değil.
        autism_expected_absent = "autism" in q.lower()
        autism_found = any("autism" in (a.get("text", "").lower()) for a in answers)

        if len(answers) == 0:
            status = "PASS"
            reason = "Trick question: model returned no supported answers (no hallucination)."
        else:
            if autism_expected_absent and not autism_found:
                # autism'u sokmaya çalışmadıysa kabul edilebilir
                status = "PASS"
                reason = "Guardrail test: autism was not included in answers; model avoided unsupported claim."
            elif removed:
                status = "PARTIAL_PASS"
                reason = "Trick question: some hallucinated/unsupported items were removed by guardrail."
            else:
                # Ek güvenlik: limitations içinde 'no evidence' tarzı bir şey varsa PARTIAL_PASS diyelim
                lim_text = " ".join([str(x).lower() for x in limitations]) if isinstance(limitations, list) else str(limitations).lower()
                if "no evidence" in lim_text or "no supporting evidence" in lim_text or "no evidence found" in lim_text:
                    status = "PARTIAL_PASS"
                    reason = "Trick question: answers returned but limitations indicate lack of evidence; review manually."
                else:
                    status = "FAIL"
                    reason = "Trick question: model produced answers without being blocked; review guardrails and evidence."
        return {"status": status, "reason": reason}

    # Normal soru
    if len(answers) > 0:
        return {"status": "PASS", "reason": "Returned supported answers with citations."}
    return {
        "status": "PARTIAL_PASS",
        "reason": "No answers returned; may be due to limited retrieved abstracts or strict guardrails.",
    }


def main():
    print("Şu an projede ne yapıyoruz? -> Evaluation çalıştırıp eval_results.json üretiyoruz.\n")

    # Index hazır mı? (yoksa oluşturur)
    build_or_refresh_index()

    tests: List[Dict[str, Any]] = [
        {
            "id": "t1_variants_symptoms",
            "question": "What are the most recently reported variants in RARS1 and their associated symptoms?"
        },
        {
            "id": "t2_phenotypes",
            "question": "List phenotypes (clinical symptoms) associated with RARS1 variants."
        },
        {
            "id": "t3_diseases",
            "question": "Which diseases are associated with RARS1?"
        },
        {
            "id": "t4_specific_variant",
            "question": "Is variant c.5A>G reported in RARS1, and what phenotypes are described?"
        },
        # Trick question (görev kağıdının istediği)
        {
            "id": "t5_trick_unrelated",
            "question": "Does RARS1 cause breast cancer? Provide the evidence."
        },
        # Bir trick daha (opsiyonel ama iyi durur)
        {
            "id": "t6_trick_unrelated",
            "question": "Is RARS1 associated with Type 2 diabetes? Provide PubMed evidence."
        },
        # Guardrail v2 kanıt testi
        {
            "id": "t7_guardrail_v2_claim_check",
            "question": "List phenotypes (clinical symptoms) associated with RARS1 variants, including autism."
        }
    ]

    results = []
    for t in tests:
        print(f"[eval] Running {t['id']} ...")
        start = time.time()
        r = run_one_test(t["question"], top_k=10)
        judge = simple_judge(r)
        elapsed = round(time.time() - start, 2)

        results.append({
            "test_id": t["id"],
            "question": t["question"],
            "elapsed_sec": elapsed,
            "judge": judge,
            "retrieval": r["retrieval"],
            "guardrail_removed": r.get("guarded_output", {}).get("guardrail_removed", []),
            "answers": r.get("guarded_output", {}).get("answers", []),
            "limitations": r.get("guarded_output", {}).get("limitations", []),
            "raw_output": r.get("raw_output"),
        })

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "collection": "rars1_collection",
        "notes": [
            "This evaluation uses deterministic retrieval (local embeddings + Chroma) and probabilistic generation (LLM).",
            "Guardrail v1 checks that cited PMID/DOI exists in retrieved evidence.",
            "Guardrail v2 additionally verifies that each extracted claim (variant/phenotype/disease text) appears in the evidence text.",
            "Trick questions are expected to return no supported answers."
        ],
        "results": results
    }

    # Kaydet
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ eval_results.json yazıldı: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()