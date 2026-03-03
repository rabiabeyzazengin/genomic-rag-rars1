import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

from rag_query import (
    build_or_refresh_index,
    retrieve,
    call_llm,
)

OUTPUT_PATH = "data/eval_results.json"


# -----------------------------
# Guardrail helpers (Evaluation için)
# -----------------------------

def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _hit_citation_sets(hits: List[Dict[str, Any]]) -> Tuple[set, set]:
    hit_pmids = {str(h.get("pmid")).strip() for h in hits if h.get("pmid")}
    hit_dois = {str(h.get("doi")).strip().lower() for h in hits if h.get("doi")}
    return hit_pmids, hit_dois


def guardrail_v1_citations_only(raw_out: Dict[str, Any], hits: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    v1: answer evidence içindeki pmid/doi retrieved hits'te var mı?
    """
    answers = raw_out.get("answers", [])
    if not isinstance(answers, list):
        return [], [{"reason": "answers_not_list", "item": answers}]

    hit_pmids, hit_dois = _hit_citation_sets(hits)

    kept, removed = [], []
    for a in answers:
        if not isinstance(a, dict):
            removed.append({"reason": "answer_not_dict", "item": a})
            continue

        ev_list = a.get("evidence", [])
        if not isinstance(ev_list, list):
            ev_list = []

        ok = False
        for ev in ev_list:
            if not isinstance(ev, dict):
                continue
            pmid = str(ev.get("pmid") or "").strip()
            doi = str(ev.get("doi") or "").strip().lower()

            if pmid and pmid in hit_pmids:
                ok = True
                break
            if doi and doi in hit_dois:
                ok = True
                break

        if ok:
            kept.append(a)
        else:
            removed.append({"reason": "missing_citation_in_hits", "item": a})

    return kept, removed


def guardrail_v2_claim_in_evidence(answers: List[Dict[str, Any]], hits: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    v2: claim text (a['text']) evidence metninde geçiyor mu? (basit containment)
    """
    evidence_blob = "\n".join([_norm(h.get("text", "")) for h in hits if isinstance(h, dict)])

    kept, removed = [], []
    for a in answers:
        if not isinstance(a, dict):
            removed.append({"reason": "answer_not_dict_v2", "item": a})
            continue

        claim = _norm(a.get("text", ""))
        if not claim:
            removed.append({"reason": "empty_claim_text", "item": a})
            continue
        if len(claim) < 4:
            removed.append({"reason": "claim_too_short", "item": a})
            continue

        if claim in evidence_blob:
            kept.append(a)
        else:
            removed.append({"reason": "claim_not_found_in_evidence", "item": a})

    return kept, removed


def apply_guardrails(raw_out: Dict[str, Any], hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    v1 + v2 uygular ve çıktı formatını normalize eder.
    """
    out = dict(raw_out or {})
    removed_all: List[Dict[str, Any]] = []

    # v1: citation check
    v1_kept, v1_removed = guardrail_v1_citations_only(out, hits)
    removed_all.extend(v1_removed)

    # v2: claim check
    v2_kept, v2_removed = guardrail_v2_claim_in_evidence(v1_kept, hits)
    removed_all.extend(v2_removed)

    out["answers"] = v2_kept
    out["guardrail_removed"] = removed_all

    if "limitations" not in out or not isinstance(out["limitations"], list):
        out["limitations"] = []

    if len(hits) == 0:
        out["limitations"].append("No evidence retrieved (relevance gate).")

    return out


# -----------------------------
# Eval core
# -----------------------------

def run_one_test(collection: Any, question: str, top_k: int = 10) -> Dict[str, Any]:
    hits = retrieve(collection, question, top_k=top_k)

    raw = call_llm(question, hits)

    # LLM error ise guardrail çalıştırmadan normalize dön
    if isinstance(raw, dict) and raw.get("error"):
        guarded = {
            "gene": "RARS1",
            "answers": [],
            "limitations": [f"LLM error: {raw.get('error')}"],
            "guardrail_removed": [],
        }
    else:
        guarded = apply_guardrails(raw, hits)

    return {
        "question": question,
        "retrieval": {
            "top_k": top_k,
            "num_hits": len(hits),
            "hit_pmids": sorted(list({h.get("pmid") for h in hits if h.get("pmid")})),
            "distances": [h.get("distance") for h in hits if "distance" in h],
            "max_distance_threshold": float(os.getenv("RETRIEVE_MAX_DISTANCE", "1.0")),
        },
        "raw_output": raw,
        "guarded_output": guarded,
    }


def judge_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keyword ile trick tespit etmiyoruz.
    Sadece davranışa bakıyoruz:

    PASS:
      - answers var ve guardrail sonrası kaldı
      - veya answers boş ama limitations "no evidence" gibi sinyal veriyor (I don't know)

    PARTIAL_PASS:
      - LLM bir şey üretmiş ama guardrail silmiş (hallucination vardı ama engellendi)

    FAIL:
      - teknik hata (OPENAI_API_KEY, JSON parse vs) veya mantıksız durumlar
    """
    raw = result.get("raw_output", {}) or {}
    guarded = result.get("guarded_output", {}) or {}

    if isinstance(raw, dict) and raw.get("error"):
        return {"status": "FAIL", "reason": f"LLM error: {raw.get('error')}"}

    answers = guarded.get("answers", [])
    removed = guarded.get("guardrail_removed", [])
    limitations = guarded.get("limitations", [])

    if isinstance(answers, list) and len(answers) > 0:
        return {"status": "PASS", "reason": "Supported answers remained after guardrail."}

    # answers boş
    lim_text = " ".join([str(x).lower() for x in limitations]) if isinstance(limitations, list) else str(limitations).lower()
    if "no evidence" in lim_text or "no relevant evidence" in lim_text or "relevance gate" in lim_text:
        # bu, istenen "I don't know"
        if removed:
            return {"status": "PARTIAL_PASS", "reason": "No final answers; hallucinated items were removed by guardrail."}
        return {"status": "PASS", "reason": "No evidence -> returned empty answers (I don't know behavior)."}

    # limitations net değilse yine partial
    if removed:
        return {"status": "PARTIAL_PASS", "reason": "Answers removed by guardrail; review retrieval threshold and prompts."}

    return {"status": "PARTIAL_PASS", "reason": "No answers; may be strict threshold or sparse abstracts."}


def main():
    print("Şu an projede ne yapıyoruz? -> Evaluation çalıştırıp eval_results.json üretiyoruz.\n")

    os.makedirs("data", exist_ok=True)

    collection = build_or_refresh_index()

    tests: List[Dict[str, Any]] = [
        {"id": "t1_variants_symptoms", "question": "What are the most recently reported variants in RARS1 and their associated symptoms?"},
        {"id": "t2_phenotypes", "question": "List phenotypes (clinical symptoms) associated with RARS1 variants."},
        {"id": "t3_diseases", "question": "Which diseases are associated with RARS1?"},
        {"id": "t4_specific_variant", "question": "Is variant c.5A>G reported in RARS1, and what phenotypes are described?"},
        # Trick örneği (task PDF şartı)
        {"id": "t5_trick_unrelated", "question": "Does RARS1 cause breast cancer? Provide the evidence."},
    ]

    results = []
    for t in tests:
        print(f"[eval] Running {t['id']} ...")
        start = time.time()

        r = run_one_test(collection, t["question"], top_k=10)
        judge = judge_output(r)

        elapsed = round(time.time() - start, 2)
        guarded = r.get("guarded_output", {}) or {}

        results.append({
            "test_id": t["id"],
            "question": t["question"],
            "elapsed_sec": elapsed,
            "judge": judge,
            "retrieval": r["retrieval"],
            "guardrail_removed": guarded.get("guardrail_removed", []),
            "answers": guarded.get("answers", []),
            "limitations": guarded.get("limitations", []),
            "raw_output": r.get("raw_output"),
        })

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "collection": "rars1",
        "relevance_gate": {
            "RETRIEVE_MAX_DISTANCE": float(os.getenv("RETRIEVE_MAX_DISTANCE", "1.0")),
            "note": "If a question is unrelated, retrieval should yield 0 hits and the system returns empty answers.",
        },
        "notes": [
            "Deterministic retrieval (embeddings + Chroma), probabilistic generation (LLM).",
            "Guardrail v1: each answer must cite PMID/DOI present in retrieved hits.",
            "Guardrail v2: each claim text must appear in retrieved evidence text (simple containment).",
            "Unrelated questions should return empty answers (I don't know) rather than hallucinations.",
        ],
        "results": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ eval_results.json yazıldı: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()