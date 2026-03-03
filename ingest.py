import os
import json
import time
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

from Bio import Entrez

# Output
OUT_PATH = "data/pubmed_raw.jsonl"

# Controls
SEARCH_TERM = os.getenv("PUBMED_QUERY", "RARS1")
RETMAX = int(os.getenv("PUBMED_RETMAX", "30"))
SLEEP_SEC = float(os.getenv("PUBMED_SLEEP_SEC", "0.34"))

# NCBI requires an email
Entrez.email = os.getenv("ENTREZ_EMAIL", "example@example.com")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _fetch_pmids(term: str, retmax: int) -> List[str]:
    h = Entrez.esearch(db="pubmed", term=term, retmax=retmax, sort="pub+date")
    r = Entrez.read(h)
    return r.get("IdList", []) or []


def _fetch_details(pmids: List[str]) -> List[Dict[str, Any]]:
    if not pmids:
        return []

    # fetch in one shot (simple)
    h = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="xml")
    data = Entrez.read(h)

    out: List[Dict[str, Any]] = []
    articles = data.get("PubmedArticle", []) or []
    for a in articles:
        try:
            med = a["MedlineCitation"]
            art = med["Article"]

            pmid = str(med["PMID"])
            title = str(art.get("ArticleTitle", "") or "")

            # Abstract
            abs_text = ""
            if "Abstract" in art and "AbstractText" in art["Abstract"]:
                parts = art["Abstract"]["AbstractText"]
                if isinstance(parts, list):
                    abs_text = " ".join([str(x) for x in parts if x])
                else:
                    abs_text = str(parts)

            # DOI
            doi = ""
            try:
                ids = a.get("PubmedData", {}).get("ArticleIdList", [])
                for x in ids:
                    if getattr(x, "attributes", {}).get("IdType") == "doi":
                        doi = str(x)
                        break
            except Exception:
                doi = ""

            out.append({
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "abstract": abs_text
            })
        except Exception:
            continue

    return out


def main():
    _ensure_dir("data")

    pmids = _fetch_pmids(SEARCH_TERM, RETMAX)
    print(f"[ingest] term={SEARCH_TERM} retmax={RETMAX} -> pmids={len(pmids)}")

    rows = _fetch_details(pmids)

    # write jsonl
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ingest] wrote: {OUT_PATH} rows={len(rows)}")

    # polite
    time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()