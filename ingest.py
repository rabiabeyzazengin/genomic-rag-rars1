import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
import json
import time
from typing import List
from Bio import Entrez
from tenacity import retry, stop_after_attempt, wait_exponential

# Entrez API email zorunlu
Entrez.email = "rabiabeyzazengin@gmail.com"  

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def search_pubmed(query: str, max_results: int = 30) -> List[str]:
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        sort="date",
        retmax=max_results,
    )
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def fetch_details(id_list: List[str]):
    ids = ",".join(id_list)
    handle = Entrez.efetch(
        db="pubmed",
        id=ids,
        rettype="abstract",
        retmode="xml",
    )
    records = Entrez.read(handle)
    handle.close()
    return records


def parse_records(records):
    parsed = []

    for article in records["PubmedArticle"]:
        medline = article["MedlineCitation"]
        article_data = medline["Article"]

        pmid = medline["PMID"]
        title = article_data.get("ArticleTitle", "")
        abstract = ""

        if "Abstract" in article_data:
            abstract_parts = article_data["Abstract"]["AbstractText"]
            abstract = " ".join(str(part) for part in abstract_parts)

        doi = None
        if "ELocationID" in article_data:
            for item in article_data["ELocationID"]:
                if item.attributes.get("EIdType") == "doi":
                    doi = str(item)

        parsed.append({
            "pmid": str(pmid),
            "title": title,
            "abstract": abstract,
            "doi": doi
        })

    return parsed


def main():
    query = "RARS1"
    print(f"Searching PubMed for: {query}")

    ids = search_pubmed(query, max_results=30)
    print(f"Found {len(ids)} articles")

    records = fetch_details(ids)
    articles = parse_records(records)

    with open("data/pubmed_raw.jsonl", "w") as f:
        for article in articles:
            f.write(json.dumps(article) + "\n")

    print("Saved to data/pubmed_raw.jsonl")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    main()