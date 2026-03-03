import json
from rag_query import build_or_refresh_index, retrieve, call_llm


def main():
    print("Genomic-RAG (RARS1) query engine\n")
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