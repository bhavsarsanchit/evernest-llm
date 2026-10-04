"""Index listing PDFs in Chroma. Identity documents are never embedded."""

from __future__ import annotations

import csv

import chromadb
from pypdf import PdfReader

from evernest_assistant.config import BLOCKED_CATEGORIES, CHROMA_DIR, DOCS_DIR, MANIFEST

COLLECTION = "listings"


def _page_text(page) -> str:
    text = page.extract_text() or ""
    return " ".join(text.split())


def index_documents() -> dict:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.get_or_create_collection(COLLECTION)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    skipped: list[str] = []

    with MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["sub_category"] in BLOCKED_CATEGORIES:
                skipped.append(row["document_id"])
                continue
            pdf_path = DOCS_DIR.parent / row["file_path"]
            if not pdf_path.exists():
                pdf_path = DOCS_DIR / row["listing_id"] / row["file_name"]
            reader = PdfReader(str(pdf_path))
            for number, page in enumerate(reader.pages, start=1):
                text = _page_text(page)
                if len(text) < 40:
                    skipped.append(f"{row['document_id']}#p{number}:no-text")
                    continue
                ids.append(f"{row['document_id']}-p{number}")
                documents.append(text)
                metadatas.append(
                    {
                        "document_id": row["document_id"],
                        "listing_id": row["listing_id"],
                        "page": number,
                        "sub_category": row["sub_category"],
                    }
                )

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return {"indexed_pages": len(ids), "skipped": skipped}


def collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION)


def search(question: str, listing_id: str, top_k: int) -> list[dict]:
    coll = collection()
    if coll.count() == 0:
        return []
    owned = coll.get(where={"listing_id": listing_id})
    available = len(owned.get("ids") or [])
    if available == 0:
        return []
    n = min(top_k, available)
    found = coll.query(
        query_texts=[question],
        n_results=n,
        where={"listing_id": listing_id},
    )
    rows = []
    docs = found.get("documents") or [[]]
    metas = found.get("metadatas") or [[]]
    for text, meta in zip(docs[0], metas[0]):
        rows.append(
            {
                "text": text,
                "document_id": meta.get("document_id", ""),
                "listing_id": meta.get("listing_id", ""),
                "page": meta.get("page"),
                "sub_category": meta.get("sub_category", ""),
            }
        )
    return rows
