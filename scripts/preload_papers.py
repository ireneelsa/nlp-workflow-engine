import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
import chromadb

PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/data/chroma_store")


def _collection_name_for(filename: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", filename.replace(".", "_"))
    name = re.sub(r"_+", "_", name)
    return name.strip("_")[:200]


def main():
    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logger.info("[preload] No PDF files found in papers/")
        return

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = {c.name for c in chroma.list_collections()}

    indexed = 0
    skipped = 0

    for filename in sorted(pdf_files):
        coll_name = _collection_name_for(filename)
        if coll_name in existing:
            logger.info(f"[preload] '{filename}' → '{coll_name}' already indexed, skipping")
            skipped += 1
            continue

        file_path = os.path.join(PAPERS_DIR, filename)
        try:
            from src.indexer import DocumentIndexer
            indexer = DocumentIndexer()
            result = indexer.index_file(file_path, coll_name)
            logger.info(
                f"[preload] Indexed '{filename}': {result['chunks']} chunks ({result['file_type']})"
            )
            indexed += 1
        except Exception as e:
            logger.error(f"[preload] Failed to index '{filename}': {e}")

    logger.info(f"[preload] Done — Preloaded {indexed} papers, skipped {skipped} already indexed")


if __name__ == "__main__":
    main()
