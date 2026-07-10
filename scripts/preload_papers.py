import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
import chromadb
from src.paper_store import download_all_papers

PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/data/chroma_store")
RESEARCH_COLLECTION = "research_papers"


def _already_indexed(chroma, filename: str) -> bool:
    try:
        col = chroma.get_collection(name=RESEARCH_COLLECTION)
        result = col.get(where={"source": filename}, limit=1)
        return bool(result.get("ids"))
    except Exception:
        return False


def main():
    # 1. Pull any new PDFs from HF Dataset
    downloaded = download_all_papers(PAPERS_DIR)
    logger.info(f"[preload] Downloaded {len(downloaded)} new papers from HF Dataset")

    # 2. Index PDFs not already in research_papers
    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logger.info("[preload] No PDF files found in papers/")
        return

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    indexed = 0
    skipped = 0

    for filename in sorted(pdf_files):
        if _already_indexed(chroma, filename):
            logger.info(f"[preload] '{filename}' already in research_papers, skipping")
            skipped += 1
            continue

        file_path = os.path.join(PAPERS_DIR, filename)
        try:
            from src.indexer import DocumentIndexer
            indexer = DocumentIndexer()
            result = indexer.index_file_to_research(file_path, filename)
            logger.info(
                f"[preload] Indexed '{filename}': {result['chunks']} chunks ({result['file_type']})"
            )
            indexed += 1
        except Exception as e:
            logger.error(f"[preload] Failed to index '{filename}': {e}")

    logger.info(
        f"[preload] Done — downloaded {len(downloaded)}, "
        f"indexed {indexed}, skipped {skipped} already in research_papers"
    )


if __name__ == "__main__":
    main()
