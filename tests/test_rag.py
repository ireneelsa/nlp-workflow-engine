import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.indexer import DocumentIndexer
from src.registry import build_default_registry

SAMPLE_DOC = """
Tesla Gigafactory India Report — 2026

Tesla Inc announced plans to build its first Indian Gigafactory in Chennai,
Tamil Nadu. The factory will cover 500 acres and produce 500,000 electric
vehicles annually. Construction begins in January 2027.

CEO Elon Musk stated the investment will exceed $3 billion. The Indian
government offered tax incentives worth $500 million over 10 years.
Prime Minister Narendra Modi called it the largest foreign investment
in India's automotive history.

The factory will create 15,000 direct jobs and 50,000 indirect jobs.
Workers will be recruited from Tamil Nadu, Andhra Pradesh, and Karnataka.
Training programs will begin in mid-2026 in partnership with IIT Madras.
"""

def test_indexing():
    indexer = DocumentIndexer()
    result = indexer.index_text(SAMPLE_DOC, "test_tesla_doc")
    assert result["chunks"] > 0
    assert result["collection_name"] == "test_tesla_doc"
    print(f"\n  Indexed {result['chunks']} chunks")

def test_rag_question():
    registry = build_default_registry()
    result = registry.run_tool("rag", {
        "question": "How many jobs will the factory create?",
        "collection_name": "test_tesla_doc"
    })
    assert result.error is None, f"RAG failed: {result.error}"
    assert "answer" in result.output
    assert len(result.output["answer"]) > 10
    print(f"\n  Question: How many jobs will the factory create?")
    print(f"  Answer: {result.output['answer']}")
    print(f"  Chunks used: {result.output['chunks_used']}")

def test_rag_specific_fact():
    registry = build_default_registry()
    result = registry.run_tool("rag", {
        "question": "What is the investment amount?",
        "collection_name": "test_tesla_doc"
    })
    assert result.error is None
    print(f"\n  Question: What is the investment amount?")
    print(f"  Answer: {result.output['answer']}")

def test_collections_list():
    indexer = DocumentIndexer()
    collections = indexer.list_collections()
    assert "test_tesla_doc" in collections
    print(f"\n  Collections: {collections}")


if __name__ == "__main__":
    print("Running RAG tests...\n")
    test_indexing();            print("PASS: indexing")
    test_rag_question();        print("PASS: RAG question answering")
    test_rag_specific_fact();   print("PASS: specific fact retrieval")
    test_collections_list();    print("PASS: collections list")
    print("\nAll RAG tests passed.")