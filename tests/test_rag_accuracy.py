import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.indexer import DocumentIndexer
from src.tools.rag import RAGTool

RAG_TEST_CASES = [
    {
        "text": "Tesla announced a new factory in Chennai. The factory will create 10000 jobs by 2027.",
        "collection": "rag_eval_test_1",
        "question": "How many jobs will be created?",
        "must_contain": ["10000", "10,000"]
    },
    {
        "text": "The meeting is scheduled for March 15 2026 at 3pm with John Smith.",
        "collection": "rag_eval_test_2",
        "question": "When is the meeting?",
        "must_contain": ["March 15", "3pm"]
    }
]

def test_rag_accuracy():
    indexer = DocumentIndexer()
    rag = RAGTool()
    passed = 0

    for case in RAG_TEST_CASES:
        indexer.index_text(case["text"], case["collection"])
        result = rag.run({"question": case["question"], "collection_name": case["collection"]})
        answer = result.output.get("answer", "") if result.output else ""

        ok = any(term.lower() in answer.lower() for term in case["must_contain"])
        passed += ok
        print(f"  Q: {case['question']}")
        print(f"  A: {answer}")
        print(f"  {'PASS' if ok else 'FAIL'}")

    accuracy = (passed / len(RAG_TEST_CASES)) * 100
    print(f"\nRAG Accuracy: {passed}/{len(RAG_TEST_CASES)} ({accuracy:.0f}%)")

if __name__ == "__main__":
    test_rag_accuracy()
