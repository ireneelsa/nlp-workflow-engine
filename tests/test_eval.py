import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.registry import build_default_registry

registry = build_default_registry()

# ── Eval cases ────────────────────────────────────────────────────────

SUMMARIZER_CASES = [
    {
        "input": {
            "text": "Elon Musk announced Tesla will open a Gigafactory in Chennai, India by 2027. It will produce 500,000 EVs annually and create 10,000 jobs. PM Modi welcomed the news.",
            "max_sentences": 2
        },
        "must_contain": ["Tesla", "Chennai"],
        "max_length": 300
    },
    {
        "input": {
            "text": "Apple released the iPhone 17 in September 2026. The new model features a titanium body, a 48MP camera, and runs on the A20 chip. It starts at $999.",
            "max_sentences": 1
        },
        "must_contain": ["iPhone"],
        "max_length": 200
    },
    {
        "input": {
            "text": "Scientists at NASA confirmed the discovery of water ice on Mars near the Jezero crater. The finding was published in the journal Nature and could support future human missions.",
            "max_sentences": 2
        },
        "must_contain": ["NASA", "Mars"],
        "max_length": 300
    }
]

CLASSIFIER_CASES = [
    {
        "input": {
            "text": "I have been waiting 3 weeks for my refund and nobody is responding to my emails.",
            "labels": ["complaint", "question", "request", "news", "feedback"]
        },
        "expected_intent": "complaint"
    },
    {
        "input": {
            "text": "What are the opening hours of your Mumbai office?",
            "labels": ["complaint", "question", "request", "news", "feedback"]
        },
        "expected_intent": "question"
    },
    {
        "input": {
            "text": "Tesla announced record profits for Q1 2026 driven by strong EV demand in Asia.",
            "labels": ["complaint", "question", "request", "news", "feedback"]
        },
        "expected_intent": "news"
    }
]

NER_CASES = [
    {
        "input": {
            "text": "Sundar Pichai visited Google's office in London on March 5, 2026.",
            "entity_types": ["PERSON", "ORG", "LOCATION", "DATE"]
        },
        "must_find_types": ["PERSON", "ORG", "LOCATION", "DATE"],
        "must_find_entities": ["Sundar Pichai", "Google", "London"]
    },
    {
        "input": {
            "text": "Microsoft and OpenAI signed a new partnership agreement in Seattle on January 10, 2026.",
            "entity_types": ["ORG", "LOCATION"]
        },
        "must_find_types": ["ORG", "LOCATION"],
        "must_find_entities": ["Microsoft", "OpenAI", "Seattle"]
    }
]


# ── Eval runner ───────────────────────────────────────────────────────

def score_summarizer():
    print("\n── Summarizer Eval ──")
    passed = 0
    for i, case in enumerate(SUMMARIZER_CASES):
        result = registry.run_tool("summarizer", case["input"])
        if result.error:
            print(f"  Case {i+1}: FAIL (error: {result.error})")
            continue

        output = result.output or ""
        checks = []
        for word in case["must_contain"]:
            checks.append(word in output)
        checks.append(len(output) <= case["max_length"])
        checks.append(len(output) > 20)

        ok = all(checks)
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  Case {i+1}: {status} | length={len(output)} | {output[:80]}...")

    score = (passed / len(SUMMARIZER_CASES)) * 100
    print(f"  Score: {passed}/{len(SUMMARIZER_CASES)} ({score:.0f}%)")
    return score


def score_classifier():
    print("\n── Intent Classifier Eval ──")
    passed = 0
    for i, case in enumerate(CLASSIFIER_CASES):
        result = registry.run_tool("intent_classifier", case["input"])
        if result.error:
            print(f"  Case {i+1}: FAIL (error: {result.error})")
            continue

        predicted = result.output.get("intent", "")
        expected = case["expected_intent"]
        ok = predicted == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  Case {i+1}: {status} | expected={expected} | got={predicted}")

    score = (passed / len(CLASSIFIER_CASES)) * 100
    print(f"  Score: {passed}/{len(CLASSIFIER_CASES)} ({score:.0f}%)")
    return score


def score_ner():
    print("\n── NER Eval ──")
    passed = 0
    for i, case in enumerate(NER_CASES):
        result = registry.run_tool("ner", case["input"])
        if result.error:
            print(f"  Case {i+1}: FAIL (error: {result.error})")
            continue

        entities = result.output.get("entities", [])
        found_types = set(e["type"] for e in entities)
        found_texts = [e["text"] for e in entities]

        type_check = all(t in found_types for t in case["must_find_types"])
        entity_check = any(
            any(must.lower() in found.lower() for found in found_texts)
            for must in case["must_find_entities"]
        )

        ok = type_check and entity_check
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  Case {i+1}: {status} | found={found_texts}")

    score = (passed / len(NER_CASES)) * 100
    print(f"  Score: {passed}/{len(NER_CASES)} ({score:.0f}%)")
    return score


if __name__ == "__main__":
    print("=" * 50)
    print("NLP WORKFLOW ENGINE — EVAL HARNESS")
    print("=" * 50)

    s1 = score_summarizer()
    s2 = score_classifier()
    s3 = score_ner()

    overall = (s1 + s2 + s3) / 3
    print(f"\n{'=' * 50}")
    print(f"OVERALL SCORE: {overall:.0f}%")
    if overall >= 80:
        print("STATUS: PASS — system is production ready")
    elif overall >= 60:
        print("STATUS: WARN — needs improvement")
    else:
        print("STATUS: FAIL — significant issues found")
    print("=" * 50)