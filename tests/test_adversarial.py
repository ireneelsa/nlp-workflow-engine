import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.registry import build_default_registry
from src.planner import Planner
from src.executor import WorkflowExecutor
from src.models import WorkflowStatus

registry = build_default_registry()
planner = Planner()
executor = WorkflowExecutor()


def test_empty_text():
    """All tools must handle empty input without crashing."""
    print("\n── Empty text ──")
    for tool in ["summarizer", "intent_classifier", "ner"]:
        result = registry.run_tool(tool, {"text": ""})
        assert result.error is not None, f"{tool} should return error on empty input"
        print(f"  {tool}: correctly returned error — {result.error[:50]}")
    print("  PASS")


def test_gibberish_text():
    """Tools must handle nonsense input without crashing."""
    print("\n── Gibberish text ──")
    gibberish = "asdkjh qwerty zxcvb 1234567 !@#$%^ asdf jkl;"
    for tool in ["summarizer", "intent_classifier", "ner"]:
        result = registry.run_tool(tool, {"text": gibberish})
        assert result.error is None, f"{tool} crashed on gibberish: {result.error}"
        print(f"  {tool}: handled gracefully — output={str(result.output)[:60]}")
    print("  PASS")


def test_very_long_text():
    """Tools must handle long text without crashing."""
    print("\n── Very long text (5000 chars) ──")
    long_text = "Elon Musk announced a new Tesla factory. " * 120
    for tool in ["summarizer", "intent_classifier", "ner"]:
        result = registry.run_tool(tool, {"text": long_text})
        assert result.error is None, f"{tool} crashed on long text: {result.error}"
        print(f"  {tool}: handled {len(long_text)} chars — OK")
    print("  PASS")


def test_single_word_text():
    """Tools must handle minimal input."""
    print("\n── Single word text ──")
    for tool in ["summarizer", "intent_classifier", "ner"]:
        result = registry.run_tool(tool, {"text": "Hello"})
        assert result.error is None, f"{tool} crashed on single word"
        print(f"  {tool}: OK — {str(result.output)[:60]}")
    print("  PASS")


def test_nonexistent_tool():
    """Registry must fail gracefully for unknown tools."""
    print("\n── Nonexistent tool ──")
    result = executor._run_with_retry("fake_tool_xyz", {"text": "test"})
    assert result.error is not None
    print(f"  Got expected error: {result.error[:60]}")
    print("  PASS")


def test_full_pipeline_with_bad_goal():
    """Planner must not crash on a vague or weird goal."""
    print("\n── Vague goal ──")
    try:
        workflow = planner.plan("do something interesting")
        state, trace = executor.execute(
            workflow,
            {"text": "Apple released a new product last week."},
            goal="do something interesting"
        )
        assert trace.final_status in ["done", "failed"]
        print(f"  Planner created {len(workflow.steps)} steps, status={trace.final_status}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL — crashed with: {e}")
        raise


if __name__ == "__main__":
    print("=" * 50)
    print("NLP WORKFLOW ENGINE — ADVERSARIAL TESTS")
    print("=" * 50)

    test_empty_text()
    test_gibberish_text()
    test_very_long_text()
    test_single_word_text()
    test_nonexistent_tool()
    test_full_pipeline_with_bad_goal()

    print(f"\n{'=' * 50}")
    print("ALL ADVERSARIAL TESTS PASSED")
    print("System handles edge cases gracefully.")
    print("=" * 50)