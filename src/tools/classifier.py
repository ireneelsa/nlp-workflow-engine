import os, time, json
from typing import Dict, Any
from dotenv import load_dotenv
from groq import Groq
from src.tools.base import BaseTool
from src.models import ToolResult
from loguru import logger

load_dotenv(override=True)


class IntentClassifierTool(BaseTool):
    name = "intent_classifier"
    description = "Classifies the intent of a piece of text. Returns the intent label and confidence."
    input_schema = {
        "text": "The text to classify",
        "labels": "List of possible intent labels to choose from"
    }

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    def run(self, input: Dict[str, Any]) -> ToolResult:
        start = time.time()
        text = input.get("text", "")
        labels = input.get("labels", ["question", "complaint", "request", "feedback", "other"])

        if not text:
            return self._make_result(input, None, start, error="No text provided")

        labels_str = ", ".join(labels)
        prompt = f"""Classify the intent of the following text.
Choose ONLY from these labels: {labels_str}

Respond in this exact JSON format with no extra text:
{{"intent": "<label>", "confidence": <0.0 to 1.0>, "reason": "<one sentence why>"}}

Text:
{text}"""

        logger.info(f"[intent_classifier] Classifying: {text[:60]}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            logger.info(f"[intent_classifier] Intent: {result.get('intent')} ({result.get('confidence')})")
            return self._make_result(input, result, start)
        except Exception as e:
            logger.error(f"[intent_classifier] Failed: {e}")
            return self._make_result(input, None, start, error=str(e))