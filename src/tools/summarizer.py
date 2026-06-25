import os, time
from typing import Dict, Any
from dotenv import load_dotenv
from groq import Groq
from src.tools.base import BaseTool
from src.models import ToolResult
from loguru import logger

load_dotenv(override=True)


class SummarizerTool(BaseTool):
    name = "summarizer"
    description = "Summarizes a long piece of text into a short, clear summary."
    input_schema = {
        "text": "The text you want to summarize",
        "max_sentences": "How many sentences the summary should be (default: 3)"
    }

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    def run(self, input: Dict[str, Any]) -> ToolResult:
        start = time.time()
        text = input.get("text", "")
        max_sentences = input.get("max_sentences", 3)

        if not text:
            return self._make_result(input, None, start, error="No text provided")

        prompt = f"""Summarize the following text in {max_sentences} sentences or fewer.
Be concise and capture only the most important points.
Return ONLY the summary, nothing else.

Text:
{text}"""

        logger.info(f"[summarizer] Running on {len(text)} chars")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256
            )
            summary = response.choices[0].message.content.strip()
            logger.info(f"[summarizer] Done: {summary[:60]}...")
            return self._make_result(input, summary, start)
        except Exception as e:
            logger.error(f"[summarizer] Failed: {e}")
            return self._make_result(input, None, start, error=str(e))