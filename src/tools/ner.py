import os, time, json
from typing import Dict, Any
from dotenv import load_dotenv
from groq import Groq
from src.tools.base import BaseTool
from src.models import ToolResult
from loguru import logger

load_dotenv(override=True)


class NERTool(BaseTool):
    name = "ner"
    description = "Extracts named entities from text — people, places, organizations, dates, and more."
    input_schema = {
        "text": "The text to extract entities from",
        "entity_types": "List of entity types to extract (default: PERSON, ORG, LOCATION, DATE)"
    }

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    def run(self, input: Dict[str, Any]) -> ToolResult:
        start = time.time()
        text = input.get("text", "")

        # Use entity_types from input if provided, otherwise default
        entity_types = input.get("entity_types") or ["PERSON", "ORG", "LOCATION", "DATE"]
        # Filter out empty strings just in case
        entity_types = [e for e in entity_types if e]

        if not text:
            return self._make_result(input, None, start, error="No text provided")

        types_str = ", ".join(entity_types)
        prompt = f"""Extract all named entities from the text below.
    Only extract these entity types: {types_str}
    Do NOT extract any other entity types even if present.

    Respond in this exact JSON format with no extra text:
    {{"entities": [{{"text": "<entity text>", "type": "<entity type>", "start_index": <int>}}]}}

    If no entities found, return: {{"entities": []}}

    Text:
    {text}"""

        logger.info(f"[ner] Extracting types={entity_types} from: {text[:60]}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)

            # Post-processing safety filter — remove any entity types not requested
            all_entities = result.get("entities", [])
            filtered = [e for e in all_entities if e.get("type") in entity_types]

            if len(filtered) < len(all_entities):
                logger.info(f"[ner] Filtered out {len(all_entities) - len(filtered)} unwanted entity types")

            final = {"entities": filtered, "entity_types_requested": entity_types}
            logger.info(f"[ner] Found {len(filtered)} entities of types {entity_types}")
            return self._make_result(input, final, start)

        except Exception as e:
            logger.error(f"[ner] Failed: {e}")
            return self._make_result(input, None, start, error=str(e))