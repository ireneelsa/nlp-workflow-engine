import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from src.models import ReflectionResult

load_dotenv(override=True)


REFLECTION_PROMPT = """You are evaluating whether a step in an NLP workflow produced good output.

Overall goal: "{goal}"
Step name: "{step_name}"
Tool used: "{tool_name}"
Step output: {output}

Evaluate:
1. Did this step produce useful, non-empty output?
2. Does the output help achieve the overall goal?
3. Should the workflow continue to the next step?

Respond in this EXACT JSON format with no extra text:
{{
  "goal_achieved": false,
  "quality": "good",
  "reason": "<one sentence evaluation>",
  "should_continue": true
}}

For "quality" use only: "good", "poor", or "failed"
Set "goal_achieved" to true ONLY if the overall goal is fully complete after this step."""


class ReflectionAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    def reflect(self, goal: str, step_id: str, step_name: str,
                tool_name: str, output: any) -> ReflectionResult:

        logger.info(f"[reflection] Evaluating step: {step_name}")

        prompt = REFLECTION_PROMPT.format(
            goal=goal,
            step_name=step_name,
            tool_name=tool_name,
            output=json.dumps(output, default=str)[:800]
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)

            result = ReflectionResult(
                step_id=step_id,
                goal_achieved=data.get("goal_achieved", False),
                quality=data.get("quality", "good"),
                reason=data.get("reason", ""),
                should_continue=data.get("should_continue", True)
            )

            logger.info(f"[reflection] {step_name}: quality={result.quality}, continue={result.should_continue}")
            return result

        except Exception as e:
            logger.error(f"[reflection] Failed: {e}")
            return ReflectionResult(
                step_id=step_id,
                goal_achieved=False,
                quality="good",
                reason="Reflection failed, continuing anyway",
                should_continue=True
            )
