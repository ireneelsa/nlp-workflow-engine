import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from typing import List
from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from src.models import Workflow, Step
from src.registry import build_default_registry

load_dotenv(override=True)


PLANNER_PROMPT = """You are a workflow planner for an NLP system.
You have access to these tools:

{tools}

A user has given you this goal:
"{goal}"

Your job is to create a workflow plan — a list of steps using the available tools that will achieve the goal.

Rules:
- Only use tools from the list above
- Each step must use exactly one tool
- Steps run in order (top to bottom)
- Use "depends_on" to reference a previous step's id if that step's output feeds into this one
- Keep it simple — 2 to 4 steps maximum
- If the goal mentions specific entity types like "people", "organizations", "locations", "dates",
  extract those keywords and pass them as entity_types to the ner tool.
  Map them like this:
    "people" or "person" → PERSON
    "organizations" or "companies" → ORG
    "locations" or "places" or "cities" or "countries" → LOCATION
    "dates" or "times" → DATE
  If the goal does not mention specific types, use all: ["PERSON", "ORG", "LOCATION", "DATE"]

Respond in this EXACT JSON format with no extra text:
{{
  "workflow_name": "<short name for this workflow>",
  "workflow_description": "<one sentence describing what this workflow does>",
  "steps": [
    {{
      "id": "step_1",
      "name": "<short step name>",
      "tool_name": "<exact tool name>",
      "depends_on": [],
      "entity_types": ["PERSON", "ORG"],
      "reason": "<why this step is needed>"
    }}
  ]
}}

Note: "entity_types" is only used when tool_name is "ner". For other tools, set it to []."""


class Planner:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.registry = build_default_registry()

    def plan(self, goal: str) -> Workflow:
        logger.info(f"[planner] Planning for goal: {goal}")

        tools = self.registry.list_tools()
        tools_str = "\n".join(
            f"- {t['name']}: {t['description']}" for t in tools
        )

        prompt = PLANNER_PROMPT.format(tools=tools_str, goal=goal)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )

        raw = response.choices[0].message.content.strip()
        logger.info(f"[planner] Raw response: {raw}")

        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw.strip())
        plan_data = json.loads(raw)

        steps = [
            Step(
                id=s["id"],
                name=s["name"],
                tool_name=s["tool_name"],
                depends_on=s.get("depends_on", []),
                input_map={"entity_types": s.get("entity_types", [])} if s["tool_name"] == "ner" else {}
            )
            for s in plan_data["steps"]
        ]

        workflow = Workflow(
            name=plan_data["workflow_name"],
            description=plan_data["workflow_description"],
            steps=steps
        )

        logger.info(f"[planner] Created workflow '{workflow.name}' with {len(steps)} steps")
        return workflow
