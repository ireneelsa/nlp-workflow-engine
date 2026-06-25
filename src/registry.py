import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any

from src.tools.base import BaseTool
from src.models import ToolResult
from loguru import logger


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_tools(self):
        return [t.to_dict() for t in self._tools.values()]

    def run_tool(self, name: str, input: Dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        logger.info(f"Running tool: {name}")
        result = tool.run(input)
        if result.error:
            logger.error(f"Tool {name} failed: {result.error}")
        else:
            logger.info(f"Tool {name} completed in {result.duration_ms}ms")
        return result


def build_default_registry() -> ToolRegistry:
    from src.tools.summarizer import SummarizerTool
    from src.tools.classifier import IntentClassifierTool
    from src.tools.ner import NERTool
    from src.tools.rag import RAGTool

    registry = ToolRegistry()
    registry.register(SummarizerTool())
    registry.register(IntentClassifierTool())
    registry.register(NERTool())
    registry.register(RAGTool())
    return registry
