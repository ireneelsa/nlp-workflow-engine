from abc import ABC, abstractmethod
from typing import Dict, Any
from src.models import ToolResult
import time


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    input_schema: Dict = {}

    @abstractmethod
    def run(self, input: Dict[str, Any]) -> ToolResult:
        pass

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def _make_result(self, input, output, start_time, error=None) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            input=input,
            output=output,
            error=error,
            duration_ms=round((time.time() - start_time) * 1000, 2)
        )