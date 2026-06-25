from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


class Step(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    tool_name: str
    depends_on: List[str] = []
    input_map: Dict[str, Any] = {}    
    status: StepStatus = StepStatus.PENDING
    output: Optional[Any] = None
    error: Optional[str] = None


class Workflow(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str = ""
    steps: List[Step] = []
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ToolResult(BaseModel):
    tool_name: str
    input: Dict[str, Any]
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentState(BaseModel):
    workflow_id: Optional[str] = None
    workflow: Optional[Workflow] = None
    current_step_id: Optional[str] = None
    memory: Dict[str, Any] = {}
    tool_results: List[ToolResult] = []
    messages: List[Dict[str, str]] = []
    status: WorkflowStatus = WorkflowStatus.PENDING

class ReflectionResult(BaseModel):
    step_id: str
    goal_achieved: bool
    quality: str  # "good", "poor", "failed"
    reason: str
    should_continue: bool


class ExecutionTrace(BaseModel):
    workflow_id: str
    workflow_name: str
    goal: str
    steps_trace: List[Dict[str, Any]] = []
    reflections: List[ReflectionResult] = []
    final_status: str = "pending"
    total_duration_ms: float = 0.0
    total_tokens_used: int = 0