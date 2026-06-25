import json, os
from src.models import Workflow, ExecutionTrace
from loguru import logger

WORKFLOWS_DIR = "workflows"
TRACES_DIR = "workflows/traces"


def save_workflow(workflow: Workflow):
    os.makedirs(WORKFLOWS_DIR, exist_ok=True)
    path = os.path.join(WORKFLOWS_DIR, f"{workflow.id}.json")
    with open(path, "w") as f:
        json.dump(workflow.model_dump(), f, indent=2)
    logger.info(f"[store] Saved workflow to {path}")
    return path


def save_trace(trace: ExecutionTrace):
    os.makedirs(TRACES_DIR, exist_ok=True)
    path = os.path.join(TRACES_DIR, f"{trace.workflow_id}.json")
    with open(path, "w") as f:
        json.dump(trace.model_dump(), f, indent=2)
    logger.info(f"[store] Saved trace to {path}")
    return path


def load_workflow(workflow_id: str) -> Workflow:
    path = os.path.join(WORKFLOWS_DIR, f"{workflow_id}.json")
    with open(path) as f:
        data = json.load(f)
    return Workflow(**data)


def list_workflows():
    os.makedirs(WORKFLOWS_DIR, exist_ok=True)
    files = [f for f in os.listdir(WORKFLOWS_DIR) if f.endswith(".json")]
    return [f.replace(".json", "") for f in files]