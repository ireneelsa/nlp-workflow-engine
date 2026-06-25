import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from typing import Dict, Any

from src.models import AgentState, WorkflowStatus, ExecutionTrace
from src.workflow_store import save_workflow


class PlannerAgent:
    def __init__(self):
        self.planner = None

    def _get_planner(self):
        if self.planner is None:
            from src.planner import Planner
            self.planner = Planner()
        return self.planner

    def run(self, state: AgentState, goal: str):
        logger.info("[PlannerAgent] Creating workflow plan")
        workflow = self._get_planner().plan(goal)
        save_workflow(workflow)
        
        state.workflow_id = workflow.id
        state.workflow = workflow
        state.status = WorkflowStatus.RUNNING


class NLPAgent:
    def __init__(self):
        self.executor = None

    def _get_executor(self):
        if self.executor is None:
            from src.executor import WorkflowExecutor
            self.executor = WorkflowExecutor()
        return self.executor

    def run(self, state: AgentState, goal: str) -> ExecutionTrace:
        logger.info("[NLPAgent] Executing workflow")
        initial_input = state.memory.get("input", {})
        new_state, trace = self._get_executor().execute(state.workflow, initial_input, goal)
        
        # Sync back to the shared state
        state.status = new_state.status
        state.memory = new_state.memory
        state.tool_results = new_state.tool_results
        state.current_step_id = new_state.current_step_id
        
        return trace
        
    def run_stream(self, state: AgentState, goal: str):
        logger.info("[NLPAgent] Executing workflow (stream)")
        initial_input = state.memory.get("input", {})
        yield from self._get_executor().execute_stream(state.workflow, initial_input, goal)


class RouterAgent:
    def __init__(self):
        self.planner = PlannerAgent()
        self.nlp = NLPAgent()

    def route_and_run(self, state: AgentState, goal: str) -> ExecutionTrace:
        logger.info("[RouterAgent] Routing request...")
        
        if not state.workflow_id or not state.workflow:
            self.planner.run(state, goal)
            
        if state.status == WorkflowStatus.RUNNING:
            return self.nlp.run(state, goal)
            
        # Fallback empty trace
        from src.models import ExecutionTrace
        return ExecutionTrace(workflow_id=state.workflow_id or "", workflow_name="", goal=goal)
        
    def route_and_stream(self, state: AgentState, goal: str):
        logger.info("[RouterAgent] Routing stream request...")
        
        if not state.workflow_id or not state.workflow:
            self.planner.run(state, goal)
            
        if state.status == WorkflowStatus.RUNNING:
            yield from self.nlp.run_stream(state, goal)
