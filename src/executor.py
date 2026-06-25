import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import Dict, Any
from loguru import logger
from src.models import (
    Workflow, AgentState, WorkflowStatus,
    StepStatus, ToolResult, ExecutionTrace, ReflectionResult
)
from src.registry import build_default_registry
from src.reflection import ReflectionAgent

MAX_RETRIES = 2
MAX_STEPS = 10


class WorkflowExecutor:
    def __init__(self):
        self.registry = build_default_registry()
        self.reflector = ReflectionAgent()

    def execute(self, workflow: Workflow, initial_input: Dict[str, Any],
                goal: str = "") -> tuple[AgentState, ExecutionTrace]:

        start_time = time.time()
        logger.info(f"[executor] Starting: '{workflow.name}' | Goal: {goal}")

        state = AgentState(
            workflow_id=workflow.id,
            memory={"input": initial_input},
            status=WorkflowStatus.RUNNING
        )

        trace = ExecutionTrace(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            goal=goal
        )

        workflow.status = WorkflowStatus.RUNNING
        steps_run = 0

        for step in workflow.steps:

            # Guard: never run more than MAX_STEPS
            if steps_run >= MAX_STEPS:
                logger.warning("[executor] Max steps reached, stopping")
                break

            logger.info(f"[executor] ── Step {steps_run + 1}: {step.name}")
            step.status = StepStatus.RUNNING
            state.current_step_id = step.id

            # Build input — always include original text + memory from deps
            tool_input = {"text": initial_input.get("text", "")}

            # Pass entity_types if this is an NER step
            if step.tool_name == "ner" and step.input_map.get("entity_types"):
                tool_input["entity_types"] = step.input_map["entity_types"]

            # If step depends on another, pass that step's output too
            for dep_id in step.depends_on:
                dep_output = state.memory.get(dep_id)
                if dep_output:
                    tool_input[f"{dep_id}_output"] = dep_output

            # Run step with retry logic
            result = self._run_with_retry(step.tool_name, tool_input)

            step_trace = {
                "step_id": step.id,
                "step_name": step.name,
                "tool_name": step.tool_name,
                "input": tool_input,
                "output": result.output,
                "error": result.error,
                "duration_ms": result.duration_ms,
                "retries": 0
            }

            if result.error:
                logger.error(f"[executor] Step '{step.name}' failed after retries: {result.error}")
                step.status = StepStatus.FAILED
                step.error = result.error
                step_trace["status"] = "failed"
                trace.steps_trace.append(step_trace)
                steps_run += 1
                continue

            # Step succeeded
            step.status = StepStatus.DONE
            step.output = result.output
            state.memory[step.id] = result.output
            state.tool_results.append(result)
            step_trace["status"] = "done"
            trace.steps_trace.append(step_trace)
            steps_run += 1

            # Reflect on this step's output
            reflection = self.reflector.reflect(
                goal=goal,
                step_id=step.id,
                step_name=step.name,
                tool_name=step.tool_name,
                output=result.output
            )
            trace.reflections.append(reflection)

            # If reflection says goal is already achieved, stop early
            if reflection.goal_achieved:
                logger.info(f"[executor] Goal achieved after step '{step.name}', stopping early")
                break

            # If reflection says quality is failed and shouldn't continue
            if not reflection.should_continue:
                logger.warning(f"[executor] Reflection says stop after '{step.name}'")
                break

        # Final status
        failed_steps = [s for s in workflow.steps if s.status == StepStatus.FAILED]
        workflow.status = WorkflowStatus.FAILED if failed_steps else WorkflowStatus.DONE
        state.status = workflow.status

        trace.final_status = workflow.status.value
        trace.total_duration_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(f"[executor] Done in {trace.total_duration_ms}ms | Status: {workflow.status}")
        return state, trace

    def _run_with_retry(self, tool_name: str, tool_input: Dict,
                        retries: int = MAX_RETRIES) -> ToolResult:
        for attempt in range(retries + 1):
            try:
                result = self.registry.run_tool(tool_name, tool_input)
                if result.error and attempt < retries:
                    logger.warning(f"[executor] Retry {attempt + 1}/{retries} for {tool_name}")
                    continue
                return result
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"[executor] Retry {attempt + 1}/{retries} for {tool_name}: {e}")
                else:
                    from src.models import ToolResult
                    return ToolResult(
                        tool_name=tool_name,
                        input=tool_input,
                        error=str(e)
                    )

    def execute_stream(self, workflow: Workflow, initial_input: Dict[str, Any],
                       goal: str = ""):
        start_time = time.time()
        logger.info(f"[executor] Starting stream: '{workflow.name}' | Goal: {goal}")

        state = AgentState(
            workflow_id=workflow.id,
            memory={"input": initial_input},
            status=WorkflowStatus.RUNNING
        )

        trace = ExecutionTrace(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            goal=goal
        )

        workflow.status = WorkflowStatus.RUNNING
        steps_run = 0

        yield {
            "type": "init",
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "goal": goal
        }

        for step in workflow.steps:
            if steps_run >= MAX_STEPS:
                logger.warning("[executor] Max steps reached, stopping")
                break

            logger.info(f"[executor] ── Step {steps_run + 1}: {step.name}")
            step.status = StepStatus.RUNNING
            state.current_step_id = step.id

            tool_input = {"text": initial_input.get("text", "")}

            if step.tool_name == "ner" and step.input_map.get("entity_types"):
                tool_input["entity_types"] = step.input_map["entity_types"]

            for dep_id in step.depends_on:
                dep_output = state.memory.get(dep_id)
                if dep_output:
                    tool_input[f"{dep_id}_output"] = dep_output

            result = self._run_with_retry(step.tool_name, tool_input)

            step_trace = {
                "step_id": step.id,
                "step_name": step.name,
                "tool_name": step.tool_name,
                "input": tool_input,
                "output": result.output,
                "error": result.error,
                "duration_ms": result.duration_ms,
                "retries": 0
            }

            if result.error:
                step.status = StepStatus.FAILED
                step.error = result.error
                step_trace["status"] = "failed"
                trace.steps_trace.append(step_trace)
                steps_run += 1
                yield {"type": "step", "step": step_trace, "reflection": None}
                continue

            step.status = StepStatus.DONE
            step.output = result.output
            state.memory[step.id] = result.output
            state.tool_results.append(result)
            step_trace["status"] = "done"
            trace.steps_trace.append(step_trace)
            steps_run += 1

            reflection = self.reflector.reflect(
                goal=goal,
                step_id=step.id,
                step_name=step.name,
                tool_name=step.tool_name,
                output=result.output
            )
            trace.reflections.append(reflection)

            refl_dict = {
                "step_id": reflection.step_id,
                "quality": reflection.quality,
                "reason": reflection.reason,
                "should_continue": reflection.should_continue
            }

            yield {"type": "step", "step": step_trace, "reflection": refl_dict}

            if reflection.goal_achieved:
                break
            if not reflection.should_continue:
                break

        failed_steps = [s for s in workflow.steps if s.status == StepStatus.FAILED]
        workflow.status = WorkflowStatus.FAILED if failed_steps else WorkflowStatus.DONE
        state.status = workflow.status

        trace.final_status = workflow.status.value
        trace.total_duration_ms = round((time.time() - start_time) * 1000, 2)

        from src.workflow_store import save_trace
        save_trace(trace)

        yield {
            "type": "done",
            "status": trace.final_status,
            "total_duration_ms": trace.total_duration_ms
        }
