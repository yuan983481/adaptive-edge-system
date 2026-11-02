from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .agent_tools import AgentExecution, AgentTools
from .llm_planner import LLMPlanner
from .schemas import ImageState, PipelinePlan


@dataclass
class AgentRunResult:
    result: np.ndarray
    plan: PipelinePlan
    trace: list[dict[str, Any]]
    stop_reason: str


class DeepSeekFPGAAgent:
    """Closed-loop, hardware-aware orchestration around the existing LLM planner.

    Loop:
        image state -> DeepSeek plan -> validate -> execute -> measure
                    -> DeepSeek refinement (when feedback says it is useful)

    The LLM never writes FPGA registers directly. All side effects go through
    AgentTools, which preserves the project's existing validator/compiler/runtime.
    """

    def __init__(
        self,
        *,
        planner: LLMPlanner,
        tools: AgentTools,
        max_iterations: int = 2,
        edge_density_min: float = 0.015,
        edge_density_max: float = 0.25,
        max_tiny_component_ratio: float = 0.65,
        min_gt_f1: float = 0.80,
    ):
        self.planner = planner
        self.tools = tools
        self.max_iterations = max(1, int(max_iterations))
        self.edge_density_min = float(edge_density_min)
        self.edge_density_max = float(edge_density_max)
        self.max_tiny_component_ratio = float(max_tiny_component_ratio)
        self.min_gt_f1 = float(min_gt_f1)

    def _assessment(self, execution: AgentExecution) -> tuple[bool, str]:
        f = execution.feedback

        # If GT exists, use it as the strongest evaluation signal.
        if execution.gt_metrics is not None:
            f1 = float(execution.gt_metrics.get("f1", 0.0))
            if f1 >= self.min_gt_f1:
                return True, f"GT F1 target reached ({f1:.3f} >= {self.min_gt_f1:.3f})"
            return False, f"GT F1 below target ({f1:.3f} < {self.min_gt_f1:.3f})"

        density = float(f.get("edge_density", 0.0))
        tiny_ratio = float(f.get("tiny_component_ratio", 0.0))

        problems: list[str] = []
        if density < self.edge_density_min:
            problems.append(
                f"edge map is too sparse: density={density:.4f} < {self.edge_density_min:.4f}"
            )
        if density > self.edge_density_max:
            problems.append(
                f"edge map is too dense: density={density:.4f} > {self.edge_density_max:.4f}"
            )
        if tiny_ratio > self.max_tiny_component_ratio and density > self.edge_density_min:
            problems.append(
                "edge map is fragmented/noisy: "
                f"tiny_component_ratio={tiny_ratio:.3f} > {self.max_tiny_component_ratio:.3f}"
            )

        if not problems:
            return True, "no-reference feedback is inside the configured acceptance window"
        return False, "; ".join(problems)

    def run(self, image_bgr: np.ndarray, state: ImageState, constraints: dict[str, Any]) -> AgentRunResult:
        plan = self.planner.plan(state, constraints)
        trace: list[dict[str, Any]] = []
        stop_reason = "maximum iterations reached"
        best: AgentExecution | None = None
        best_score = float("-inf")

        for iteration in range(1, self.max_iterations + 1):
            execution = self.tools.execute_pipeline(image_bgr, plan, iteration)
            accepted, assessment = self._assessment(execution)

            score = self._score(execution)
            if best is None or score > best_score:
                best = execution
                best_score = score

            record = execution.trace_dict()
            record["accepted"] = accepted
            record["assessment"] = assessment
            record["agent_score"] = score
            trace.append(record)

            if accepted:
                stop_reason = assessment
                best = execution
                break

            if iteration >= self.max_iterations:
                stop_reason = assessment + "; maximum iterations reached"
                break

            plan = self.planner.refine(
                state,
                constraints,
                previous_plan=plan,
                execution_feedback=execution.feedback,
                assessment=assessment,
                iteration=iteration,
            )

        assert best is not None
        return AgentRunResult(
            result=best.result,
            plan=best.plan,
            trace=trace,
            stop_reason=stop_reason,
        )

    def _score(self, execution: AgentExecution) -> float:
        """Choose the safest/best observed iteration if the loop hits its limit."""
        if execution.gt_metrics is not None:
            return float(execution.gt_metrics.get("f1", 0.0))

        f = execution.feedback
        density = float(f.get("edge_density", 0.0))
        tiny_ratio = float(f.get("tiny_component_ratio", 1.0))
        midpoint = (self.edge_density_min + self.edge_density_max) / 2.0
        span = max((self.edge_density_max - self.edge_density_min) / 2.0, 1e-6)
        density_score = max(0.0, 1.0 - abs(density - midpoint) / span)
        continuity_score = max(0.0, 1.0 - tiny_ratio)
        return 0.65 * density_score + 0.35 * continuity_score
