from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .evaluation import edge_feedback_metrics, edge_metrics
from .pipeline_executor import PipelineExecutor
from .pipeline_validator import PipelineValidator
from .schemas import PipelinePlan


@dataclass
class AgentExecution:
    iteration: int
    plan: PipelinePlan
    result: np.ndarray
    executor_ms: float
    feedback: dict[str, float]
    gt_metrics: dict[str, float] | None = None

    def trace_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "plan": self.plan.to_dict(),
            "executor_ms": self.executor_ms,
            "feedback": dict(self.feedback),
            "gt_metrics": None if self.gt_metrics is None else dict(self.gt_metrics),
        }


class AgentTools:
    """Deterministic tools exposed to the DeepSeek-driven agent loop.

    DeepSeek proposes/revises plans, but Python remains responsible for validation,
    hardware execution and measurement. This keeps the real-time path auditable
    and prevents the LLM from directly touching MMIO/DMA registers.
    """

    def __init__(
        self,
        *,
        validator: PipelineValidator,
        executor: PipelineExecutor,
        constraints: dict,
        gt_edge: np.ndarray | None = None,
    ):
        self.validator = validator
        self.executor = executor
        self.constraints = dict(constraints)
        self.gt_edge = gt_edge

    def validate_pipeline(self, plan: PipelinePlan) -> None:
        self.validator.validate(
            plan,
            max_pre=int(self.constraints["max_pre_ops"]),
            max_post=int(self.constraints["max_post_ops"]),
            allow_hed=bool(self.constraints["allow_hed"]),
            target_backend=str(self.constraints["backend"]),
            image_rows=int(self.constraints["image_rows"]),
            image_cols=int(self.constraints["image_cols"]),
        )

    def execute_pipeline(self, image_bgr: np.ndarray, plan: PipelinePlan, iteration: int) -> AgentExecution:
        self.validate_pipeline(plan)
        t0 = time.perf_counter()
        result = self.executor.run(image_bgr, plan)
        executor_ms = (time.perf_counter() - t0) * 1000.0

        feedback = edge_feedback_metrics(result)
        feedback["executor_ms"] = float(executor_ms)

        gt_result = None
        if self.gt_edge is not None:
            gt = self.gt_edge
            if gt.shape != result.shape:
                gt = cv2.resize(
                    gt,
                    (result.shape[1], result.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            gt_result = edge_metrics(result, gt)
            feedback.update({f"gt_{k}": float(v) for k, v in gt_result.items()})

        return AgentExecution(
            iteration=iteration,
            plan=plan,
            result=result,
            executor_ms=executor_ms,
            feedback=feedback,
            gt_metrics=gt_result,
        )
