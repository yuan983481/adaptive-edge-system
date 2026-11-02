from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

import cv2
import numpy as np

from .agent_tools import AgentTools
from .llm_planner import LLMPlanner
from .pipeline_planner import RulePlanner
from .pipeline_validator import PipelineValidator
from .schemas import ImageState, PipelinePlan
from .temporal_controller import FastLoopController


@dataclass
class AsyncPlanResult:
    scene: str
    plan: PipelinePlan | None
    source: str
    elapsed_ms: float
    request_frame: int
    reason: str
    error: str = ""


@dataclass
class VideoRunStats:
    frames: int = 0
    slow_replans: int = 0
    deepseek_replans: int = 0
    fallback_replans: int = 0
    fast_adjustments: int = 0
    processing_ms_sum: float = 0.0
    llm_ms_sum: float = 0.0

    async_requests: int = 0
    async_applied: int = 0
    async_rejected: int = 0
    cache_hits: int = 0
    cache_switches: int = 0
    pending_frames: int = 0
    plan_generation: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        avg_ms = self.processing_ms_sum / max(1, self.frames)
        return {
            "frames": self.frames,
            "slow_replans": self.slow_replans,
            "deepseek_replans": self.deepseek_replans,
            "fallback_replans": self.fallback_replans,
            "fast_adjustments": self.fast_adjustments,
            "avg_processing_ms": avg_ms,
            "effective_processing_fps": 1000.0 / avg_ms if avg_ms > 0 else 0.0,
            "llm_ms_total": self.llm_ms_sum,
            "async_requests": self.async_requests,
            "async_applied": self.async_applied,
            "async_rejected": self.async_rejected,
            "cache_hits": self.cache_hits,
            "cache_switches": self.cache_switches,
            "pending_frames": self.pending_frames,
            "plan_generation": self.plan_generation,
        }


class DualLoopVideoAgent:
    """V4 continuous-video controller.

    Fast path (main/video thread):
      analyze -> EMA/hysteresis -> cached plan -> local threshold correction -> execute

    Slow path (single background thread):
      DeepSeek plan/refine only. It never touches DMA/MMIO or executes FPGA work.
      A completed proposal is validated and atomically adopted by the main thread.
    """

    def __init__(
        self,
        *,
        analyzer: Callable[[np.ndarray], ImageState],
        validator: PipelineValidator,
        tools: AgentTools,
        constraints: dict[str, Any],
        fast_controller: FastLoopController,
        llm_planner: LLMPlanner | None,
        rule_planner: RulePlanner | None = None,
        enable_scene_cache: bool = True,
        refine_on_feedback: bool = True,
    ):
        self.analyzer = analyzer
        self.validator = validator
        self.tools = tools
        self.constraints = dict(constraints)
        self.fast = fast_controller
        self.llm_planner = llm_planner
        self.rule_planner = rule_planner or RulePlanner()
        self.enable_scene_cache = bool(enable_scene_cache)
        self.refine_on_feedback = bool(refine_on_feedback)

        self.current_plan: PipelinePlan | None = None
        self.current_plan_source = "none"
        self.current_scene = "normal"
        self.previous_feedback: dict[str, float] | None = None
        self.plan_generation = 0

        self.scene_cache: dict[str, PipelinePlan] = {}
        self.scene_cache_source: dict[str, str] = {}

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepseek-slow-loop")
        self._future: Future[AsyncPlanResult] | None = None
        self._pending_scene: str | None = None
        self._pending_request_frame: int | None = None

    @property
    def slow_pending(self) -> bool:
        return self._future is not None and not self._future.done()

    def _validate(self, plan: PipelinePlan) -> None:
        self.tools.validate_pipeline(plan)

    def _rule_plan(self, state: ImageState) -> PipelinePlan:
        plan = self.rule_planner.plan(state, self.constraints)
        self._validate(plan)
        return plan

    def _async_plan_job(
        self,
        *,
        state: ImageState,
        scene: str,
        request_frame: int,
        reason: str,
        previous_plan: PipelinePlan | None,
        previous_feedback: dict[str, float] | None,
    ) -> AsyncPlanResult:
        if self.llm_planner is None:
            return AsyncPlanResult(
                scene=scene,
                plan=None,
                source="unavailable",
                elapsed_ms=0.0,
                request_frame=request_frame,
                reason=reason,
                error="DeepSeek planner unavailable",
            )

        t0 = time.perf_counter()
        try:
            # Feedback-triggered slow events are handled as a refinement request.
            # This preserves the closed-loop behavior without executing hardware
            # from the background thread.
            if (
                self.refine_on_feedback
                and previous_plan is not None
                and previous_feedback is not None
                and ("feedback" in reason.lower() or "periodic" in reason.lower())
            ):
                plan = self.llm_planner.refine(
                    state,
                    self.constraints,
                    previous_plan=previous_plan,
                    execution_feedback=previous_feedback,
                    assessment=reason,
                    iteration=1,
                )
                source = "deepseek-refine"
            else:
                plan = self.llm_planner.plan(state, self.constraints)
                source = "deepseek-plan"
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return AsyncPlanResult(
                scene=scene,
                plan=plan,
                source=source,
                elapsed_ms=elapsed_ms,
                request_frame=request_frame,
                reason=reason,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return AsyncPlanResult(
                scene=scene,
                plan=None,
                source="deepseek-error",
                elapsed_ms=elapsed_ms,
                request_frame=request_frame,
                reason=reason,
                error=str(exc),
            )

    def _submit_slow_request(
        self,
        *,
        state: ImageState,
        scene: str,
        frame_index: int,
        reason: str,
    ) -> bool:
        if self.llm_planner is None or self.slow_pending:
            return False

        self._pending_scene = scene
        self._pending_request_frame = int(frame_index)
        self._future = self._executor.submit(
            self._async_plan_job,
            state=deepcopy(state),
            scene=str(scene),
            request_frame=int(frame_index),
            reason=str(reason),
            previous_plan=deepcopy(self.current_plan),
            previous_feedback=deepcopy(self.previous_feedback),
        )
        return True

    def _collect_async_result(self, active_scene: str) -> dict[str, Any]:
        info = {
            "ready": False,
            "applied": False,
            "rejected": False,
            "scene": None,
            "source": None,
            "llm_ms": 0.0,
            "error": "",
            "request_frame": None,
        }
        if self._future is None or not self._future.done():
            return info

        future = self._future
        self._future = None
        self._pending_scene = None
        self._pending_request_frame = None
        info["ready"] = True

        try:
            result = future.result()
        except Exception as exc:
            info["rejected"] = True
            info["error"] = str(exc)
            return info

        info.update({
            "scene": result.scene,
            "source": result.source,
            "llm_ms": float(result.elapsed_ms),
            "error": result.error,
            "request_frame": result.request_frame,
        })
        if result.plan is None:
            info["rejected"] = True
            return info

        try:
            self._validate(result.plan)
        except Exception as exc:
            info["rejected"] = True
            info["error"] = f"proposal validation failed: {exc}"
            return info

        # Cache valid proposals even if the scene changed while DeepSeek was thinking.
        if self.enable_scene_cache:
            self.scene_cache[result.scene] = deepcopy(result.plan)
            self.scene_cache_source[result.scene] = result.source

        # Only switch immediately when the proposal still matches the active scene.
        if result.scene == active_scene:
            self.current_plan = deepcopy(result.plan)
            self.current_plan_source = result.source
            self.plan_generation += 1
            info["applied"] = True
        return info

    def _apply_scene_cache(self, scene: str) -> bool:
        if not self.enable_scene_cache or scene not in self.scene_cache:
            return False
        cached = deepcopy(self.scene_cache[scene])
        self._validate(cached)
        self.current_plan = cached
        self.current_plan_source = f"cache:{self.scene_cache_source.get(scene, 'validated')}"
        self.plan_generation += 1
        return True

    def process_frame(self, frame: np.ndarray, frame_index: int) -> tuple[np.ndarray, dict[str, Any]]:
        state = self.analyzer(frame)
        decision = self.fast.observe(
            state,
            frame_index=frame_index,
            previous_feedback=self.previous_feedback,
        )
        self.current_scene = decision.scene

        async_info = self._collect_async_result(decision.scene)
        cache_hit = False
        cache_switch = False
        slow_requested = False
        bootstrap_rule = False
        fast_reason = ""
        fast_adjusted = False

        # V4 bootstrap: never block frame 0 on DeepSeek. Start with a deterministic
        # valid plan and let DeepSeek improve it in the background.
        if self.current_plan is None:
            self.current_plan = self._rule_plan(decision.filtered_state)
            self.current_plan_source = "rule-bootstrap"
            self.plan_generation += 1
            bootstrap_rule = True
            if self.enable_scene_cache:
                self.scene_cache[decision.scene] = deepcopy(self.current_plan)
                self.scene_cache_source[decision.scene] = "rule-bootstrap"

        # A confirmed scene change can immediately switch to an already validated
        # scene-specific plan, without waiting for network inference.
        if (not bootstrap_rule) and decision.scene_changed and self.enable_scene_cache and decision.scene in self.scene_cache:
            cache_hit = True
            # Avoid repeatedly replacing the same scene plan on a no-op transition.
            cache_switch = self._apply_scene_cache(decision.scene)

        if decision.slow_trigger:
            # On cache hit, the cached plan solves the immediate real-time need.
            # We still allow an async refresh unless another request is pending.
            slow_requested = self._submit_slow_request(
                state=decision.filtered_state,
                scene=decision.scene,
                frame_index=frame_index,
                reason=decision.slow_reason,
            )

        tuned, fast_reason, fast_adjusted = self.fast.tune_plan(
            self.current_plan,
            self.previous_feedback,
            frame_index=frame_index,
        )
        if fast_adjusted:
            try:
                self._validate(tuned)
                self.current_plan = tuned
                self.current_plan_source = "fast-adjust"
                self.plan_generation += 1
                # Keep the cache aligned with the actually used plan for this scene.
                if self.enable_scene_cache:
                    self.scene_cache[decision.scene] = deepcopy(tuned)
                    self.scene_cache_source[decision.scene] = "fast-adjust"
            except Exception:
                fast_reason = "fast correction rejected; cached plan retained"
                fast_adjusted = False

        # Only the main thread executes the pipeline. This is critical for FPGA
        # safety: background DeepSeek work never races DMA/MMIO accesses.
        execution = self.tools.execute_pipeline(frame, self.current_plan, iteration=1)
        result = execution.result
        feedback = execution.feedback
        self.previous_feedback = dict(feedback)

        meta = {
            "frame": int(frame_index),
            "scene": decision.scene,
            "scene_changed": bool(decision.scene_changed),
            "change_score": float(decision.change_score),
            "slow_trigger": bool(decision.slow_trigger),
            "slow_reason": decision.slow_reason,
            "slow_requested": bool(slow_requested),
            "slow_pending": bool(self.slow_pending),
            "pending_scene": self._pending_scene,
            "planner_source": self.current_plan_source,
            "bootstrap_rule": bool(bootstrap_rule),
            "cache_hit": bool(cache_hit),
            "cache_switch": bool(cache_switch),
            "cache_scenes": sorted(self.scene_cache.keys()),
            "plan_generation": int(self.plan_generation),
            "async_result": async_info,
            "llm_ms": float(async_info.get("llm_ms", 0.0)),
            "fast_reason": fast_reason,
            "fast_adjusted": bool(fast_adjusted),
            "state": decision.filtered_state.to_dict(),
            "feedback": dict(feedback),
            "plan": self.current_plan.to_dict(),
        }
        return result, meta

    def close(self, *, wait: bool = False) -> None:
        # The main result is already complete even if a final background request
        # is still running. `wait=False` avoids extending video shutdown latency.
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


def make_demo_frame(frame_bgr: np.ndarray, edge: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    if edge.ndim == 2:
        edge_bgr = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    else:
        edge_bgr = edge
    if edge_bgr.shape[:2] != frame_bgr.shape[:2]:
        edge_bgr = cv2.resize(edge_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))

    canvas = np.hstack([frame_bgr, edge_bgr])
    h, w = canvas.shape[:2]
    overlay_h = min(142, h)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, overlay_h), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0)

    async_status = "pending" if meta.get("slow_pending") else "idle"
    if meta.get("async_result", {}).get("applied"):
        async_status = "applied"
    elif meta.get("async_result", {}).get("rejected"):
        async_status = "rejected"

    lines = [
        f"scene: {meta['scene']} | gen={meta.get('plan_generation', 0)}",
        f"plan: {meta['planner_source']} | slow={meta['slow_trigger']} requested={meta.get('slow_requested')} async={async_status}",
        f"cache_hit={meta.get('cache_hit')} | fast_adjust={meta.get('fast_adjusted')} | change={meta['change_score']:.3f}",
        f"edge_density: {float(meta['feedback'].get('edge_density', 0.0)):.4f}",
    ]
    if meta.get("plan"):
        p = meta["plan"]
        pre = ",".join(x["op"] for x in p.get("preprocess", [])) or "bypass"
        edge_name = p.get("edge_detector", {}).get("op", "?")
        post = ",".join(x["op"] for x in p.get("postprocess", [])) or "bypass"
        lines.append(f"pipeline: {pre} -> {edge_name} -> {post}")

    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (12, 24 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
