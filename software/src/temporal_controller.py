from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy

from .pipeline_planner import DEFAULT_THRESHOLDS
from .schemas import ImageState, PipelinePlan


_STATE_FIELDS = (
    "noise_level",
    "impulse_noise",
    "blur_level",
    "low_light",
    "low_contrast",
    "edge_density",
    "image_complexity",
    "texture_complexity",
)


@dataclass
class TemporalDecision:
    raw_state: ImageState
    filtered_state: ImageState
    scene: str
    scene_changed: bool
    change_score: float
    slow_trigger: bool
    slow_reason: str


@dataclass
class FastLoopController:
    """Deterministic per-frame controller used by the V4 fast loop.

    Responsibilities:
      * EMA smoothing of image-state values.
      * Hysteresis scene classification.
      * Event generation for the asynchronous slow loop.
      * Conservative local edge-threshold tuning with persistence + cooldown.
    """

    ema_alpha: float = 0.25
    enter_factor: float = 1.05
    exit_factor: float = 0.85
    scene_change_threshold: float = 0.18
    min_replan_interval_frames: int = 30
    periodic_replan_frames: int = 300
    feedback_bad_frames: int = 5
    edge_density_min: float = 0.015
    edge_density_max: float = 0.25
    max_tiny_component_ratio: float = 0.65
    threshold_step: int = 4

    # V4: do not change the detector on a single abnormal frame.
    fast_adjust_bad_frames: int = 3
    # V4: after an adjustment, wait before applying another one.
    fast_adjust_cooldown_frames: int = 5

    _ema: dict[str, float] | None = field(default=None, init=False)
    _active: dict[str, bool] = field(default_factory=dict, init=False)
    _scene: str = field(default="normal", init=False)
    _last_slow_frame: int = field(default=-10**9, init=False)
    _bad_feedback_streak: int = field(default=0, init=False)
    _fast_bad_streak: int = field(default=0, init=False)
    _last_fast_adjust_frame: int = field(default=-10**9, init=False)

    def __post_init__(self) -> None:
        self.ema_alpha = float(min(1.0, max(0.01, self.ema_alpha)))
        self.min_replan_interval_frames = max(1, int(self.min_replan_interval_frames))
        self.periodic_replan_frames = max(self.min_replan_interval_frames, int(self.periodic_replan_frames))
        self.feedback_bad_frames = max(1, int(self.feedback_bad_frames))
        self.threshold_step = max(1, int(self.threshold_step))
        self.fast_adjust_bad_frames = max(1, int(self.fast_adjust_bad_frames))
        self.fast_adjust_cooldown_frames = max(0, int(self.fast_adjust_cooldown_frames))
        self._active = {name: False for name in DEFAULT_THRESHOLDS}

    def _filter(self, state: ImageState) -> ImageState:
        current = state.to_dict()
        if self._ema is None:
            self._ema = {k: float(current[k]) for k in _STATE_FIELDS}
        else:
            a = self.ema_alpha
            for k in _STATE_FIELDS:
                self._ema[k] = a * float(current[k]) + (1.0 - a) * self._ema[k]
        return ImageState(**self._ema).clamp()

    def _update_hysteresis(self, filtered: ImageState) -> str:
        for name, threshold in DEFAULT_THRESHOLDS.items():
            value = float(getattr(filtered, name))
            if self._active[name]:
                if value < float(threshold) * self.exit_factor:
                    self._active[name] = False
            else:
                if value >= float(threshold) * self.enter_factor:
                    self._active[name] = True

        labels: list[str] = []
        if self._active.get("low_light"):
            labels.append("low_light")
        if self._active.get("low_contrast"):
            labels.append("low_contrast")
        if self._active.get("impulse_noise"):
            labels.append("impulse_noise")
        elif self._active.get("noise_level"):
            labels.append("noise")
        if self._active.get("blur_level"):
            labels.append("blur")
        return "+".join(labels) if labels else "normal"

    def observe(
        self,
        state: ImageState,
        *,
        frame_index: int,
        previous_feedback: dict[str, float] | None = None,
    ) -> TemporalDecision:
        previous_filtered = None if self._ema is None else dict(self._ema)
        filtered = self._filter(state)

        if previous_filtered is None:
            change_score = 1.0
        else:
            change_score = max(
                abs(float(getattr(filtered, k)) - float(previous_filtered[k]))
                for k in _STATE_FIELDS
            )

        old_scene = self._scene
        self._scene = self._update_hysteresis(filtered)
        scene_changed = self._scene != old_scene

        feedback_bad = self._feedback_is_bad(previous_feedback)
        self._bad_feedback_streak = self._bad_feedback_streak + 1 if feedback_bad else 0

        since_slow = int(frame_index) - self._last_slow_frame
        cooldown_ok = since_slow >= self.min_replan_interval_frames

        trigger = False
        reason = "reuse cached pipeline"
        if frame_index == 0 or self._last_slow_frame < 0:
            trigger = True
            reason = "initial plan"
        # V4: a hysteresis-confirmed scene transition bypasses the generic
        # replan cooldown. This fixes delayed blur->normal / dark->normal recovery.
        elif scene_changed:
            trigger = True
            reason = f"scene changed: {old_scene} -> {self._scene}"
        elif cooldown_ok and change_score >= self.scene_change_threshold:
            trigger = True
            reason = f"feature change score={change_score:.3f}"
        elif cooldown_ok and self._bad_feedback_streak >= self.feedback_bad_frames:
            trigger = True
            reason = f"quality feedback bad for {self._bad_feedback_streak} frames"
        elif since_slow >= self.periodic_replan_frames:
            trigger = True
            reason = f"periodic re-plan after {since_slow} frames"

        if trigger:
            self._last_slow_frame = int(frame_index)
            self._bad_feedback_streak = 0

        return TemporalDecision(
            raw_state=state,
            filtered_state=filtered,
            scene=self._scene,
            scene_changed=scene_changed,
            change_score=float(change_score),
            slow_trigger=trigger,
            slow_reason=reason,
        )

    def _feedback_is_bad(self, feedback: dict[str, float] | None) -> bool:
        if not feedback:
            return False
        density = float(feedback.get("edge_density", 0.0))
        tiny = float(feedback.get("tiny_component_ratio", 0.0))
        if density < self.edge_density_min or density > self.edge_density_max:
            return True
        return tiny > self.max_tiny_component_ratio and density > self.edge_density_min

    def _feedback_direction(self, feedback: dict[str, float] | None) -> tuple[int, str]:
        if not feedback:
            return 0, "no fast-loop correction"
        density = float(feedback.get("edge_density", 0.0))
        tiny = float(feedback.get("tiny_component_ratio", 0.0))
        if density < self.edge_density_min:
            return -1, "sparse edges -> lower detector threshold"
        if density > self.edge_density_max:
            return 1, "dense edges -> raise detector threshold"
        if tiny > self.max_tiny_component_ratio:
            return 1, "fragmented edges -> raise detector threshold"
        return 0, "feedback inside fast-loop window"

    def tune_plan(
        self,
        plan: PipelinePlan,
        feedback: dict[str, float] | None,
        *,
        frame_index: int,
    ) -> tuple[PipelinePlan, str, bool]:
        """Apply a small deterministic detector correction.

        V4 only adjusts after `fast_adjust_bad_frames` consecutive abnormal
        observations and then enforces `fast_adjust_cooldown_frames` before the
        next adjustment. Structural changes remain a slow-loop responsibility.
        """
        direction, reason = self._feedback_direction(feedback)
        if direction == 0:
            self._fast_bad_streak = 0
            return plan, reason, False

        self._fast_bad_streak += 1
        if self._fast_bad_streak < self.fast_adjust_bad_frames:
            return (
                plan,
                f"{reason}; waiting persistence {self._fast_bad_streak}/{self.fast_adjust_bad_frames}",
                False,
            )

        since_adjust = int(frame_index) - self._last_fast_adjust_frame
        if since_adjust < self.fast_adjust_cooldown_frames:
            return (
                plan,
                f"{reason}; cooldown {since_adjust}/{self.fast_adjust_cooldown_frames} frames",
                False,
            )

        updated = deepcopy(plan)
        step = self.threshold_step * direction
        op = updated.edge
        if op.op == "canny":
            low = int(op.params.get("low_threshold", 50)) + step
            high = int(op.params.get("high_threshold", 120)) + 2 * step
            low = max(0, min(254, low))
            high = max(low + 1, min(255, high))
            op.params["low_threshold"] = low
            op.params["high_threshold"] = high
        elif op.op in {"sobel", "roberts"}:
            threshold = int(op.params.get("threshold", 80)) + step
            op.params["threshold"] = max(0, min(255, threshold))
        else:
            self._fast_bad_streak = 0
            return plan, f"{op.op} has no fast-loop threshold tuning", False

        updated.reason = (updated.reason + "; " if updated.reason else "") + f"fast-loop: {reason}"
        self._last_fast_adjust_frame = int(frame_index)
        self._fast_bad_streak = 0
        return updated, reason, True
