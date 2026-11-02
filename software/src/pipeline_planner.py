import json
from pathlib import Path
from .schemas import ImageState, Operator, PipelinePlan

# ===== Default calibrated thresholds =====
DEFAULT_THRESHOLDS = {
    "noise_level": 0.2389,
    "impulse_noise": 0.2825,
    "blur_level": 0.1834,
    "low_light": 0.3776,
    "low_contrast": 0.3598,
}

# ===== Helper functions =====
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def relative_strength(score: float, threshold: float) -> float:
    """
    Convert a raw CNN output into a calibrated relative score.
    Interpretation:
        ratio < 1.0  -> not detected
        ratio >= 1.0 -> detected
    IMPORTANT:
    Ratios are mainly used for parameter adaptation.
    They should not be interpreted as probabilities.
    """
    return float(score / max(threshold, 1e-8))

# ===== Rule Planner =====
class RulePlanner:
    def __init__(self, threshold_path: str | Path = "models/image_analyzer_thresholds.json"):
        """
        Adaptive rule-based planner.
        The planner uses:
            1. Image Analyzer V2 CNN scores
            2. Calibrated per-task thresholds
            3. Classical image features
            4. Runtime constraints
        """
        self.threshold_path = Path(threshold_path)
        self.thresholds = self._load_thresholds()

    # ----- Threshold loading -----
    def _load_thresholds(self) -> dict[str, float]:
        # Fall back to embedded calibrated thresholds if the JSON file is unavailable
        if not self.threshold_path.exists():
            print("[RulePlanner] Warning: threshold file not found:")
            print(f"{self.threshold_path}")
            print("[RulePlanner] Using built-in calibrated thresholds.")
            return dict(DEFAULT_THRESHOLDS)
        try:
            with open(self.threshold_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("thresholds", {})
            thresholds = {}
            for name, fallback in DEFAULT_THRESHOLDS.items():
                value = raw.get(name, fallback)
                thresholds[name] = float(value)
            return thresholds
        except Exception as exc:
            print("[RulePlanner] Warning: failed to load threshold file:")
            print(f"{exc}")
            print("[RulePlanner] Using built-in calibrated thresholds.")
            return dict(DEFAULT_THRESHOLDS)

    # ----- Detection helper -----
    def _detected(self, state: ImageState, name: str) -> bool:
        score = float(getattr(state, name))
        threshold = self.thresholds[name]
        return score >= threshold

    # ----- Relative severity helper -----
    def _ratio(self, state: ImageState, name: str) -> float:
        score = float(getattr(state, name))
        threshold = self.thresholds[name]
        return relative_strength(score, threshold)

    # ----- Noise preprocessing -----
    def _plan_noise(self, state: ImageState) -> tuple[Operator | None, str | None]:
        impulse_detected = self._detected(state, "impulse_noise")
        noise_detected = self._detected(state, "noise_level")
        # Impulse noise has priority over general noise.
        # Median filtering is more appropriate for salt-and-pepper / impulse noise.
        if impulse_detected:
            ratio = self._ratio(state, "impulse_noise")
            if ratio < 1.5:
                kernel = 3
            elif ratio < 2.2:
                kernel = 3
            else:
                kernel = 5
            op = Operator("median", {"kernel":kernel})
            reason = (
                f"impulse_noise detected (score={state.impulse_noise:.4f}, "
                f"threshold={self.thresholds['impulse_noise']:.4f}, "
                f"ratio={ratio:.2f}x) -> median {kernel}x{kernel}"
            )
            return op, reason
        # General / sensor noise
        if noise_detected:
            ratio = self._ratio(state, "noise_level")
            if ratio < 1.4:
                kernel = 3
                sigma = 0.8
            elif ratio < 2.0:
                kernel = 3
                sigma = 1.0

            else:
                kernel = 5
                sigma = 1.2
            op = Operator("gaussian", {"kernel": kernel, "sigma": sigma})
            reason = (
                f"noise_level detected (score={state.noise_level:.4f}, "
                f"threshold={self.thresholds['noise_level']:.4f}, "
                f"ratio={ratio:.2f}x) -> gaussian {kernel}x{kernel}, "
                f"sigma={sigma:.1f}"
            )
            return op, reason
        return None, None

    # ----- Low-light preprocessing -----
    def _plan_low_light(self, state: ImageState) -> tuple[Operator | None, str | None]:
        if not self._detected(state, "low_light"):
            return None, None
        ratio = self._ratio(state, "low_light")
        # Gamma parameter adaptation
        if ratio < 1.3:
            gamma = 1.25
        elif ratio < 1.6:
            gamma = 1.40
        elif ratio < 2.0:
            gamma = 1.60
        else:
            gamma = 1.80
        op = Operator("gamma", {"gamma": gamma})
        reason = (
            f"low_light detected (score={state.low_light:.4f}, "
            f"threshold={self.thresholds['low_light']:.4f}, "
            f"ratio={ratio:.2f}x) -> gamma correction gamma={gamma:.2f}"
        )
        return op, reason

    # ----- Low-contrast preprocessing -----
    def _plan_low_contrast(self, state: ImageState, constraints: dict) -> tuple[Operator | None, str | None]:
        if not self._detected(state, "low_contrast"):
            return None, None
        ratio = self._ratio(state, "low_contrast")
        # Adapt CLAHE strength
        target_backend = str(constraints.get("backend", "software")).lower()
        if target_backend == "fpga":
            if ratio < 1.4:
                clip_limit = 0.25
            elif ratio < 1.8:
                clip_limit = 0.50
            else:
                clip_limit = 0.75
        else:
            if ratio < 1.4:
                clip_limit = 1.5
            elif ratio < 1.8:
                clip_limit = 2.0
            else:
                clip_limit = 2.5
        op = Operator("clahe", {"clip_limit": clip_limit})
        reason = (
            f"low_contrast detected (score={state.low_contrast:.4f}, "
            f"threshold={self.thresholds['low_contrast']:.4f}, "
            f"ratio={ratio:.2f}x) -> CLAHE clip_limit={clip_limit:.1f}"
        )
        return op, reason

    # ----- Blur preprocessing -----
    def _plan_blur(self, state: ImageState) -> tuple[Operator | None, str | None]:
        if not self._detected(state, "blur_level"):
            return None, None
        ratio = self._ratio(state, "blur_level")
        # Adaptive unsharp-mask strength
        if ratio < 1.3:
            strength = 0.30
        elif ratio < 1.7:
            strength = 0.50
        elif ratio < 2.2:
            strength = 0.70
        else:
            strength = 0.90
        op = Operator("sharpen", {"strength": strength})
        reason = (
            f"blur_level detected (score={state.blur_level:.4f}, "
            f"threshold={self.thresholds['blur_level']:.4f}, "
            f"ratio={ratio:.2f}x) -> sharpen strength={strength:.2f}"
        )
        return op, reason

    # ----- Edge detector planning -----
    def _plan_edge(self, state: ImageState, constraints: dict) -> tuple[Operator, str]:
        target_backend = str(constraints.get("backend", "software")).lower()
        allow_hed = bool(constraints.get("allow_hed", True) and target_backend != "fpga")
        latency = constraints.get("latency_priority", "balanced")
        quality = constraints.get("quality_priority", "balanced")
        resource = constraints.get("resource_budget", "medium")
        # HED
        if (
            allow_hed
            and quality == "high"
            and latency != "high"
            and resource != "low"
            and state.image_complexity >= 0.72
            and state.texture_complexity >= 0.55
        ):
            return Operator("hed", {}), f"complex/texture-rich image + high quality preference -> HED"
        # Strict real-time / resource mode
        if latency == "high" or resource == "low":
            if not self._detected(state, "noise_level") and not self._detected(state, "impulse_noise") and state.image_complexity < 0.45:
                threshold = self._adaptive_roberts_threshold(state)
                return Operator("roberts", {"threshold": threshold}), f"clean/simple image + strict latency/resource -> Roberts threshold={threshold}"
            threshold = self._adaptive_sobel_threshold(state)
            return Operator("sobel", {"threshold": threshold}), f"real-time/resource priority -> Sobel threshold={threshold}"
        # Quality / complex image
        if quality == "high" or state.image_complexity >= 0.60 or state.edge_density >= 0.55:
            low, high = self._adaptive_canny_thresholds(state)
            return Operator("canny", {"low_threshold": low, "high_threshold": high}), f"quality/complexity preference -> Canny thresholds=({low}, {high})"
        # Balanced default
        threshold = self._adaptive_sobel_threshold(state)
        return Operator("sobel", {"threshold": threshold}), f"balanced default -> Sobel threshold={threshold}"

    # ----- Adaptive edge thresholds -----
    def _adaptive_sobel_threshold(self, state: ImageState) -> int:
        threshold = 80.0
        # Sparse edges: reduce threshold slightly
        if state.edge_density < 0.10:
            threshold -= 15.0
        elif state.edge_density < 0.25:
            threshold -= 8.0
        # Dense edge maps: increase threshold to suppress weak responses
        elif state.edge_density > 0.60:
            threshold += 10.0
        # Noise can generate false edges
        if self._detected(state, "noise_level"):
            ratio = self._ratio(state, "noise_level")
            threshold += min(15.0, 5.0 * ratio)
        if self._detected(state, "impulse_noise"):
            threshold += 10.0
        threshold = clamp(threshold, 40.0, 130.0)
        return int(round(threshold))

    def _adaptive_roberts_threshold(self, state: ImageState) -> int:
        threshold = 55.0
        if state.edge_density < 0.10:
            threshold -= 10.0
        elif state.edge_density > 0.50:
            threshold += 10.0
        threshold = clamp(threshold, 30.0, 100.0)
        return int(round(threshold))

    def _adaptive_canny_thresholds(self, state: ImageState) -> tuple[int, int]:
        low = 50.0
        high = 120.0
        # Sparse / weak edge structure
        if state.edge_density < 0.10:
            low -= 12.0
            high -= 20.0
        elif state.edge_density < 0.25:
            low -= 5.0
            high -= 10.0
        # Dense edge structure
        elif state.edge_density > 0.60:
            low += 10.0
            high += 20.0
        # Noisy image -> suppress weak false edges
        if self._detected(state, "noise_level"):
            ratio = self._ratio(state, "noise_level")
            adjustment = min(15.0, 5.0 * ratio)
            low += adjustment
            high += adjustment
        low = clamp(low, 20.0, 120.0)
        high = clamp(high, low + 20.0, 220.0)
        return int(round(low)), int(round(high))

    # ----- Post-processing planning -----
    def _plan_postprocess(self, state: ImageState, edge: Operator, max_post: int) -> tuple[list[Operator], list[str]]:
        post = []
        reasons = []
        if max_post <= 0:
            return post, reasons
        noise_detected = self._detected(state, "noise_level")
        impulse_detected = self._detected(state, "impulse_noise")
        if state.edge_density < 0.25 and edge.op in {"canny", "sobel"}:
            post.append(Operator(
                "closing",
                {"kernel": 3, "iterations": 1}
            ))
            reasons.append("sparse edge structure -> morphological closing")
        return post[:max_post], reasons

    # ----- Main planning function -----
    def plan(self, state: ImageState, constraints: dict) -> PipelinePlan:
        # Clamp state to [0, 1]
        state = state.clamp()
        pre: list[Operator] = []
        reasons: list[str] = []
        max_pre = int(constraints.get("max_pre_ops", 3))
        max_post = int(constraints.get("max_post_ops", 2))
        # Noise suppression
        noise_op, noise_reason = self._plan_noise(state)
        if noise_op is not None and len(pre) < max_pre:
            pre.append(noise_op)
            reasons.append(noise_reason)
        # Illumination / contrast
        light_op, light_reason = self._plan_low_light(state)
        contrast_op, contrast_reason = self._plan_low_contrast(state, constraints)
        if light_op is not None and len(pre) < max_pre:
            pre.append(light_op)
            reasons.append(light_reason)
        if contrast_op is not None and len(pre) < max_pre:
            pre.append(contrast_op)
            reasons.append(contrast_reason)
        # Blur / sharpening
        blur_op, blur_reason = self._plan_blur(state)
        if blur_op is not None:
            if len(pre) < max_pre:
                pre.append(blur_op)
                reasons.append(blur_reason)
            else:
                reasons.append(f"blur detected but sharpen skipped because max_pre_ops={max_pre}")
        # Edge detector
        edge, edge_reason = self._plan_edge(state, constraints)
        reasons.append(edge_reason)
        # Post-processing
        post, post_reasons = self._plan_postprocess(state, edge, max_post)
        reasons.extend(post_reasons)
        # Final plan
        return PipelinePlan(
            preprocess=pre[:max_pre],
            edge=edge,
            postprocess=post[:max_post],
            reason="; ".join(reasons)
        )