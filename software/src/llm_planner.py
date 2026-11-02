import json
import os
from pathlib import Path
from typing import Any
import requests
from .hardware_capability import FPGA_CAPABILITY
from .llm_contract import SYSTEM_RULES
from .operator_library import OPERATOR_LIBRARY
from .pipeline_validator import PipelineValidationError, PipelineValidator
from .schemas import ImageState, PipelinePlan
from .pipeline_planner import DEFAULT_THRESHOLDS

class LLMPlanner:
    """
    Runtime LLM planner using an OpenAI-compatible chat-completions endpoint.
    Required environment variables:LLM_BASE_URL/LLM_MODEL
    Optional:LLM_API_KEY/LLM_TIMEOUT_S/LLM_TEMPERATURE
    """
    def __init__(
        self,
        skill_path: str | Path | None = None,
        validator: PipelineValidator | None = None,
        timeout_s: float | None = None,
        max_repair_attempts: int = 1,
        threshold_path: str | Path = "models/image_analyzer_thresholds.json",
    ):
        self.skill_path = Path(skill_path) if skill_path else None
        self.skill = self._load_skill()
        self.threshold_path = Path(threshold_path)
        self.thresholds = self._load_thresholds()
        self.validator = validator if validator is not None else PipelineValidator(self.skill_path)
        self.base_url = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY", "").strip()
        self.model = os.environ.get("LLM_MODEL", "").strip()
        self.timeout_s = float(timeout_s if timeout_s is not None else os.environ.get("LLM_TIMEOUT_S", "30"))
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
        self.max_repair_attempts = max(0, int(max_repair_attempts))
        if not self.base_url:
            raise RuntimeError("LLM_BASE_URL is required when planner=llm")
        if not self.model:
            raise RuntimeError("LLM_MODEL is required when planner=llm")

    def _load_skill(self) -> dict[str, Any]:
        if self.skill_path is None or not self.skill_path.exists():
            return {}
        try:
            return json.loads(self.skill_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"failed to load image_pipeline_skill.json: {exc}") from exc

    def _load_thresholds(self) -> dict[str, float]:
        # Match RulePlanner behavior: the packaged project may intentionally
        # omit the calibration JSON, so the LLM/Agent path must remain usable.
        if not self.threshold_path.exists():
            print(f"[LLMPlanner] threshold file not found: {self.threshold_path}; using built-in calibrated thresholds")
            return dict(DEFAULT_THRESHOLDS)
        try:
            data = json.loads(self.threshold_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[LLMPlanner] failed to load thresholds ({exc}); using built-in calibrated thresholds")
            return dict(DEFAULT_THRESHOLDS)
        thresholds = data.get("thresholds")
        if not isinstance(thresholds, dict):
            print("[LLMPlanner] invalid threshold JSON; using built-in calibrated thresholds")
            return dict(DEFAULT_THRESHOLDS)
        result: dict[str, float] = {}
        for name, fallback in DEFAULT_THRESHOLDS.items():
            result[name] = float(thresholds.get(name, fallback))
        return result

    def _endpoint(self) -> str:
        suffix = "/chat/completions"
        if self.base_url.endswith(suffix):
            return self.base_url
        return self.base_url + suffix

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, messages: list[dict[str, str]]) -> str:
        response = requests.post(
            self._endpoint(),
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "stream": False,
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
                "max_tokens": 1024,
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected DeepSeek response format") from exc
        if not isinstance(content, str):
            raise RuntimeError("DeepSeek response content is not text")
        return content.strip()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        # Accept accidental ```json ... ``` fences.
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start < 0 or end < start:
                raise RuntimeError("LLM did not return a JSON object")
            try:
                data = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"cannot parse LLM JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("LLM output must be exactly one JSON object")
        return data

    @staticmethod
    def _constraints(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "allow_hed": bool(raw.get("allow_hed", False)),
            "latency_priority": str(raw.get("latency_priority", "balanced")),
            "quality_priority": str(raw.get("quality_priority", "balanced")),
            "resource_budget": str(raw.get("resource_budget", "medium")),
            "max_pre_ops": int(raw.get("max_pre_ops", 3)),
            "max_post_ops": int(raw.get("max_post_ops", 2)),
            "target_backend": str(raw.get("backend", "software")).lower(),
            "image_rows": int(raw["image_rows"]) if raw.get("image_rows") is not None else None,
            "image_cols": int(raw["image_cols"]) if raw.get("image_cols") is not None else None,
        }

    def _payload(self, state: ImageState, constraints: dict[str, Any]) -> dict[str, Any]:
        # Raw image state
        values = state.to_dict()
        degradation_names = [
            "noise_level",
            "impulse_noise",
            "blur_level",
            "low_light",
            "low_contrast",
        ]
        # Calibrated degradation assessment
        degradation_assessment = {}
        for name in degradation_names:
            score = float(values.get(name, 0.0))
            threshold = float(self.thresholds[name])
            ratio = score / threshold if threshold > 0 else 0.0
            detected = score >= threshold
            degradation_assessment[name] = {
                "score": score,
                "threshold": threshold,
                "ratio": ratio,
                "detected": detected,
            }
        # Rank by calibrated ratio, NOT raw CNN score.
        ranked = sorted(
            degradation_assessment.items(),
            key=lambda item: item[1]["ratio"],
            reverse=True,
        )
        # Only these are actually detected.
        active_degradations = [{"name": name, **info} for name, info in ranked if info["detected"]]
        # Full ranking is retained for explanation/debugging.
        ranked_degradations = [{"name": name, **info} for name, info in ranked]
        # Runtime constraints
        resolved_constraints = self._constraints(constraints)
        # Base LLM payload
        payload: dict[str, Any] = {
            "image_state": values,
            "degradation_assessment": degradation_assessment,
            "active_degradations": active_degradations,
            "ranked_degradations": ranked_degradations,
            "constraints": resolved_constraints,
            "operator_library": OPERATOR_LIBRARY,
            "image_pipeline_skill": self.skill,
        }
        # FPGA capability is relevant ONLY when FPGA is the requested target backend.
        if resolved_constraints["target_backend"] == "fpga":
            payload["fpga_capability"] = FPGA_CAPABILITY
        print("\n=== LLM Degradation Assessment ===")
        print(json.dumps(degradation_assessment, indent=2, ensure_ascii=False))
        return payload

    def _decode_and_validate(self, text: str, constraints: dict[str, Any]) -> PipelinePlan:
        data = self._extract_json(text)
        plan = self.validator.from_dict(data)
        c = self._constraints(constraints)
        self.validator.validate(
            plan,
            max_pre=c["max_pre_ops"],
            max_post=c["max_post_ops"],
            allow_hed=c["allow_hed"],
            target_backend=c["target_backend"],
            image_rows=c.get("image_rows"),
            image_cols=c.get("image_cols"),
        )
        return plan

    def _repair(self, state: ImageState, constraints: dict[str, Any], bad_output: str, error: Exception) -> str:
        payload = self._payload(state, constraints)
        payload["previous_invalid_output"] = bad_output
        payload["validation_error"] = str(error)
        payload["instruction"] = "Repair the previous pipeline. Return valid JSON only."
        return self._request(
            [
                {"role": "system", "content": SYSTEM_RULES},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ]
        )

    def refine(
        self,
        state: ImageState,
        constraints: dict[str, Any],
        *,
        previous_plan: PipelinePlan,
        execution_feedback: dict[str, Any],
        assessment: str,
        iteration: int,
    ) -> PipelinePlan:
        """Ask DeepSeek to revise a *validated, actually executed* pipeline.

        This is the key closed-loop step that turns one-shot LLM planning into
        an agentic plan -> execute -> measure -> re-plan workflow.
        """
        payload = self._payload(state.clamp(), constraints)
        payload.update({
            "agent_iteration": int(iteration),
            "previous_plan": previous_plan.to_dict(),
            "execution_feedback": execution_feedback,
            "feedback_assessment": str(assessment),
            "instruction": (
                "The previous pipeline was already validated and executed. "
                "Use the measured execution_feedback to revise it only when justified. "
                "Prefer a small parameter change over adding operators. "
                "Return exactly one valid pipeline JSON object using the normal schema."
            ),
        })
        raw = self._request([
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ])
        last_error: Exception | None = None
        for attempt in range(self.max_repair_attempts + 1):
            try:
                return self._decode_and_validate(raw, constraints)
            except (PipelineValidationError, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_repair_attempts:
                    break
                raw = self._repair(state, constraints, raw, exc)
        raise RuntimeError(
            "LLM agent refinement returned an invalid pipeline"
            + (f": {last_error}" if last_error else "")
        )

    def plan(self, state: ImageState, constraints: dict[str, Any]) -> PipelinePlan:
        state = state.clamp()
        raw = self._request(
            [
                {"role": "system", "content": SYSTEM_RULES},
                {
                    "role": "user",
                    "content": json.dumps(self._payload(state, constraints), ensure_ascii=False),
                },
            ]
        )
        last_error: Exception | None = None
        for attempt in range(self.max_repair_attempts + 1):
            try:
                return self._decode_and_validate(raw, constraints)
            except (PipelineValidationError, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_repair_attempts:
                    break
                raw = self._repair(state, constraints, raw, exc)
        raise RuntimeError("LLM planner returned an invalid pipeline" + (f": {last_error}" if last_error else ""))