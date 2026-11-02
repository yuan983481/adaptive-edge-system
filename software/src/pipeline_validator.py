import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .hardware_capability import FPGA_CAPABILITY, FPGA_CLAHE_TILE_GRID, FPGA_MAX_HEIGHT, FPGA_MAX_WIDTH, gaussian_effective_sigma
from .operator_library import OPERATOR_LIBRARY, STAGE_BY_OPERATOR
from .schemas import Operator, PipelinePlan

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resolved_backend: str = "software"

class PipelineValidationError(ValueError):
    pass

def _parse_operator(raw: dict[str, Any]) -> Operator:
    if not isinstance(raw, dict):
        raise PipelineValidationError("operator must be a JSON object")
    name = str(raw.get("op", "")).strip().lower()
    if not name:
        raise PipelineValidationError("operator is missing 'op'")
    if "params" in raw:
        params = raw["params"]
        if not isinstance(params, dict):
            raise PipelineValidationError(f"{name}.params must be an object")
        params = dict(params)
    else:
        params = {k: v for k, v in raw.items() if k != "op"}
    if name == "canny":
        if "low" in params and "low_threshold" not in params:
            params["low_threshold"] = params.pop("low")
        if "high" in params and "high_threshold" not in params:
            params["high_threshold"] = params.pop("high")
    return Operator(name, params)

class PipelineValidator:
    def __init__(self, skill_path: str | Path | None = None):
        self.skill_path = Path(skill_path) if skill_path else None
        self.skill: dict[str, Any] = {}
        if self.skill_path and self.skill_path.exists():
            try:
                self.skill = json.loads(self.skill_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[PipelineValidator] skill warning: {exc}")

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PipelinePlan:
        if not isinstance(data, dict):
            raise PipelineValidationError("pipeline must be one JSON object")
        edge_raw = data.get("edge_detector") or data.get("edge_detection") or data.get("edge")
        if edge_raw is None:
            raise PipelineValidationError("missing edge_detector")
        pre = data.get("preprocess", [])
        post = data.get("postprocess", [])
        if not isinstance(pre, list) or not isinstance(post, list):
            raise PipelineValidationError("preprocess and postprocess must be lists")
        return PipelinePlan(
            preprocess=[_parse_operator(x) for x in pre],
            edge=_parse_operator(edge_raw),
            postprocess=[_parse_operator(x) for x in post],
            reason=str(data.get("reason", "")),
        )

    @staticmethod
    def _check_value(op_name: str, param_name: str, value: Any, rule: dict[str, Any], errors: list[str],) -> None:
        kind = rule.get("type")
        if kind == "int":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                errors.append(f"{op_name}.{param_name} must be int")
                return
        elif kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{op_name}.{param_name} must be numeric")
                return
        if "choices" in rule and value not in rule["choices"]:
            errors.append(f"{op_name}.{param_name}={value} not in {rule['choices']}")
        if "min" in rule and float(value) < float(rule["min"]):
            errors.append(f"{op_name}.{param_name} < {rule['min']}")
        if "max" in rule and float(value) > float(rule["max"]):
            errors.append(f"{op_name}.{param_name} > {rule['max']}")

    def _check_operator(self, stage: str, op: Operator, errors: list[str]) -> None:
        actual_stage = STAGE_BY_OPERATOR.get(op.op)
        if actual_stage is None:
            errors.append(f"unknown operator: {op.op}")
            return
        if actual_stage != stage:
            errors.append(f"{op.op} belongs to {actual_stage}, not {stage}")
            return
        spec = OPERATOR_LIBRARY[stage][op.op]
        rules = spec.get("params", {})
        for (name, value) in list(op.params.items()):
            if name not in rules:
                errors.append(f"{op.op}: unsupported parameter '{name}'")
                continue
            self._check_value(op.op, name, value, rules[name], errors,)
        # Deterministic execution
        for (name, value) in spec.get("defaults", {}).items():
            op.params.setdefault(name, value)

    @staticmethod
    def _check_fpga_operator(stage: str, op: Operator, errors: list[str], warnings: list[str]) -> None:
        cap = FPGA_CAPABILITY.get(stage, {}).get(op.op, {})
        if not cap.get("supported", False,):
            errors.append(f"FPGA does not support {stage}.{op.op}")
            return
        # kernel
        if "kernel" in op.params and "kernels" in cap:
            kernel = int(op.params["kernel"])
            if kernel not in cap["kernels"]:
                errors.append(f"FPGA {op.op} kernel={kernel} unsupported; allowed={cap['kernels']}")
        # iterations
        if "iterations" in op.params and "iterations" in cap:
            iterations = int(op.params["iterations"])
            if iterations not in cap["iterations"]:
                errors.append(f"FPGA {op.op} iterations={iterations} unsupported; allowed={cap['iterations']}")
        # Gamma
        if op.op == "gamma":
            gamma = float(op.params.get("gamma", 1.0))
            lo, hi = cap["gamma_range"]
            if not (float(lo) <= gamma <= float(hi)):
                errors.append(f"FPGA gamma={gamma} outside [{lo}, {hi}]")
        # Gaussian
        elif op.op == "gaussian":
            sigma = float(op.params.get("sigma", 1.0))
            lo, hi = cap["sigma_input_range"]
            if not (float(lo) <= sigma <= float(hi)):
                errors.append(f"FPGA gaussian sigma={sigma} outside [{lo}, {hi}]")
            elif "kernel" in op.params:
                effective = gaussian_effective_sigma(int(op.params["kernel"]), sigma,)
                if abs(effective - sigma) > 1e-9:
                    warnings.append(f"FPGA gaussian sigma={sigma:g} is discretely quantized to effective sigma≈{effective:g} for kernel={int(op.params['kernel'])}")
        # CLAHE
        elif op.op == "clahe":
            clip = float(op.params.get("clip_limit", 0.25,))
            lo, hi = cap["clip_limit_range"]
            if not (float(lo) <= clip <= float(hi)):
                errors.append(f"FPGA CLAHE clip_limit={clip} outside Q0.8 logical range [{lo}, {hi}]")
            tile_grid = int(op.params.get("tile_grid", FPGA_CLAHE_TILE_GRID))
            if tile_grid not in cap["tile_grid"]:
                errors.append(f"FPGA CLAHE tile_grid={tile_grid} unsupported; allowed={cap['tile_grid']}")
        # Sharpen
        elif op.op == "sharpen":
            strength = float(op.params.get("strength", 0.5))
            lo, hi = cap["strength_range"]
            if not (float(lo) <= strength <= float(hi)):
                errors.append(f"FPGA sharpen strength={strength} outside [{lo}, {hi}]")
        # Sobel / Roberts
        elif op.op in {"sobel", "roberts"}:
            threshold = int(op.params.get("threshold", 80))
            lo, hi = cap["threshold_range"]
            if not (int(lo) <= threshold <= int(hi)):
                errors.append(f"FPGA {op.op} threshold={threshold} outside [{lo}, {hi}]")
        # Canny
        elif op.op == "canny":
            low = int(op.params.get("low_threshold", 50))
            high = int(op.params.get("high_threshold", 120))
            if not (0 <= low <= high <= 255):
                errors.append("FPGA Canny requires 0 <= low_threshold <= high_threshold <= 255")

    def validate(
        self,
        plan: PipelinePlan,
        max_pre: int = 3,
        max_post: int = 2,
        allow_hed: bool = False,
        target_backend: str = "software",
        raise_on_error: bool = True,
        image_rows: int | None = None,
        image_cols: int | None = None,
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        backend = str(target_backend).lower()
        if backend not in {"software", "fpga"}:
            errors.append(f"unknown target_backend: {target_backend}")

        # ----- Generic project validation -----
        if len(plan.preprocess) > int(max_pre):
            errors.append(f"too many preprocess operators: {len(plan.preprocess)} > {max_pre}")
        if len(plan.postprocess) > int(max_post):
            errors.append(f"too many postprocess operators: {len(plan.postprocess)} > {max_post}")
        for op in plan.preprocess:
            self._check_operator("preprocess", op, errors)
        self._check_operator("edge_detection", plan.edge, errors)
        for op in plan.postprocess:
            self._check_operator("postprocess", op, errors)
        if plan.edge.op == "hed" and not allow_hed:
            errors.append("HED is disabled by allow_hed=False")
        if plan.edge.op == "canny":
            lo = int(plan.edge.params.get("low_threshold", 50))
            hi = int(plan.edge.params.get("high_threshold", 120))
            # Keep common software/FPGA planner contract strict.
            if not (0 <= lo <= hi <= 255):
                errors.append("Canny requires 0 <= low_threshold <= high_threshold <= 255")
        # Sharpen-first is NOT rejected.It is a planning preference, not an executability rule.
        if plan.edge.op == "canny" and any(x.op == "opening" for x in plan.postprocess):
            warnings.append("Opening after Canny may erase valid thin edges.")

        # ----- FPGA-specific validation -----
        if backend == "fpga":
            pipe_cap = FPGA_CAPABILITY["pipeline"]
            if len(plan.preprocess) > int(pipe_cap["max_pre_ops"]):
                errors.append(f"FPGA has only {pipe_cap['max_pre_ops']} preprocess slots")
            if len(plan.postprocess) > int(pipe_cap["max_post_ops"]):
                errors.append(f"FPGA has only {pipe_cap['max_post_ops']} postprocess slots")
            for op in plan.preprocess:
                self._check_fpga_operator("preprocess", op, errors, warnings)
            self._check_fpga_operator("edge_detection", plan.edge, errors, warnings)
            for op in plan.postprocess:
                self._check_fpga_operator("postprocess", op, errors, warnings)
            # FPGA has only one shared Gamma LUT.
            gamma_values = [float(op.params.get("gamma", 1.0)) for op in plan.preprocess if op.op == "gamma"]
            if len(gamma_values) > 1:
                first = gamma_values[0]
                if any(abs(first - x) > 1e-12 for x in gamma_values[1:]):
                    errors.append("FPGA has one shared Gamma LUT; multiple Gamma stages must use the same gamma value")
            # FPGA dimensions
        #
        # Large logical frames are allowed because the FPGA runtime now
        # processes them through the overlapping 64x64 tiled executor.
        # FPGA_MAX_WIDTH / FPGA_MAX_HEIGHT describe only the direct
        # whole-frame path, not the maximum logical image size.
        if image_rows is not None and image_cols is not None:
            rows = int(image_rows)
            cols = int(image_cols)

            if rows < 1 or cols < 1:
                errors.append(
                    f"FPGA image dimensions must be positive; got {cols}x{rows}"
                )
            else:
                direct_fit = (
                    rows <= FPGA_MAX_HEIGHT
                    and cols <= FPGA_MAX_WIDTH
                )

                if not direct_fit:
                    warnings.append(
                        f"FPGA image size {cols}x{rows} exceeds direct "
                        f"Phase-3 limit {FPGA_MAX_WIDTH}x{FPGA_MAX_HEIGHT}; "
                        "using tiled FPGA execution"
                    )

                # Direct whole-frame CLAHE requires dimensions divisible by
                # the hardware tile grid. Large logical frames use the tiled
                # executor, whose hardware transactions are padded to 64x64.
                if (
                    direct_fit
                    and any(op.op == "clahe" for op in plan.preprocess)
                ):
                    if (
                        rows % FPGA_CLAHE_TILE_GRID != 0
                        or cols % FPGA_CLAHE_TILE_GRID != 0
                    ):
                        errors.append(
                            "FPGA CLAHE requires rows and cols to be "
                            f"multiples of {FPGA_CLAHE_TILE_GRID}; "
                            f"got {cols}x{rows}"
                        )

        resolved_backend = "hed" if plan.edge.op == "hed" else backend
        result = ValidationResult(
            valid=not errors,
            errors=errors,
            warnings=warnings,
            resolved_backend=resolved_backend,
        )
        if raise_on_error and not result.valid:
            raise PipelineValidationError("; ".join(result.errors))
        return result

def validate_pipeline(
    plan: PipelinePlan,
    *,
    max_pre: int = 3,
    max_post: int = 2,
    allow_hed: bool = False,
    target_backend: str = "software",
    image_rows: int | None = None,
    image_cols: int | None = None,
) -> ValidationResult:
    return PipelineValidator().validate(
        plan,
        max_pre=max_pre,
        max_post=max_post,
        allow_hed=allow_hed,
        target_backend=target_backend,
        raise_on_error=False,
        image_rows=image_rows,
        image_cols=image_cols,
    )