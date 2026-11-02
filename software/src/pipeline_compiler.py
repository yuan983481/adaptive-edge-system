from dataclasses import dataclass, field
from .hardware_capability import FPGA_MAX_HEIGHT, FPGA_MAX_WIDTH, OPERATOR_IDS, gaussian_effective_sigma
from .pipeline_validator import PipelineValidator
from .schemas import Operator, PipelinePlan

class PipelineCompileError(RuntimeError):
    pass

@dataclass(frozen=True)
class SlotConfig:
    op_name: str
    op_id: int
    p0: int = 0
    p1: int = 0
    p2: int = 0
    p3: int = 0

@dataclass
class CompiledPipeline:
    backend: str
    rows: int
    cols: int
    slots: dict[str, SlotConfig] = field(default_factory=dict)
    registers: dict[str, int] = field(default_factory=dict)
    gamma_lut: list[int] | None = None
    gamma_lut_words: list[int] | None = None
    warnings: list[str] = field(default_factory=list)

def _u8(value: int, label: str) -> int:
    iv = int(value)
    if not (0 <= iv <= 255):
        raise PipelineCompileError(f"{label}={iv} does not fit uint8")
    return iv

def _q0_8(value: float, label: str) -> int:
    q = int(round(float(value) * 256.0))
    if not (0 <= q <= 255):
        raise PipelineCompileError(f"{label}={value} cannot be represented as unsigned Q0.8")
    return q

def _q4_8(value: float, label: str) -> int:
    q = int(round(float(value) * 256.0))
    # HLS: ap_uint<12> sigma = p1 & 0xFFF
    if not (0 <= q <= 0xFFF):
        raise PipelineCompileError(f"{label}={value} cannot be represented as unsigned Q4.8")
    return q

# ===== Gamma LUT =====
def build_gamma_lut(gamma: float) -> list[int]:
    """
    Exactly match current software_ops.py Gamma semantics.
    """
    gamma = float(gamma)
    if gamma <= 0.0:
        raise PipelineCompileError("gamma must be > 0")
    inv = 1.0 / gamma
    return [max(0, min(255, int(((i / 255.0) ** inv) * 255.0))) for i in range(256)]

def pack_gamma_lut(lut: list[int]) -> list[int]:
    if len(lut) != 256:
        raise PipelineCompileError(f"Gamma LUT must have 256 entries, got {len(lut)}")
    words: list[int] = []
    for i in range(0, 256, 4):
        b0 = _u8(lut[i], f"gamma_lut[{i}]")
        b1 = _u8(lut[i + 1], f"gamma_lut[{i + 1}]")
        b2 = _u8(lut[i + 2], f"gamma_lut[{i + 2}]")
        b3 = _u8(lut[i + 3], f"gamma_lut[{i + 3}]")
        words.append(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
    return words

def _bypass() -> SlotConfig:
    return SlotConfig("bypass", OPERATOR_IDS["bypass"])

# ===== Operator encoder =====
def _encode_operator(op: Operator) -> SlotConfig:
    name = op.op
    if name not in OPERATOR_IDS:
        raise PipelineCompileError(f"no FPGA operator ID for {name!r}")
    p0 = 0
    p1 = 0
    p2 = 0
    p3 = 0
    if name == "gamma":
        # Gamma has no scalar register parameter.Actual curve is sent via LUT.
        pass
    elif name == "gaussian":
        p0 = int(op.params.get("kernel", 3))
        p1 = _q4_8(float(op.params.get("sigma", 1.0)), "gaussian.sigma")
    elif name == "median":
        p0 = int(op.params.get("kernel", 3))
    elif name == "clahe":
        p0 = _q0_8(float(op.params.get("clip_limit", 0.25)), "clahe.clip_limit")
        # Hardware currently fixes these values but contract says p1/p2=8.
        p1 = 8
        p2 = 8
    elif name == "sharpen":
        p0 = _q0_8(float(op.params.get("strength", 0.5)), "sharpen.strength")
    elif name in {"roberts", "sobel"}:
        p0 = _u8(int(op.params.get("threshold", 80)), f"{name}.threshold")
    elif name == "canny":
        p0 = _u8(int(op.params.get("low_threshold", 50)), "canny.low_threshold")
        p1 = _u8(int(op.params.get("high_threshold", 120)), "canny.high_threshold")
    elif name in {"opening", "closing"}:
        p0 = int(op.params.get("kernel", 3))
        p1 = int(op.params.get("iterations", 1))
    elif name == "bypass":
        pass
    else:
        raise PipelineCompileError(f"FPGA compiler does not implement {name!r}")
    return SlotConfig(name, OPERATOR_IDS[name], p0, p1, p2, p3)

def _registers_from_slots(rows: int, cols: int, slots: dict[str, SlotConfig]) -> dict[str, int]:
    regs: dict[str, int] = {
        "rows": int(rows),
        "cols": int(cols),
    }
    for (slot_name, slot) in slots.items():
        regs[f"{slot_name}_op"] = slot.op_id
        regs[f"{slot_name}_p0"] = slot.p0
        regs[f"{slot_name}_p1"] = slot.p1
        regs[f"{slot_name}_p2"] = slot.p2
        regs[f"{slot_name}_p3"] = slot.p3
    return regs

# ===== Main compiler =====
def compile_pipeline(plan: PipelinePlan, rows: int, cols: int, *, validator: PipelineValidator | None = None) -> CompiledPipeline:
    rows = int(rows)
    cols = int(cols)
    if not (1 <= rows <= FPGA_MAX_HEIGHT and 1 <= cols <= FPGA_MAX_WIDTH):
        raise PipelineCompileError(f"FPGA image size {cols}x{rows} outside 1..{FPGA_MAX_WIDTH} x 1..{FPGA_MAX_HEIGHT}")
    validator = validator or PipelineValidator()
    result = validator.validate(
        plan,
        max_pre=3,
        max_post=2,
        allow_hed=False,
        target_backend="fpga",
        raise_on_error=False,
        image_rows=rows,
        image_cols=cols,
    )
    if not result.valid:
        raise PipelineCompileError("; ".join(result.errors))
    slots: dict[str, SlotConfig] = {
        "pre0": _bypass(),
        "pre1": _bypass(),
        "pre2": _bypass(),
        "edge": _bypass(),
        "post0": _bypass(),
        "post1": _bypass(),
    }
    # PRE0 / PRE1 / PRE2
    for (index, op) in enumerate(plan.preprocess):
        slots[f"pre{index}"] = _encode_operator(op)
    # EDGE
    slots["edge"] = _encode_operator(plan.edge)
    # POST0 / POST1
    for (index, op) in enumerate(plan.postprocess):
        slots[f"post{index}"] = _encode_operator(op)
    # Gamma uses one shared LUT.
    gamma_values = [float(op.params.get("gamma", 1.0)) for op in plan.preprocess if op.op == "gamma"]
    if gamma_values:
        gamma_lut = build_gamma_lut(gamma_values[0])
        gamma_lut_words = pack_gamma_lut(gamma_lut)
    else:
        gamma_lut = None
        gamma_lut_words = None
    warnings = list(result.warnings)
    for op in plan.preprocess:
        if op.op == "gaussian":
            kernel = int(op.params.get("kernel", 3))
            sigma = float(op.params.get("sigma", 1.0))
            effective = gaussian_effective_sigma(kernel, sigma)
            message = f"compiled Gaussian kernel={kernel}, requested sigma={sigma:g}, effective FPGA preset≈{effective:g}"
            if message not in warnings:
                warnings.append(message)
    return CompiledPipeline(
        backend="fpga",
        rows=rows,
        cols=cols,
        slots=slots,
        registers=_registers_from_slots(rows, cols, slots),
        gamma_lut=gamma_lut,
        gamma_lut_words=gamma_lut_words,
        warnings=warnings,
    )
