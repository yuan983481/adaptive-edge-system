from typing import Any

# ===== Phase-3 FPGA contract =====
FPGA_CONTRACT_VERSION = {
    "major": 1,
    "minor": 0,
}
FPGA_EXPECTED_CAPABILITY = 0x010007FF
FPGA_MAX_WIDTH = 640
FPGA_MAX_HEIGHT = 480
FPGA_CLAHE_TILE_GRID = 8

# ===== Operator IDs =====
OPERATOR_IDS: dict[str, int] = {
    "bypass": 0,
    "gamma": 1,
    "gaussian": 2,
    "median": 3,
    "clahe": 4,
    "sharpen": 5,
    "roberts": 6,
    "sobel": 7,
    "canny": 8,
    "opening": 9,
    "closing": 10,
}

# ===== CTRL AXI-Lite relative offsets =====
CTRL_REGISTERS: dict[str, int] = {
    "ctrl": 0x000,
    "gier": 0x004,
    "ip_ier": 0x008,
    "ip_isr": 0x00C,
    "rows": 0x010,
    "cols": 0x018,
    "pre0_op": 0x020,
    "pre0_p0": 0x028,
    "pre0_p1": 0x030,
    "pre0_p2": 0x038,
    "pre0_p3": 0x040,
    "pre1_op": 0x048,
    "pre1_p0": 0x050,
    "pre1_p1": 0x058,
    "pre1_p2": 0x060,
    "pre1_p3": 0x068,
    "pre2_op": 0x070,
    "pre2_p0": 0x078,
    "pre2_p1": 0x080,
    "pre2_p2": 0x088,
    "pre2_p3": 0x090,
    "edge_op": 0x098,
    "edge_p0": 0x0A0,
    "edge_p1": 0x0A8,
    "edge_p2": 0x0B0,
    "edge_p3": 0x0B8,
    "post0_op": 0x0C0,
    "post0_p0": 0x0C8,
    "post0_p1": 0x0D0,
    "post0_p2": 0x0D8,
    "post0_p3": 0x0E0,
    "post1_op": 0x0E8,
    "post1_p0": 0x0F0,
    "post1_p1": 0x0F8,
    "post1_p2": 0x100,
    "post1_p3": 0x108,
    "status": 0x110,
    "status_ctrl": 0x114,
    "capability": 0x120,
    "capability_ctrl": 0x124,
}

# ===== GAMMA AXI-Lite =====
GAMMA_LUT_BASE = 0x100
GAMMA_LUT_ENTRIES = 256
# 4 x uint8 packed into one uint32
GAMMA_LUT_WORDS = 64

# ===== Hardware error codes =====
FPGA_ERROR_CODES: dict[int, str] = {
    0: "OK",
    1: "ERR_BAD_OP",
    2: "ERR_SLOT_MISMATCH",
    3: "ERR_UNSUPPORTED",
    4: "ERR_PARAM",
    5: "ERR_DIMS",
}

# ===== FPGA capability =====
FPGA_CAPABILITY: dict[str, Any] = {
    "contract_version": FPGA_CONTRACT_VERSION,
    "expected_capability_register": FPGA_EXPECTED_CAPABILITY,
    "device": "XC7Z020-2CLG400I",
    "image": {
        "format": "GRAY8",
        "min_width": 1,
        "max_width": FPGA_MAX_WIDTH,
        "min_height": 1,
        "max_height": FPGA_MAX_HEIGHT,
        "input_stream": "DMA sends rows*cols bytes; hardware rebuilds pixel position from rows/cols",
        "tuser": "first pixel of frame",
        "output_tlast": "last pixel of frame",
        "dma_packet": "one frame per DMA transfer",
    },
    "supports_multistage_pipeline": True,
    "pipeline": {
        "order": ["pre0", "pre1", "pre2", "edge", "post0", "post1"],
        "max_pre_ops": 3,
        "max_post_ops": 2,
        "pre_slots": ["pre0", "pre1", "pre2"],
        "edge_slots": ["edge"],
        "post_slots": ["post0", "post1"],
    },
    "preprocess": {
        "gamma": {
            "supported": True,
            "op_id": OPERATOR_IDS["gamma"],
            "gamma_range": [0.2, 3.0],
            "transport": "256x8 PS-written LUT",
        },
        "gaussian": {
            "supported": True,
            "op_id": OPERATOR_IDS["gaussian"],
            "kernels": [3, 5],
            "sigma_encoding": "Q4.8 in p1; hardware reads low 12 bits",
            "sigma_input_range": [0.0, 3.0],
            "effective_sigma_presets": {
                "3": [0.8, 1.5],
                "5": [1.0, 1.5, 2.0],
            },
            "selection_thresholds_q4_8": {
                "3": [256],
                "5": [320, 448],
            },
        },
        "median": {
            "supported": True,
            "op_id": OPERATOR_IDS["median"],
            "kernels": [3, 5],
        },
        "clahe": {
            "supported": True,
            "op_id": OPERATOR_IDS["clahe"],
            "clip_limit_encoding": "Q0.8 in p0",
            "clip_limit_range": [0.1, 255.0 / 256.0],
            "tile_grid": [FPGA_CLAHE_TILE_GRID,],
            "requires_rows_cols_multiple_of": FPGA_CLAHE_TILE_GRID,
            "interpolation": "horizontal_only",
        },
        "sharpen": {
            "supported": True,
            "op_id": OPERATOR_IDS["sharpen"],
            "kernel": 3,
            "strength_encoding": "Q0.8 in p0",
            "strength_range": [0.0, 255.0 / 256.0],
        },
    },
    "edge_detection": {
        "roberts": {
            "supported": True,
            "op_id": OPERATOR_IDS["roberts"],
            "threshold_range": [0, 255],
        },
        "sobel": {
            "supported": True,
            "op_id": OPERATOR_IDS["sobel"],
            "threshold_range": [0, 255],
        },
        "canny": {
            "supported": True,
            "op_id": OPERATOR_IDS["canny"],
            "low_threshold_range": [0, 255],
            "high_threshold_range": [0, 255],
            "constraint": "low_threshold <= high_threshold",
            "implementation":"Gaussian3x3 -> Sobel L1 -> NMS -> double threshold -> forward hysteresis",
        },
        "hed": {
            "supported": False,
            "reason": "HED is software-only",
        },
    },
    "postprocess": {
        "erosion": {
            "supported": False,
        },
        "dilation": {
            "supported": False,
        },
        "opening": {
            "supported": True,
            "op_id": OPERATOR_IDS["opening"],
            "kernels": [3],
            "iterations": [1],
        },
        "closing": {
            "supported": True,
            "op_id": OPERATOR_IDS["closing"],
            "kernels": [3],
            "iterations": [1],
        },
    },
}

def fpga_op_supported(stage: str, op_name: str) -> bool:
    return bool(FPGA_CAPABILITY.get(stage, {}).get(op_name, {}).get("supported", False))

def gaussian_effective_sigma(kernel: int, sigma: float) -> float:
    """
    Return the effective discrete sigma selected by the current Phase-3 FPGA implementation.
    """
    q = int(round(float(sigma) * 256.0))
    if int(kernel) == 3:
        if q < 256:
            return 0.8
        return 1.5
    if int(kernel) == 5:
        if q < 320:
            return 1.0
        if q < 448:
            return 1.5
        return 2.0
    raise ValueError(f"unsupported FPGA Gaussian kernel: {kernel}")