import cv2
import numpy as np
from .hed_runner import HEDRunner
from .schemas import PipelinePlan
from .software_ops import apply_edge, apply_postprocess, apply_preprocess, ensure_gray_u8

class PipelineExecutor:
    def __init__(self, backend: str = "software", fpga_runtime=None, hed_runner: HEDRunner | None = None):
        self.backend = str(backend).lower()
        self.fpga_runtime = fpga_runtime
        self.hed_runner = hed_runner

    def run(self, image_bgr: np.ndarray, plan: PipelinePlan) -> np.ndarray:
        # FPGA backend
        if self.backend == "fpga":
            if plan.edge.op == "hed":
                raise RuntimeError("HED is software-only and cannot run on FPGA")
            if self.fpga_runtime is None:
                raise RuntimeError("FPGA backend selected without fpga_runtime")
            gray = ensure_gray_u8(image_bgr)
            return self.fpga_runtime.process_gray(gray, plan)
        # Software backend
        if self.backend != "software":
            raise RuntimeError(f"unknown executor backend: {self.backend}")
        if plan.edge.op == "hed":
            return self._run_hed_branch(image_bgr, plan)
        return self._run_software(image_bgr, plan)

    def _run_software(self, image_bgr: np.ndarray, plan: PipelinePlan) -> np.ndarray:
        x = ensure_gray_u8(image_bgr)
        for op in plan.preprocess:
            x = apply_preprocess(x, op)
        x = apply_edge(x, plan.edge)
        for op in plan.postprocess:
            x = apply_postprocess(x, op)
        return x

    def _run_hed_branch(self, image_bgr: np.ndarray, plan: PipelinePlan) -> np.ndarray:
        if self.hed_runner is None:
            raise RuntimeError("Pipeline selected HED but no HED model is configured")
        x_gray = ensure_gray_u8(image_bgr)
        for op in plan.preprocess:
            x_gray = apply_preprocess(x_gray, op)
        x_bgr = cv2.cvtColor(x_gray, cv2.COLOR_GRAY2BGR)
        edge = self.hed_runner(x_bgr)
        for op in plan.postprocess:
            edge = apply_postprocess(edge, op)
        return edge