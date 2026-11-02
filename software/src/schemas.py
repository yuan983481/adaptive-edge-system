from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass
class ImageState:
    noise_level: float = 0.0
    impulse_noise: float = 0.0
    blur_level: float = 0.0
    low_light: float = 0.0
    low_contrast: float = 0.0
    edge_density: float = 0.0
    image_complexity: float = 0.0
    texture_complexity: float = 0.0

    def clamp(self) -> "ImageState":
        for k, v in asdict(self).items():
            setattr(self, k, float(max(0.0, min(1.0, v))))
        return self

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

@dataclass
class Operator:
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, **self.params}

@dataclass
class PipelinePlan:
    preprocess: list[Operator] = field(default_factory=list)
    edge: Operator = field(default_factory=lambda: Operator("sobel", {"threshold": 80}))
    postprocess: list[Operator] = field(default_factory=list)
    reason: str = ""
    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess": [x.to_dict() for x in self.preprocess],
            "edge_detector": self.edge.to_dict(),
            "postprocess": [x.to_dict() for x in self.postprocess],
            "reason": self.reason,
        }