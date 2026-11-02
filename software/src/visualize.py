import cv2
import numpy as np
from pathlib import Path

def make_side_by_side(input_bgr: np.ndarray, edge_gray: np.ndarray, title: str = "") -> np.ndarray:
    if edge_gray.ndim == 2:
        edge_bgr = cv2.cvtColor(edge_gray, cv2.COLOR_GRAY2BGR)
    else:
        edge_bgr = edge_gray
    h = input_bgr.shape[0]
    if edge_bgr.shape[0] != h:
        scale = h / edge_bgr.shape[0]
        edge_bgr = cv2.resize(edge_bgr, None, fx=scale, fy=scale)
    out = np.hstack([input_bgr, edge_bgr])
    if title:
        cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")