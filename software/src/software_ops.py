import cv2
import numpy as np
from .schemas import Operator

def ensure_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def apply_preprocess(image: np.ndarray, op: Operator) -> np.ndarray:
    x = ensure_gray_u8(image)
    if op.op == "gaussian":
        k = int(op.params.get("kernel", 3))
        sigma = float(op.params.get("sigma", 0.0))
        return cv2.GaussianBlur(x, (k, k), sigma)
    if op.op == "median":
        k = int(op.params.get("kernel", 3))
        return cv2.medianBlur(x, k)
    if op.op == "gamma":
        gamma = float(op.params.get("gamma", 1.0))
        inv = 1.0 / max(gamma, 1e-6)
        lut = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(x, lut)
    if op.op == "clahe":
        clip = float(op.params.get("clip_limit", 2.0))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        return clahe.apply(x)
    if op.op == "sharpen":
        strength = float(op.params.get("strength", 0.5))
        blur = cv2.GaussianBlur(x, (3, 3), 0)
        out = cv2.addWeighted(x, 1.0 + strength, blur, -strength, 0)
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported preprocess operator: {op.op}")

def apply_edge(image: np.ndarray, op: Operator) -> np.ndarray:
    x = ensure_gray_u8(image)
    if op.op == "roberts":
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        gx = cv2.filter2D(x, cv2.CV_32F, kx)
        gy = cv2.filter2D(x, cv2.CV_32F, ky)
        mag = cv2.magnitude(gx, gy)
        mag = np.clip(mag, 0, 255).astype(np.uint8)
        t = int(op.params.get("threshold", 55))
        _, out = cv2.threshold(mag, t, 255, cv2.THRESH_BINARY)
        return out
    if op.op == "sobel":
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = np.clip(mag, 0, 255).astype(np.uint8)
        t = int(op.params.get("threshold", 80))
        _, out = cv2.threshold(mag, t, 255, cv2.THRESH_BINARY)
        return out
    if op.op == "canny":
        lo = int(op.params.get("low_threshold", 50))
        hi = int(op.params.get("high_threshold", 120))
        return cv2.Canny(x, lo, hi)
    raise ValueError(f"Unsupported edge operator: {op.op}")

def apply_postprocess(image: np.ndarray, op: Operator) -> np.ndarray:
    x = ensure_gray_u8(image)
    k = int(op.params.get("kernel", 3))
    iterations = int(op.params.get("iterations", 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if op.op == "erosion":
        return cv2.erode(x, kernel, iterations=iterations)
    if op.op == "dilation":
        return cv2.dilate(x, kernel, iterations=iterations)
    if op.op == "opening":
        return cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if op.op == "closing":
        return cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    raise ValueError(f"Unsupported postprocess operator: {op.op}")