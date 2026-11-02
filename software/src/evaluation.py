import cv2
import numpy as np

def _binary(x: np.ndarray, threshold: int = 127) -> np.ndarray:
    if x.ndim == 3:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return x > threshold

def edge_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p = _binary(pred)
    g = _binary(gt)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))

def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(structural_similarity(a, b, data_range=255))

def edge_feedback_metrics(edge: np.ndarray, threshold: int = 127) -> dict[str, float]:
    """No-reference feedback for the agent loop.

    These are deliberately simple, explainable measurements that can also be
    reproduced in FPGA/PS logic later: edge density plus connected-component
    fragmentation statistics. They are not a replacement for GT metrics.
    """
    mask = _binary(edge, threshold=threshold).astype(np.uint8)
    edge_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    density = edge_pixels / max(total_pixels, 1)

    if edge_pixels == 0:
        return {
            "edge_density": 0.0,
            "component_count": 0.0,
            "tiny_component_ratio": 1.0,
            "mean_component_size": 0.0,
        }

    nlabels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if nlabels <= 1:
        return {
            "edge_density": float(density),
            "component_count": 0.0,
            "tiny_component_ratio": 0.0,
            "mean_component_size": float(edge_pixels),
        }

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    tiny = areas <= 4
    tiny_pixels = float(areas[tiny].sum())
    return {
        "edge_density": float(density),
        "component_count": float(len(areas)),
        "tiny_component_ratio": float(tiny_pixels / max(edge_pixels, 1)),
        "mean_component_size": float(areas.mean()),
    }
