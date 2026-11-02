import cv2
import numpy as np
from .schemas import ImageState

def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def classical_image_state(image: np.ndarray) -> ImageState:
    gray = _gray(image).astype(np.uint8)
    x = gray.astype(np.float32) / 255.0

    mean = float(x.mean())
    std = float(x.std())

    low_light = np.clip((0.45 - mean) / 0.45, 0, 1)
    low_contrast = np.clip((0.20 - std) / 0.20, 0, 1)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blur_level = np.clip(1.0 - lap_var / 500.0, 0, 1)

    canny = cv2.Canny(gray, 80, 160)
    edge_density = float((canny > 0).mean())

    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - smooth.astype(np.float32)
    noise_sigma = float(residual.std())
    noise_level = np.clip(noise_sigma / 30.0, 0, 1)

    med = cv2.medianBlur(gray, 3)
    diff = np.abs(gray.astype(np.int16) - med.astype(np.int16))
    extreme = ((gray < 8) | (gray > 247)) & (diff > 35)
    impulse_noise = np.clip(float(extreme.mean()) / 0.02, 0, 1)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    image_complexity = np.clip(entropy / 8.0, 0, 1)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    texture_complexity = np.clip(float(np.mean(grad)) / 80.0, 0, 1)

    return ImageState(
        noise_level=float(noise_level),
        impulse_noise=float(impulse_noise),
        blur_level=float(blur_level),
        low_light=float(low_light),
        low_contrast=float(low_contrast),
        edge_density=float(np.clip(edge_density / 0.30, 0, 1)),
        image_complexity=float(image_complexity),
        texture_complexity=float(texture_complexity),
    ).clamp()