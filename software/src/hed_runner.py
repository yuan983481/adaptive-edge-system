import cv2
import numpy as np
from pathlib import Path

class HEDRunner:
    def __init__(
        self, 
        prototxt: str | Path,
        caffemodel: str | Path,
        input_size=(500, 500),
    ):
        prototxt = Path(prototxt)
        caffemodel = Path(caffemodel)
        if not prototxt.exists() or not caffemodel.exists():
            raise FileNotFoundError("HED model files not found. Update config.yaml with valid paths.")
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        self.input_size = tuple(input_size)

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0,
            size=self.input_size,
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        hed = self.net.forward()[0, 0]
        hed = cv2.resize(hed, (w, h))
        hed = np.clip(hed * 255.0, 0, 255).astype(np.uint8)
        _, edge = cv2.threshold(hed, 80, 255, cv2.THRESH_BINARY)
        return edge