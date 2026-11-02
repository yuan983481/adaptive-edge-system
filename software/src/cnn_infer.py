import cv2
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torchvision import transforms
from .schemas import ImageState

# ===== V2 configuration =====
CNN_TARGETS = [
    "noise_level",
    "impulse_noise",
    "blur_level",
    "low_light",
    "low_contrast",
]

# ===== V2 CNN architecture =====
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.block(x)

class ImageAnalyzerV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x

# ===== Classical image features =====
def calculate_edge_density(image_bgr):
    """
    Edge pixel ratio based on Canny.
    Returns approximately [0, 1].
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges) / edges.size)

def calculate_image_complexity(image_bgr):
    """
    Estimate global image complexity using grayscale entropy.
    Normalized approximately to [0, 1].
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total <= 0:
        return 0.0
    prob = hist / total
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    # 8 bits is the maximum grayscale entropy.
    return float(np.clip(entropy / 8.0, 0.0, 1.0))

def calculate_texture_complexity(image_bgr):
    """
    Estimate local texture activity from Sobel gradient magnitude.
    Returns a normalized value in [0, 1].
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    # Scale mean gradient into a convenient [0, 1] range.
    value = float(np.mean(magnitude) * 2.0)
    return float(np.clip(value, 0.0, 1.0))

# ===== CNN Image Analyzer V2 =====
class CNNImageAnalyzer:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint}")

        # Create V2 model
        self.model = ImageAnalyzerV2().to(self.device)

        # Load checkpoint
        saved = torch.load(checkpoint, map_location=self.device)
        if isinstance(saved, dict) and "model_state_dict" in saved:
            state_dict = saved["model_state_dict"]
        # Compatibility with some other checkpoint formats.
        elif isinstance(saved, dict) and "model" in saved:
            state_dict = saved["model"]
        else:
            state_dict = saved
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                "\nFailed to load Image Analyzer V2 checkpoint.\n"
                f"Checkpoint: {checkpoint}\n"
                "The checkpoint architecture does not match ImageAnalyzerV2.\n"
                "Make sure you are loading models/image_analyzer_v2_best.pth.\n"
                f"Original PyTorch error:\n{e}"
            ) from e
        self.model.eval()

        # Must match V2 training preprocessing
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        print(f"[CNN] Image Analyzer V2 loaded: {checkpoint}")
        print(f"[CNN] device={self.device}")

    # ----- Analyze -----
    @torch.no_grad()
    def analyze(self, image_bgr: np.ndarray) -> ImageState:
        if image_bgr is None:
            raise ValueError("image_bgr is None")

        # CNN V2: 5 degradation states
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        prediction = self.model(x).squeeze(0).cpu().numpy()
        cnn_state = {name: float(value) for name, value in zip(CNN_TARGETS, prediction)}

        # Classical CV: 3 structural features
        edge_density = calculate_edge_density(image_bgr)
        image_complexity = calculate_image_complexity(image_bgr)
        texture_complexity = calculate_texture_complexity(image_bgr)

        # Complete 8-D Image State
        state = ImageState(
            noise_level=cnn_state["noise_level"],
            impulse_noise=cnn_state["impulse_noise"],
            blur_level=cnn_state["blur_level"],
            low_light=cnn_state["low_light"],
            low_contrast=cnn_state["low_contrast"],
            edge_density=edge_density,
            image_complexity=image_complexity,
            texture_complexity=texture_complexity,
        )
        return state.clamp()