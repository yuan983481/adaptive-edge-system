import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ===== Configuration =====
TARGETS = [
    "noise_level",
    "impulse_noise",
    "blur_level",
    "low_light",
    "low_contrast",
]

# ===== CNN architecture =====
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
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
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.regressor(x)

# ===== Validation Dataset =====
class ValidationDataset(Dataset):
    def __init__(self, root, label_file):
        self.root = Path(root)
        with open(label_file, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.root / record["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor([float(record[name]) for name in TARGETS], dtype=torch.float32)
        source = record.get("source_dataset", "unknown")
        degradation = record.get("degradation", "unknown")
        return image, target, source, degradation

# ===== Load model ======
def load_model(model_path, device):
    model = ImageAnalyzerV2().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ===== Collect predictions =====
@torch.no_grad()
def collect_predictions(model, loader, device):
    all_predictions = []
    all_targets = []
    all_sources = []
    all_degradations = []
    for (images, targets, sources, degradations) in loader:
        images = images.to(device, non_blocking=True)
        predictions = model(images)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.numpy())
        all_sources.extend(list(sources))
        all_degradations.extend(list(degradations))
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return predictions, targets, all_sources, all_degradations

# ===== Metrics =====
def binary_metrics(prediction_scores, ground_truth, prediction_threshold, label_threshold):
    true_positive_mask = ground_truth >= label_threshold
    predicted_positive_mask = prediction_scores >= prediction_threshold
    tp = np.sum(true_positive_mask & predicted_positive_mask)
    fp = np.sum((~true_positive_mask) & predicted_positive_mask)
    fn = np.sum(true_positive_mask & (~predicted_positive_mask))
    tn = np.sum((~true_positive_mask) & (~predicted_positive_mask))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

# ===== Threshold search =====
def find_best_threshold(prediction_scores, ground_truth, label_threshold, min_threshold=0.01, max_threshold=0.99, steps=990):
    positive_count = int(np.sum(ground_truth >= label_threshold))
    negative_count = int(len(ground_truth) - positive_count)
    # Cannot calibrate if validation data contains only one class.
    if positive_count == 0 or negative_count == 0:
        return None
    candidates = np.linspace(min_threshold, max_threshold, steps)
    best = None
    for threshold in candidates:
        metrics = binary_metrics(prediction_scores, ground_truth, threshold, label_threshold)
        result = {
            "threshold": float(threshold),
            **metrics
        }
        if best is None:
            best = result
            continue
        # Primary objective: F1
        if result["f1"] > best["f1"]:
            best = result
        # Tie-breaker 1: prefer higher recall
        elif abs(result["f1"] - best["f1"]) < 1e-12 and result["recall"] > best["recall"]:
            best = result
        # Tie-breaker 2: prefer higher threshold if F1 and recall are identical.
        elif abs(result["f1"] - best["f1"]) < 1e-12 and abs(result["recall"] - best["recall"]) < 1e-12 and result["threshold"] > best["threshold"]:
            best = result
    best["positive_samples"] = positive_count
    best["negative_samples"] = negative_count
    return best

# ===== Score distribution statistics =====
def distribution_statistics(prediction_scores, ground_truth, label_threshold):
    positive_mask = ground_truth >= label_threshold
    negative_mask = ~positive_mask
    positive_scores = prediction_scores[positive_mask]
    negative_scores = prediction_scores[negative_mask]
    def describe(values):
        if len(values) == 0:
            return None
        return {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
        }
    return {
        "positive": describe(positive_scores),
        "negative": describe(negative_scores)
    }

# ===== Source diagnostics =====
def source_diagnostics(predictions, targets, sources, degradations):
    diagnostics = {}
    unique_sources = sorted(set(sources))
    for source in unique_sources:
        indices = [i for i, value in enumerate(sources) if value == source]
        if not indices:
            continue
        pred = predictions[indices]
        gt = targets[indices]
        source_result = {
            "samples": len(indices),
            "mean_prediction": {},
            "mean_target": {},
        }
        for target_index, name in enumerate(TARGETS):
            source_result["mean_prediction"][name] = float(np.mean(pred[:, target_index]))
            source_result["mean_target"][name] = float(np.mean(gt[:, target_index]))
        diagnostics[source] = source_result
    return diagnostics

def main():
    parser = argparse.ArgumentParser(description="Calibrate Image Analyzer V2 prediction thresholds.")
    parser.add_argument("--data-root", default="datasets/ImageAnalyzerV2")
    parser.add_argument("--model", default="models/image_analyzer_v2_best.pth")
    parser.add_argument("--output", default="models/image_analyzer_thresholds.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    # label >= 0.25 is considered a meaningful degradation
    # IMPORTANT: This is the label-space definition, not the CNN output threshold.
    parser.add_argument("--label-positive-threshold", type=float, default=0.25)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    model_path = Path(args.model)
    output_path = Path(args.output)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset not found: {data_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    label_file = data_root / "val_labels.json"
    if not label_file.exists():
        raise FileNotFoundError(f"Validation labels not found: {label_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72 + "IMAGE ANALYZER V2 CALIBRATION" + "=" * 72)
    print(f"Device          : {device}")
    print(f"Model           : {model_path}")
    print(f"Validation data : {label_file}")
    print(f"Label positive  : >= {args.label_positive_threshold:.3f}")

    # Dataset
    dataset = ValidationDataset(data_root, label_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )
    print(f"Validation samples: {len(dataset)}")

    # Model
    model = load_model(model_path, device)
    print("Collecting predictions...")
    (predictions, targets, sources, degradations) = collect_predictions(model, loader, device)
    print("Prediction collection completed.")

    # Calibration
    thresholds = {}
    calibration = {}
    distributions = {}
    print("=" * 72 + "CALIBRATION RESULTS" + "=" * 72)
    for index, name in enumerate(TARGETS):
        pred = predictions[:, index]
        gt = targets[:, index]
        best = find_best_threshold(pred, gt, args.label_positive_threshold)
        distributions[name] = distribution_statistics(pred, gt, args.label_positive_threshold)
        print(f"[{name}]")
        if best is None:
            print("Cannot calibrate: validation set does not contain both classes.")
            # Conservative fallback.
            thresholds[name] = 0.50
            calibration[name] = {
                "status": "fallback",
                "threshold": 0.50
            }
            continue
        thresholds[name] = best["threshold"]
        calibration[name] = {
            "status": "calibrated",
            **best
        }
        print(f"  threshold : {best['threshold']:.4f}")
        print(f"  precision : {best['precision']:.4f}")
        print(f"  recall    : {best['recall']:.4f}")
        print(f"  F1        : {best['f1']:.4f}")
        print(f"  accuracy  : {best['accuracy']:.4f}")
        print(f"  positive  : {best['positive_samples']}")
        print(f"  negative  : {best['negative_samples']}")

    # Source diagnostics
    diagnostics = source_diagnostics(predictions, targets, sources, degradations)

    # Save
    output_data = {
        "version": 1,
        "model": str(model_path),
        "validation_samples": len(dataset),
        "label_positive_threshold": args.label_positive_threshold,
        "thresholds": thresholds,
        "calibration": calibration,
        "score_distributions": distributions,
        "source_diagnostics": diagnostics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Summary
    print("=" * 72 + "FINAL THRESHOLDS" + "=" * 72)
    for name in TARGETS:
        print(f"{name:<20}: {thresholds[name]:.4f}")
    print(f"Saved to: {output_path}")
    print("=" * 72)

if __name__ == "__main__":
    main()