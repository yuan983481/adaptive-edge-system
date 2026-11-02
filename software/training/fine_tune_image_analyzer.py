import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

TARGETS = [
    "noise_level",
    "impulse_noise",
    "blur_level",
    "low_light",
    "low_contrast",
]

# ===== Seed =====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===== Dataset =====
class V2Dataset(Dataset):
    def __init__(self, root, label_file, train=False):
        self.root = Path(root)
        with open(label_file, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.root / record["image"]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor([float(record[name]) for name in TARGETS], dtype=torch.float32)
        return image, target

# ===== CNN =====
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
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
            ConvBlock(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # V2 = 5 outputs
            nn.Linear(64, 5),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.regressor(x)

# ===== Load V1 weights =====
def load_v1_weights(model, checkpoint_path):
    print("Loading V1 checkpoint:")
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        old_state = checkpoint["model_state_dict"]
    else:
        old_state = checkpoint
    new_state = model.state_dict()
    loaded = []
    skipped = []
    for key, value in old_state.items():
        if key in new_state and new_state[key].shape == value.shape:
            new_state[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(new_state)
    print(f"Loaded tensors : {len(loaded)}")
    print(f"Skipped tensors: {len(skipped)}")
    print("Skipped layers (expected for 8 -> 5 output):")
    for key in skipped:
        print(f"  {key}")

# ===== Loss =====
class WeightedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        # Slightly emphasize real-world-important noise / blur / low-light.
        self.register_buffer(
            "weights",
            torch.tensor(
                [
                    1.2,  # noise
                    1.0,  # impulse
                    1.2,  # blur
                    1.2,  # low light
                    1.0,  # low contrast
                ], dtype=torch.float32
            )
        )
    def forward(self, prediction, target):
        error = (prediction - target) ** 2
        return (error * self.weights).mean()

# ===== Training =====
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(images)
        loss = criterion(prediction, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item() * images.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    abs_error = torch.zeros(len(TARGETS), device=device)
    count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction = model(images)
        loss = criterion(prediction, targets)
        total += loss.item() * images.size(0)
        abs_error += torch.sum(torch.abs(prediction - targets), dim=0)
        count += images.size(0)
    loss = total / len(loader.dataset)
    mae = (abs_error / count).cpu().numpy()
    return loss, mae

# ===== Plot =====
def plot_curve(train_loss, val_loss, path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Image Analyzer V2 Fine-tuning")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/ImageAnalyzerV2")
    parser.add_argument("--v1-model", default="models/image_analyzer_best.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    data_root = Path(args.data_root)
    output = Path("models")
    output.mkdir(exist_ok=True)
    # ----- Device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70 + "Image Analyzer V2 Fine-tuning" + "=" * 70)
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print("GPU    : " + torch.cuda.get_device_name(0))

    # ----- Dataset -----
    train_dataset = V2Dataset(data_root, data_root / "train_labels.json", train=True )
    val_dataset = V2Dataset(data_root, data_root / "val_labels.json", train=False)
    print(f"Train  : {len(train_dataset)}")
    print(f"Val    : {len(val_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    # ----- Model -----
    model = ImageAnalyzerV2()
    v1_path = Path(args.v1_model)
    if not v1_path.exists():
        raise FileNotFoundError(f"V1 model not found: {v1_path}")
    load_v1_weights(model, v1_path)
    model = model.to(device)
    criterion = WeightedMSE().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # ----- Fine tune -----
    best_loss = float("inf")
    bad_epochs = 0
    train_history = []
    val_history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, mae = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        train_history.append(train_loss)
        val_history.append(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}]")
        print(f"Train Loss : {train_loss:.6f}")
        print(f"Val Loss   : {val_loss:.6f}")
        print(f"LR         : {lr:.8f}")
        print(f"Time       : {time.time()-start:.1f}s")
        print("Validation MAE:")
        for name, value in zip(TARGETS, mae):
            print(f"  {name:<20}{value:.4f}")
        checkpoint = {
            "version": 2,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "targets": TARGETS,
        }
        torch.save(checkpoint, output / "image_analyzer_v2_latest.pth")
        if val_loss < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            torch.save(checkpoint, output / "image_analyzer_v2_best.pth")
            print(">>> Best V2 model updated.")
        else:
            bad_epochs += 1
            print(f"No improvement: {bad_epochs}/{args.patience}")
        if bad_epochs >= args.patience:
            print("Early stopping.")
            break

    # ----- Plot -----
    plot_curve(train_history, val_history, output / "training_curve_v2.png")
    print("=" * 70 + "V2 Fine-tuning Completed" + "=" * 70)
    print(f"Best validation loss : {best_loss:.6f}")
    print("Best model : models/image_analyzer_v2_best.pth")
    print("Curve      : models/training_curve_v2.png")

if __name__ == "__main__":
    main()