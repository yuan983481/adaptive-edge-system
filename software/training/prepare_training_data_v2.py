import argparse
import json
import math
import random
from pathlib import Path
import cv2
import numpy as np

# ===== Configuration =====
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
TARGETS = [
    "noise_level",
    "impulse_noise",
    "blur_level",
    "low_light",
    "low_contrast",
]
 
# ===== Basic utilities =====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def find_images(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read: {path}")
    return image

def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode: {path}")
    encoded.tofile(str(path))

def resize_pair(a, b):
    """
    Ensure paired images have identical size.
    """
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b

def random_crop(image, size):
    h, w = image.shape[:2]
    if h < size or w < size:
        scale = max(size / h, size / w)
        new_h = math.ceil(h * scale)
        new_w = math.ceil(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = image.shape[:2]
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return image[y:y + size, x:x + size].copy()

def random_pair_crop(a, b, size):
    """
    Crop the same spatial location from paired images.
    """
    a, b = resize_pair(a, b)
    h, w = a.shape[:2]
    if h < size or w < size:
        scale = max(size / h, size / w)
        new_h = math.ceil(h * scale)
        new_w = math.ceil(w * scale)
        a = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(b, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = a.shape[:2]
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return a[y:y + size, x:x + size].copy(), b[y:y + size, x:x + size].copy()

# ===== Synthetic degradation =====
def gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    result = image.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)

def impulse_noise(image, amount):
    output = image.copy()
    h, w = image.shape[:2]
    n = int(h * w * amount)
    # Salt
    y = np.random.randint(0, h, n // 2)
    x = np.random.randint(0, w, n // 2)
    output[y, x] = 255
    # Pepper
    y = np.random.randint(0, h, n // 2)
    x = np.random.randint(0, w, n // 2)
    output[y, x] = 0
    return output

def gaussian_blur(image, sigma):
    kernel = int(2 * round(3 * sigma) + 1)
    kernel = max(3, kernel)
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(image, (kernel, kernel), sigma)

def motion_blur(image, length, angle):
    """
    Synthetic linear motion blur.
    """
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    center = ((length - 1) / 2, (length - 1) / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    return cv2.filter2D(image, -1, kernel)

def low_light(image, gamma):
    x = image.astype(np.float32) / 255.0
    x = np.power(x, gamma)
    return np.clip(x * 255, 0, 255).astype(np.uint8)

def low_contrast(image, factor):
    x = image.astype(np.float32)
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    x = mean + factor * (x - mean)
    return np.clip(x, 0, 255).astype(np.uint8)

# ===== Labels =====
def empty_label():
    return {
        "noise_level": 0.0,
        "impulse_noise": 0.0,
        "blur_level": 0.0,
        "low_light": 0.0,
        "low_contrast": 0.0,
    }

def synthetic_sample(clean):
    label = empty_label()
    modes = [
        "clean",
        "gaussian_noise",
        "impulse_noise",
        "gaussian_blur",
        "motion_blur",
        "low_light",
        "low_contrast",
        "mixed",
    ]
    mode = random.choice(modes)
    output = clean.copy()
    if mode == "clean":
        pass
    elif mode == "gaussian_noise":
        sigma = random.uniform(5, 50)
        output = gaussian_noise(output, sigma)
        label["noise_level"] = min(sigma / 50.0, 1.0)
    elif mode == "impulse_noise":
        amount = random.uniform(0.005, 0.08)
        output = impulse_noise(output, amount)
        label["impulse_noise"] = min(amount / 0.08, 1.0)
    elif mode == "gaussian_blur":
        sigma = random.uniform(0.5, 4.0)
        output = gaussian_blur(output, sigma)
        label["blur_level"] = min(sigma / 4.0, 1.0)
    elif mode == "motion_blur":
        length = random.randint(5, 31)
        angle = random.uniform(0, 180)
        output = motion_blur(output, length, angle)
        label["blur_level"] = min(length / 31.0, 1.0)
    elif mode == "low_light":
        gamma = random.uniform(1.3, 3.0)
        output = low_light(output, gamma)
        label["low_light"] = min((gamma - 1.0) / 2.0, 1.0)
    elif mode == "low_contrast":
        factor = random.uniform(0.2, 0.7)
        output = low_contrast(output, factor)
        label["low_contrast"] = min((1.0 - factor) / 0.8, 1.0 )
    elif mode == "mixed":
        choices = random.sample(["noise", "blur", "light", "contrast"], random.randint(2, 3))
        if "noise" in choices:
            sigma = random.uniform(5, 35)
            output = gaussian_noise(output, sigma)
            label["noise_level"] = min(sigma / 50, 1.0)
        if "blur" in choices:
            length = random.randint(5, 21)
            angle = random.uniform(0, 180)
            output = motion_blur(output, length, angle)
            label["blur_level"] = min(length / 31, 1.0)
        if "light" in choices:
            gamma = random.uniform(1.3, 2.5)
            output = low_light(output, gamma)
            label["low_light"] = min((gamma - 1) / 2, 1.0)
        if "contrast" in choices:
            factor = random.uniform(0.3, 0.7)
            output = low_contrast(output, factor)
            label["low_contrast"] = min((1 - factor) / 0.8, 1.0)
    return output, label, mode

# ===== Real degradation strength estimation =====
def estimate_noise_level(noisy, clean):
    """
    Estimate noise severity using normalized RMSE.
    This is not a physical noise parameter;
    it is a training severity score.
    """
    noisy, clean = resize_pair(noisy, clean)
    diff = noisy.astype(np.float32) - clean.astype(np.float32)
    rmse = np.sqrt(np.mean(diff ** 2))
    # Around 35 intensity RMSE maps to strong noise.
    return float(np.clip(rmse / 35.0, 0, 1))

def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())

def estimate_blur_level(blur, sharp):
    blur, sharp = resize_pair(blur, sharp)
    b = sharpness(blur)
    s = sharpness(sharp)
    if s < 1e-6:
        return 0.0
    severity = 1.0 - (b / s)
    return float(np.clip(severity, 0, 1))

def mean_luminance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def estimate_low_light(low, high):
    low, high = resize_pair(low, high)
    l = mean_luminance(low)
    h = mean_luminance(high)
    if h < 1:
        return 0.0
    severity = 1.0 - (l / h)
    return float(np.clip(severity, 0, 1))

# ===== Pair discovery =====
def pair_by_relative_path(degraded_root, clean_root):
    degraded_root = Path(degraded_root)
    clean_root = Path(clean_root)
    pairs = []
    for degraded in find_images(degraded_root):
        relative = degraded.relative_to(degraded_root)
        clean = clean_root / relative
        if clean.exists():
            pairs.append((degraded, clean))
    return pairs

def find_sidd_pairs(root):
    root = Path(root)
    input_root = root / "images" / "validation" / "input"
    gt_candidates = [
        root / "images" / "validation" / "groundtruth",
        root / "images" / "validation" / "ground_truth",
        root / "images" / "validation" / "gt",
        root / "images" / "validation" / "target",
    ]
    for gt_root in gt_candidates:
        if gt_root.exists():
            pairs = pair_by_relative_path(input_root, gt_root)
            if pairs:
                return pairs
    return []

def find_lol_pairs(root):
    root = Path(root)
    pairs = []
    for split in ["train", "test"]:
        low = root / split / "low"
        high = root / split / "high"
        if low.exists() and high.exists():
            pairs.extend(pair_by_relative_path(low, high))
    return pairs

def find_gopro_pairs(root):
    root = Path(root)
    pairs = []
    # Handles:
    # images/train/scene/blur/*.png
    # images/train/scene/sharp/*.png
    # images/test/scene/blur/*.png
    # images/test/scene/sharp/*.png
    for blur_dir in root.rglob("blur"):
        if not blur_dir.is_dir():
            continue
        sharp_dir = blur_dir.parent / "sharp"
        if not sharp_dir.exists():
            continue
        pairs.extend(pair_by_relative_path(blur_dir, sharp_dir))
    return pairs

# ===== Split pairs before crop generation =====
def split_items(items, val_ratio, seed):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) <= 1:
        return items, []
    n_val = max(1, int(len(items) * val_ratio))
    n_val = min(n_val, len(items) - 1)
    return (items[n_val:], items[:n_val])

# ===== Record writer =====
def make_record(relative_path, label, source, degradation):
    return {
        "image": str(relative_path).replace("\\", "/"),
        "source_dataset": source,
        "degradation": degradation,
        **label,
    }

# ===== Generate synthetic DIV2K =====
def generate_div2k(images, output_root, split, crop_size, samples_per_image):
    records = []
    out_dir = output_root / split / "images"
    print(f"\n[{split}] DIV2K: {len(images)} source images")
    counter = 0
    for index, path in enumerate(images):
        try:
            image = read_image(path)
        except Exception as e:
            print(f"\n[WARNING] {e}")
            continue
        for sample in range(samples_per_image):
            clean = random_crop(image, crop_size)
            degraded, label, mode = synthetic_sample(clean)
            name = f"div2k_{index:04d}_{sample:02d}_{mode}.png"
            relative = Path(split) / "images" / name
            save_image(output_root / relative, degraded)
            records.append(make_record(relative, label, "DIV2K", mode))
            counter += 1
            if counter % 500 == 0:
                print(f"\rGenerated {counter}", end="", flush=True)
    print(f"\nDIV2K {split}: {len(records)} samples")
    return records

# ===== Generate real paired samples =====
def generate_real_pairs(pairs, output_root, split, crop_size, crops_per_pair, source, degradation):
    records = []
    print(f"\n[{split}] {source}: {len(pairs)} pairs")
    for pair_index, (degraded_path, clean_path) in enumerate(pairs):
        try:
            degraded = read_image(degraded_path)
            clean = read_image(clean_path)
        except Exception as e:
            print(f"\n[WARNING] {e}")
            continue
        for crop_index in range(crops_per_pair):
            d_crop, c_crop = random_pair_crop(degraded, clean, crop_size)
            label = empty_label()
            if degradation == "real_noise":
                label["noise_level"] = estimate_noise_level(d_crop, c_crop)
            elif degradation == "real_low_light":
                label["low_light"] = estimate_low_light(d_crop, c_crop)
            elif degradation == "real_motion_blur":
                label["blur_level"] = estimate_blur_level(d_crop, c_crop)
            name = f"{source.lower()}_{pair_index:05d}_{crop_index:02d}.png"
            relative = Path(split) / "images" / name
            save_image(output_root / relative, d_crop)
            records.append(make_record(relative, label, source, degradation))
    print(f"{source} {split}: {len(records)} samples")
    return records

# ===== Save JSON =====
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--output", default="datasets/ImageAnalyzerV2")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--div2k-train-samples", type=int, default=6)
    parser.add_argument("--div2k-val-samples", type=int, default=2)
    parser.add_argument("--real-train-crops", type=int, default=3)
    parser.add_argument("--real-val-crops", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("=" * 70 + "Image Analyzer V2 Dataset Generator" + "=" * 70)

    # DIV2K
    div2k_root = root / "DIV2K"
    div2k = find_images(div2k_root)
    if not div2k:
        raise RuntimeError(f"No DIV2K images found under {div2k_root}")
    div_train, div_val = split_items(div2k, args.val_ratio, args.seed)
    # SIDD
    sidd_pairs = find_sidd_pairs(root / "SIDD_Validation")
    sidd_train, sidd_val = split_items(sidd_pairs, args.val_ratio, args.seed + 1)
    # LOL
    lol_pairs = find_lol_pairs(root / "LOLv1")
    lol_train, lol_val = split_items(lol_pairs, args.val_ratio, args.seed + 2)
    # GoPro
    gopro_pairs = find_gopro_pairs(root / "GoPro_Large")
    gopro_train, gopro_val = split_items(gopro_pairs, args.val_ratio, args.seed + 3)
    print("=== SOURCE SUMMARY ===")
    print(f"DIV2K : {len(div2k)} images")
    print(f"SIDD  : {len(sidd_pairs)} pairs")
    print(f"LOL   : {len(lol_pairs)} pairs")
    print(f"GoPro : {len(gopro_pairs)} pairs")

    # Generate
    train_records = []
    val_records = []
    train_records += generate_div2k(
        div_train,
        output,
        "train",
        args.crop_size,
        args.div2k_train_samples
    )
    val_records += generate_div2k(
        div_val,
        output,
        "val",
        args.crop_size,
        args.div2k_val_samples
    )
    # SIDD
    train_records += generate_real_pairs(
        sidd_train,
        output,
        "train",
        args.crop_size,
        args.real_train_crops,
        "SIDD",
        "real_noise"
    )
    val_records += generate_real_pairs(
        sidd_val,
        output,
        "val",
        args.crop_size,
        args.real_val_crops,
        "SIDD",
        "real_noise"
    )
    # LOL
    train_records += generate_real_pairs(
        lol_train,
        output,
        "train",
        args.crop_size,
        args.real_train_crops,
        "LOL",
        "real_low_light"
    )
    val_records += generate_real_pairs(
        lol_val,
        output,
        "val",
        args.crop_size,
        args.real_val_crops,
        "LOL",
        "real_low_light"
    )
    # GoPro
    train_records += generate_real_pairs(
        gopro_train,
        output,
        "train",
        args.crop_size,
        args.real_train_crops,
        "GoPro",
        "real_motion_blur"
    )
    val_records += generate_real_pairs(
        gopro_val,
        output,
        "val",
        args.crop_size,
        args.real_val_crops,
        "GoPro",
        "real_motion_blur"
    )
    random.shuffle(train_records)
    random.shuffle(val_records)
    save_json(train_records, output / "train_labels.json")
    save_json(val_records, output / "val_labels.json")
    info = {
        "version": 2,
        "crop_size": args.crop_size,
        "targets": TARGETS,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "sources": {
            "DIV2K": len(div2k),
            "SIDD_pairs": len(sidd_pairs),
            "LOL_pairs": len(lol_pairs),
            "GoPro_pairs": len(gopro_pairs),
        }
    }
    save_json(info, output / "dataset_info.json")
    print("=" * 70 + "V2 DATASET SUMMARY" + "=" * 70)
    print(f"Train samples : {len(train_records)}")
    print(f"Val samples   : {len(val_records)}")
    print(f"Total         : {len(train_records) + len(val_records)}")
    print("CNN targets:")
    for target in TARGETS:
        print(f"  - {target}")
    if not sidd_pairs:
        print("[WARNING] No SIDD pairs detected.")
    if not lol_pairs:
        print("[WARNING] No LOL pairs detected.")
    if not gopro_pairs:
        print("[WARNING] No GoPro pairs detected.")
    print(f"Saved to: {output}")

if __name__ == "__main__":
    main()