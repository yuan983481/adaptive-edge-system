import argparse
import csv
import shutil
from pathlib import Path
import cv2
import numpy as np

# ===== Configuration =====
DEFAULT_THRESHOLD = 0.1834
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

# ===== CLI =====
def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose GoPro blur predictions from adaptive_results.csv")
    parser.add_argument(
        "--csv",
        default="outputs/evaluation/adaptive_results.csv",
        help="Evaluation CSV",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Blur detection threshold",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of samples selected for each diagnostic group",
    )
    parser.add_argument(
        "--out",
        default="outputs/evaluation/gopro_blur_diagnosis",
        help="Diagnosis output directory",
    )
    return parser.parse_args()

# ===== CSV loading =====
def load_gopro_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset", "").strip().lower() != "gopro":
                continue
            if row.get("status", "").strip().lower() != "ok":
                continue
            try:
                row["_blur"] = float(row["blur_level"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    return rows

# ===== Select diagnostic samples =====
def select_groups(rows: list[dict], threshold: float, count: int) -> dict[str, list[dict]]:
    ordered = sorted(rows, key=lambda x: x["_blur"])
    # Lowest CNN blur scores
    lowest = ordered[:count]
    # Highest CNN blur scores
    highest = ordered[-count:] if count > 0 else []
    highest = list(reversed(highest))
    # Closest samples to calibrated threshold
    borderline = sorted(rows, key=lambda x: abs(x["_blur"] - threshold))[:count]
    return {
        "lowest": lowest,
        "borderline": borderline,
        "highest": highest,
    }

# ===== GoPro sharp-image lookup =====
def find_sharp_pair(blur_path: Path) -> Path | None:
    parts = list(blur_path.parts)
    blur_index = None
    for i, part in enumerate(parts):
        if part.lower() == "blur":
            blur_index = i
            break
    if blur_index is None:
        return None
    sharp_parts = list(parts)
    sharp_parts[blur_index] = "sharp"
    sharp_path = Path(*sharp_parts)
    if sharp_path.exists():
        return sharp_path
    # Some datasets may use different case.
    if blur_index == 0:
        return None
    sequence_dir = Path(*parts[:blur_index])
    if not sequence_dir.exists():
        return None
    for candidate_dir in sequence_dir.iterdir():
        if candidate_dir.is_dir() and candidate_dir.name.lower() == "sharp":
            candidate = candidate_dir / blur_path.name
            if candidate.exists():
                return candidate
    return None

# ===== Blur measurements =====
def variance_of_laplacian(image: np.ndarray) -> float:
    """
    Classical sharpness indicator.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    value = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(value)

def tenengrad(image: np.ndarray) -> float:
    """
    Gradient-based sharpness indicator.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )
    magnitude_squared = gx * gx + gy * gy
    return float(np.mean(magnitude_squared))

# ===== Visualization =====
def normalize_for_display(image: np.ndarray, target_height: int = 360) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    scale = target_height / float(h)
    new_width = max(1, int(round(w * scale)))
    return cv2.resize(
        image,
        (new_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

def add_label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    overlay_height = 45
    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1], overlay_height),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        result,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result

def make_comparison(blur_image: np.ndarray, sharp_image: np.ndarray | None, blur_score: float, threshold: float) -> np.ndarray:
    blur_display = normalize_for_display(blur_image)
    blur_display = add_label(blur_display, f"Blur input | CNN={blur_score:.4f} | threshold={threshold:.4f}")
    if sharp_image is None:
        return blur_display
    sharp_display = normalize_for_display(sharp_image)
    sharp_display = add_label(sharp_display, "Sharp reference")
    # Match heights
    height = min(blur_display.shape[0], sharp_display.shape[0])
    blur_display = blur_display[:height]
    sharp_display = sharp_display[:height]
    return np.hstack([blur_display, sharp_display])

# ===== Diagnose one sample =====
def diagnose_sample(row: dict, group: str, rank: int, threshold: float, output_dir: Path,) -> dict:
    image_path = Path(row["image"])
    blur_score = float(row["_blur"])
    result = {
        "group": group,
        "rank": rank,
        "image": str(image_path),
        "sharp_image": "",
        "blur_level": blur_score,
        "threshold": threshold,
        "detected": int(blur_score >= threshold),
        "distance_to_threshold": blur_score - threshold,
        "blur_laplacian_var": "",
        "sharp_laplacian_var": "",
        "laplacian_ratio": "",
        "blur_tenengrad": "",
        "sharp_tenengrad": "",
        "tenengrad_ratio": "",
        "comparison_image": "",
        "status": "",
        "error": "",
    }
    try:
        blur_image = cv2.imread(str(image_path))
        if blur_image is None:
            raise RuntimeError(f"Cannot read blur image: {image_path}")
        sharp_path = find_sharp_pair(image_path)
        sharp_image = None
        if sharp_path is not None:
            sharp_image = cv2.imread(str(sharp_path))
            if sharp_image is not None:
                result["sharp_image"] = str(sharp_path)
        # Classical sharpness indicators
        blur_lap = variance_of_laplacian(blur_image)
        blur_ten = tenengrad(blur_image)
        result["blur_laplacian_var"] = blur_lap
        result["blur_tenengrad"] = blur_ten
        if sharp_image is not None:
            sharp_lap = variance_of_laplacian(sharp_image)
            sharp_ten = tenengrad(sharp_image)
            result["sharp_laplacian_var"] = sharp_lap
            result["sharp_tenengrad"] = sharp_ten
            result["laplacian_ratio"] = blur_lap / max(sharp_lap, 1e-12)
            result["tenengrad_ratio"] = blur_ten / max(sharp_ten, 1e-12)
        # Save comparison
        group_dir = output_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        sequence_name = image_path.parent.parent.name if len(image_path.parents) >= 2 else "sequence"
        output_name = f"{rank:02d}_{sequence_name}_{image_path.stem}_score_{blur_score:.4f}.jpg"
        comparison_path = group_dir / output_name
        comparison = make_comparison(blur_image, sharp_image, blur_score, threshold)
        ok = cv2.imwrite(str(comparison_path), comparison)
        if not ok:
            raise RuntimeError(f"Failed to save comparison: {comparison_path}")
        result["comparison_image"] = str(comparison_path)
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result

# ===== CSV output =====
CSV_FIELDS = [
    "group",
    "rank",
    "image",
    "sharp_image",
    "blur_level",
    "threshold",
    "detected",
    "distance_to_threshold",
    "blur_laplacian_var",
    "sharp_laplacian_var",
    "laplacian_ratio",
    "blur_tenengrad",
    "sharp_tenengrad",
    "tenengrad_ratio",
    "comparison_image",
    "status",
    "error",
]

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

# ===== Summary =====
def print_group(name: str, rows: list[dict]):
    print(f"--- {name.upper()} ---")
    for row in rows:
        print(f"{int(row['rank']):02d} | blur={float(row['blur_level']):.4f} | detected={int(row['detected'])} | {Path(row['image']).name}")

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_gopro_rows(csv_path)
    if not rows:
        raise RuntimeError("No successful GoPro rows were found in the CSV.")
    print("=" * 70 + "GOPRO BLUR DIAGNOSIS" + "=" * 70)
    print(f"GoPro samples : {len(rows)}")
    print(f"Threshold : {args.threshold:.4f}")
    detected = sum(row["_blur"] >= args.threshold for row in rows)
    print(f"Detected : {detected}/{len(rows)} ({100.0 * detected / len(rows):.1f}%)")
    groups = select_groups(rows, threshold=args.threshold, count=args.count)
    diagnosis_rows = []
    for group_name in ("lowest", "borderline", "highest"):
        group_rows = []
        for rank, row in enumerate(groups[group_name], start=1):
            result = diagnose_sample(
                row=row,
                group=group_name,
                rank=rank,
                threshold=args.threshold,
                output_dir=output_dir,
            )
            diagnosis_rows.append(result)
            group_rows.append(result)
        print_group(group_name, group_rows)
    diagnosis_csv = output_dir / "gopro_blur_diagnosis.csv"
    write_csv(diagnosis_csv, diagnosis_rows)
    print("=" * 70 + "DIAGNOSIS COMPLETE" + "=" * 70)
    print(f"CSV: {diagnosis_csv}")
    print(f"Comparison images: {output_dir}")

if __name__ == "__main__":
    main()