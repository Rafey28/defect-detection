"""Run a trained PatchCore model on test images and save Original vs. Heatmap Overlay comparisons.

The model is looked up in this order (see resolve_weights):
    1. --weights, if given
    2. models/<category>/model.pt                      written by train.py (local run)
    3. <category>_model.zip in the project root or models/   downloaded from Drive after a Colab run
    4. results/.../<category>/v*/weights/torch/model.pt  older runs, extracting results.zip if needed

Example:
    python inference.py --category metal_nut --trust-model --image ./data/mvtec/metal_nut/test/scratch/000.png
"""

import argparse
import json
import os
import pathlib
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # render to files only; no display window needed
import matplotlib.pyplot as plt
import numpy as np
from anomalib.deploy import TorchInferencer

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LEGACY_RESULTS_ZIP = PROJECT_ROOT / "results.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_inference"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect defects in images with a trained PatchCore model.")
    parser.add_argument("--category", default="metal_nut", help="Category the model was trained on.")
    parser.add_argument("--weights", type=Path, help="Exported Torch model (.pt). Default: found via --category.")
    parser.add_argument("--image", type=Path, required=True, help="A single image or a directory of images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where visualizations go.")
    parser.add_argument("--trust-model", action="store_true",
                        help="Allow loading the .pt file, which runs pickled code. Only use with models you trained.")
    return parser.parse_args()


def extract_bundle_zip(category: str) -> Path | None:
    """Extract <category>_model.zip (made by train.py on Colab) into models/."""
    for zip_path in (PROJECT_ROOT / f"{category}_model.zip", MODELS_DIR / f"{category}_model.zip"):
        if not zip_path.is_file():
            continue
        print(f"      Extracting {zip_path} -> {MODELS_DIR}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(MODELS_DIR)
        model_path = MODELS_DIR / category / "model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"{zip_path.name} does not contain {category}/model.pt")
        return model_path
    return None


def find_legacy_model(category: str) -> Path | None:
    """Find a model from runs made before train.py wrote models/<category>/.

    The first Colab run's results.zip wraps results/<category>_results.zip, so both levels are unpacked.
    """
    pattern = f"**/{category}/v*/weights/torch/model.pt"
    candidates = list(RESULTS_DIR.glob(pattern))
    if not candidates and LEGACY_RESULTS_ZIP.is_file():
        print(f"      Extracting {LEGACY_RESULTS_ZIP}")
        with zipfile.ZipFile(LEGACY_RESULTS_ZIP) as zf:
            zf.extractall(PROJECT_ROOT)
        for inner in RESULTS_DIR.glob("*_results.zip"):
            if not (RESULTS_DIR / inner.stem).exists():
                with zipfile.ZipFile(inner) as zf:
                    zf.extractall(RESULTS_DIR / inner.stem)
        candidates = list(RESULTS_DIR.glob(pattern))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def resolve_weights(weights: Path | None, category: str) -> Path:
    if weights is not None:
        if weights.is_file():
            return weights
        raise FileNotFoundError(f"Model file not found: {weights}")

    bundled = MODELS_DIR / category / "model.pt"
    if bundled.is_file():
        return bundled
    model_path = extract_bundle_zip(category) or find_legacy_model(category)
    if model_path is None:
        raise FileNotFoundError(
            f"No model for '{category}'. Train one with: python train.py --category {category}\n"
            f"       or, after training on Colab, copy {category}_model.zip from Drive into {PROJECT_ROOT}"
        )
    return model_path


def print_model_info(weights: Path, category: str) -> None:
    metadata_path = weights.parent / "metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("category") != category:
        print(f"      WARNING: model was trained on '{metadata.get('category')}', not '{category}'")
    auroc = metadata.get("test_metrics", {}).get("image_AUROC")
    details = f"trained {metadata.get('trained_at', '?')} on {metadata.get('trained_on', '?')}"
    print(f"      {details}" + (f", test image AUROC {auroc:.4f}" if auroc is not None else ""))


@contextmanager
def posix_paths_as_windows():
    """Let a model pickled on Linux (e.g. Colab) load on Windows.

    The exported model contains PosixPath objects, which cannot be instantiated on Windows,
    so they are temporarily unpickled as WindowsPath instead.
    """
    if os.name != "nt":
        yield
        return
    original = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = original


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            return images
        raise FileNotFoundError(f"No images found in directory: {path}")
    raise FileNotFoundError(f"Image path not found: {path}")


def to_numpy(value) -> np.ndarray:
    """Convert a tensor/array prediction field to a NumPy array without the batch/channel dims."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.squeeze(np.asarray(value))


def make_overlay(image_rgb: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a JET-colored anomaly map over the original image."""
    height, width = image_rgb.shape[:2]
    heat = cv2.resize(anomaly_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    # PatchCore's post-processor normalizes maps to [0, 1]; min-max only if the map is unnormalized.
    if heat.max() > 1.0 or heat.min() < 0.0:
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, heat_color, alpha, 0)


def save_comparison(image_rgb: np.ndarray, overlay: np.ndarray, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[1].imshow(overlay)
    axes[1].set_title("Heatmap Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    print("[1/3] Locating model weights")
    try:
        weights = resolve_weights(args.weights, args.category)
        images = collect_images(args.image)
    except (FileNotFoundError, zipfile.BadZipFile) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(f"[2/3] Loading model from {weights}")
    print_model_info(weights, args.category)
    if args.trust_model:
        os.environ["TRUST_REMOTE_CODE"] = "1"
    try:
        with posix_paths_as_windows():
            inferencer = TorchInferencer(path=weights, device="auto")
    except ValueError as err:
        if "TRUST_REMOTE_CODE" not in str(err):
            raise
        print("ERROR: Anomalib blocks loading .pt files by default because they can run code. "
              "If you trained this model yourself, rerun with --trust-model.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[3/3] Running inference on {len(images)} image(s)\n")

    for image_path in images:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"  SKIP {image_path}: could not read image", file=sys.stderr)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        predictions = inferencer.predict(image=image_path)
        score = float(to_numpy(predictions.pred_score))
        is_defective = bool(to_numpy(predictions.pred_label))
        anomaly_map = to_numpy(predictions.anomaly_map)

        verdict = "DEFECTIVE" if is_defective else "NORMAL"
        print(f"  {image_path}: {verdict} (anomaly score: {score:.4f})")

        overlay = make_overlay(image_rgb, anomaly_map)
        out_path = args.output_dir / f"{image_path.parent.name}_{image_path.stem}_result.png"
        save_comparison(image_rgb, overlay, f"{verdict} | score: {score:.4f}", out_path)
        print(f"    saved -> {out_path}")

    print("\nInference complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
