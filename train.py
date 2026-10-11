"""Train a PatchCore anomaly detector on one MVTec AD category, locally or in Google Colab.

Both produce the same model bundle, which inference.py loads directly:

    models/<category>/model.pt        exported Torch model
    models/<category>/metadata.json   training settings and test metrics

Local (peaks at roughly 5-6 GB of RAM; use Colab on smaller machines):
    python train.py --category metal_nut

Google Colab: run notebooks/train_colab.ipynb. It clones this repo, trains on a GPU and saves
<category>_model.zip to Google Drive (or offers it as a download). Put that zip in the project root
on your computer and inference.py extracts it into models/ automatically.

Dataset: data/mvtec/<category>/{train,test,ground_truth}, or a data.zip (see dataset_zip.py) in the
project root or, on Colab, in the Drive folder. See README.md.
"""

import argparse
import json
import os
import platform
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import anomalib
import torch
from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning.pytorch import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parent
DRIVE_MOUNT = Path("/content/drive/MyDrive")
DEFAULT_DRIVE_DIR = DRIVE_MOUNT / "industrial_defect_detection"
EXTRACT_INCOMPLETE = ".extract_incomplete"  # present only while (or if interrupted during) extraction

BACKBONE = "wide_resnet50_2"
LAYERS = ("layer2", "layer3")
CORESET_SAMPLING_RATIO = 0.1


@dataclass(frozen=True)
class RunPaths:
    data_root: Path         # folder containing the category folders (extracted here from data_zip if needed)
    data_zip: Path | None   # fallback source when data_root has no usable category folder
    results_dir: Path       # Anomalib logs, checkpoints and test visualizations
    models_dir: Path        # lean model bundles read by inference.py
    drive_dir: Path | None  # Colab with Drive mounted: data.zip source and persistent copy of the bundle


def running_in_colab() -> bool:
    # COLAB_RELEASE_TAG is set in the Colab runtime and inherited by `!python ...` subprocesses.
    return "COLAB_RELEASE_TAG" in os.environ or (sys.platform == "linux" and Path("/content").is_dir())


def default_num_workers(mode: str) -> int:
    if mode == "colab":
        return 2  # Colab usually has 2 CPU cores
    # On Windows each DataLoader worker is a new process that re-imports torch and Anomalib (~1 GB each),
    # which ran an 8 GB machine out of memory during coreset selection. Load in the main process instead.
    if sys.platform == "win32":
        return 0
    return min(4, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchCore on an MVTec AD category (local or Colab).")
    parser.add_argument("--mode", choices=("auto", "local", "colab"), default="auto",
                        help="Where training runs. 'auto' detects Colab.")
    parser.add_argument("--category", default="metal_nut", help="MVTec AD category, e.g. metal_nut, screw, tile.")
    parser.add_argument("--data-root", type=Path, help="Folder with the category folders. Default: data/mvtec in the project.")
    parser.add_argument("--data-zip", type=Path, help="Zip to extract into --data-root if it is empty. "
                        "Default: data.zip in the project or, on Colab, in --drive-dir.")
    parser.add_argument("--drive-dir", type=Path, default=DEFAULT_DRIVE_DIR,
                        help="Colab only: Drive folder holding data.zip; also receives the model bundle.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, help="Default: 2 on Colab, 0 on Windows, up to 4 elsewhere.")
    parser.add_argument("--seed", type=int, default=42, help="Makes PatchCore's random coreset sampling repeatable.")
    parser.add_argument("--val-split", choices=("from_test", "same_as_test"), default="from_test",
                        help="from_test: tune the threshold on half the test set and report metrics on the other "
                             "half. same_as_test: Anomalib's default, which tunes and reports on the same images.")
    parser.add_argument("--skip-test", action="store_true", help="Skip evaluation after fitting.")
    parser.add_argument("--save-full-results", action="store_true",
                        help="Colab with Drive: also copy logs, checkpoint and test images to Drive (~450 MB).")
    return parser.parse_args()


def build_paths(args: argparse.Namespace, mode: str) -> RunPaths:
    """Use the same project-relative layout everywhere. On Colab the cloned repo sits on the VM's fast
    local disk; only if train.py itself is run from Drive do data and outputs move to /content."""
    work_root = PROJECT_ROOT
    if mode == "colab" and PROJECT_ROOT.is_relative_to("/content/drive"):
        work_root = Path("/content/defect-detection")
    drive_dir = args.drive_dir if mode == "colab" and DRIVE_MOUNT.is_dir() else None

    zip_candidates = [work_root / "data.zip"] + ([drive_dir / "data.zip"] if drive_dir else [])
    return RunPaths(
        data_root=args.data_root or work_root / "data" / "mvtec",
        data_zip=args.data_zip or next((z for z in zip_candidates if z.is_file()), None),
        results_dir=work_root / "results",
        models_dir=work_root / "models",
        drive_dir=drive_dir,
    )


def is_category_dir(path: Path) -> bool:
    return (path / "train" / "good").is_dir() and (path / "test").is_dir()


def find_category_dir(base: Path, category: str) -> Path | None:
    """Locate <...>/<category>/ under base, whatever folders the zip wrapped it in."""
    if not base.is_dir():
        return None
    if is_category_dir(base / category):
        return base / category
    return next((p for p in base.rglob(category) if p.is_dir() and is_category_dir(p)), None)


def can_extract_into(dest: Path) -> bool:
    """Only extract into a folder this script owns: missing, empty, or left by an interrupted extraction.
    Extracting into an existing dataset folder would nest a second copy of the dataset inside it."""
    return not dest.exists() or not any(dest.iterdir()) or (dest / EXTRACT_INCOMPLETE).is_file()


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract zip_path into dest. The marker is removed only after a full extraction,
    so an extraction interrupted by a crash or disconnect is redone on the next run."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / EXTRACT_INCOMPLETE).write_text(zip_path.name)
    print(f"      Extracting {zip_path} -> {dest} (this can take a few minutes)")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    (dest / EXTRACT_INCOMPLETE).unlink()


def resolve_dataset(paths: RunPaths, category: str) -> Path:
    """Return the dataset root (the parent of the category folder), extracting data_zip if needed."""
    category_dir = find_category_dir(paths.data_root, category)
    if category_dir is not None:
        return category_dir.parent

    if not can_extract_into(paths.data_root):
        available = sorted(p.name for p in paths.data_root.iterdir() if is_category_dir(p))
        raise FileNotFoundError(
            f"No '{category}' folder with train/good and test under {paths.data_root}. "
            f"Available categories: {', '.join(available) or 'none'}"
        )
    if paths.data_zip is None or not paths.data_zip.is_file():
        raise FileNotFoundError(
            f"No '{category}' dataset under {paths.data_root} and no data.zip found. Put the MVTec AD "
            f"'{category}' folder there, or pass a zip with --data-zip (see README.md)."
        )
    extract_zip(paths.data_zip, paths.data_root)
    category_dir = find_category_dir(paths.data_root, category)
    if category_dir is None:
        raise FileNotFoundError(f"{paths.data_zip.name} does not contain a '{category}' folder with "
                                f"train/good and test.")
    return category_dir.parent


def write_bundle(exported: Path, paths: RunPaths, metadata: dict) -> Path:
    """Move the exported model into models/<category>/ next to its metadata."""
    bundle_dir = paths.models_dir / metadata["category"]
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model_path = exported.replace(bundle_dir / "model.pt")
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return model_path


def zip_bundle(paths: RunPaths, category: str) -> Path:
    """Zip models/<category>/ as <category>_model.zip. Its layout (<category>/model.pt) extracts
    straight into the models/ folder that inference.py reads on another machine."""
    return Path(shutil.make_archive(
        str(paths.models_dir.parent / f"{category}_model"), "zip", root_dir=paths.models_dir, base_dir=category
    ))


def copy_to_drive(paths: RunPaths, drive_dir: Path, bundle_zip: Path, category: str,
                  save_full_results: bool) -> Path:
    drive_models = drive_dir / "models"
    drive_models.mkdir(parents=True, exist_ok=True)
    drive_bundle = Path(shutil.copy2(bundle_zip, drive_models))

    if save_full_results:
        full = shutil.make_archive(str(paths.models_dir.parent / f"{category}_results"), "zip", paths.results_dir)
        drive_results = drive_dir / "results"
        drive_results.mkdir(parents=True, exist_ok=True)
        print(f"      Full results: {shutil.copy2(full, drive_results)}")
    return drive_bundle


def train(args: argparse.Namespace, mode: str, paths: RunPaths) -> int:
    print(f"[1/6] Locating dataset (category: {args.category})")
    dataset_root = resolve_dataset(paths, args.category)
    print(f"      Dataset root: {dataset_root}")

    print(f"[2/6] Setting up MVTecAD datamodule (seed {args.seed}, validation split: {args.val_split})")
    seed_everything(args.seed, workers=True)
    num_workers = args.num_workers if args.num_workers is not None else default_num_workers(mode)
    datamodule = MVTecAD(
        root=dataset_root,
        category=args.category,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=num_workers,
        val_split_mode=args.val_split,
        val_split_ratio=0.5,
        seed=args.seed,
    )

    print("[3/6] Initializing PatchCore model")
    model = Patchcore(backbone=BACKBONE, layers=LAYERS, coreset_sampling_ratio=CORESET_SAMPLING_RATIO)
    engine = Engine(default_root_dir=paths.results_dir)

    print("[4/6] Fitting model (PatchCore builds its memory bank in a single epoch)")
    engine.fit(model=model, datamodule=datamodule)
    checkpoint = getattr(engine.trainer.checkpoint_callback, "best_model_path", "")
    if not checkpoint or not Path(checkpoint).is_file():
        raise RuntimeError("Training finished but no Lightning checkpoint was saved.")
    print(f"      Lightning checkpoint: {checkpoint}")

    metrics: dict[str, float] = {}
    if not args.skip_test:
        print("      Evaluating on the test split")
        results = engine.test(model=model, datamodule=datamodule, ckpt_path=checkpoint)
        metrics = {name: float(value) for name, value in (results[0] if results else {}).items()}

    print("[5/6] Exporting model bundle")
    exported = engine.export(model=model, export_type=ExportType.TORCH)
    if exported is None or not Path(exported).is_file():
        raise RuntimeError("Anomalib did not produce an exported Torch model.")
    metadata = {
        "category": args.category,
        "model": "Patchcore",
        "backbone": BACKBONE,
        "layers": list(LAYERS),
        "coreset_sampling_ratio": CORESET_SAMPLING_RATIO,
        "seed": args.seed,
        "val_split": args.val_split,
        "test_metrics": metrics,
        "trained_on": mode,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "anomalib_version": anomalib.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    model_path = write_bundle(Path(exported), paths, metadata)
    print(f"      Model bundle: {model_path.parent}")

    infer_cmd = f"python inference.py --category {args.category} --trust-model --image <image or folder>"
    if mode == "colab":
        print("[6/6] Packaging model bundle for your computer")
        location = zip_bundle(paths, args.category)
        if paths.drive_dir is not None:
            location = copy_to_drive(paths, paths.drive_dir, location, args.category, args.save_full_results)
        else:
            print("      Google Drive is not mounted, so the zip only exists until this Colab session ends.")
        print(f"      Saved: {location}")
        next_steps = (f"  1. Download {location}\n"
                      f"     and put it in the project's root folder on your computer.\n"
                      f"  2. {infer_cmd}")
    else:
        print("[6/6] Done: the model is ready for inference.py")
        next_steps = f"  {infer_cmd}"

    if metrics:
        print("\nTest metrics: " + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))
    print(f"\nTraining complete. Next:\n{next_steps}")
    return 0


def main() -> int:
    args = parse_args()
    mode = args.mode if args.mode != "auto" else ("colab" if running_in_colab() else "local")
    print(f"Running in {mode} mode")

    if mode == "colab" and not DRIVE_MOUNT.is_dir():
        print("Note: Google Drive is not mounted. The model zip will need to be downloaded from this session.")

    try:
        return train(args, mode, build_paths(args, mode))
    except (FileNotFoundError, zipfile.BadZipFile, RuntimeError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
