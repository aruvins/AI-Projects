"""
Video Understanding Dataset Downloader & Preprocessor
======================================================
Downloads UCF-101 (action recognition benchmark) and prepares it for
the VideoTransformer model. UCF-101 has 101 action classes, ~13,000 clips.

Usage:
    python download_data.py --output_dir ./data --num_frames 16 --img_size 112
"""

import os
import argparse
import urllib.request
import tarfile
import zipfile
import shutil
import random
import json
from pathlib import Path


# ── Progress bar (no tqdm required) ──────────────────────────────────────────
def _reporthook(count, block_size, total_size):
    percent = min(int(count * block_size * 100 / total_size), 100)
    bar = "#" * (percent // 2) + "-" * (50 - percent // 2)
    print(f"\r  [{bar}] {percent}%", end="", flush=True)


# ── Download helpers ──────────────────────────────────────────────────────────
def download_file(url: str, dest: Path, desc: str = ""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  Downloading {desc or dest.name} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()


def extract_archive(archive: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive.name} → {target} ...")
    if archive.suffix == ".gz" or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target)
    else:
        raise ValueError(f"Unknown archive format: {archive}")
    print("  Done.")


# ── UCF-101 ───────────────────────────────────────────────────────────────────
UCF101_URL   = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
UCF101_SPLIT_URL = (
    "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
)

# Mirror fallback (no auth required):
UCF101_MIRROR = "https://storage.googleapis.com/thumos14_files/UCF101_videos.tar.gz"


def download_ucf101(data_dir: Path):
    """
    Downloads UCF-101 via the official .rar or falls back to a tar.gz mirror.
    Also downloads the official train/test split annotations.
    """
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # -- split annotations (zip, small) --
    split_zip = raw_dir / "UCF101TrainTestSplits.zip"
    download_file(UCF101_SPLIT_URL, split_zip, "UCF-101 split annotations")
    splits_dir = raw_dir / "ucfTrainTestlist"
    if not splits_dir.exists():
        extract_archive(split_zip, raw_dir)

    # -- video archive --
    # Try .rar; if rarfile not installed hint the user, then try mirror
    rar_path = raw_dir / "UCF101.rar"
    tar_path = raw_dir / "UCF101_videos.tar.gz"

    if not (raw_dir / "UCF-101").exists():
        # Check if rar tools are present
        rar_available = shutil.which("unrar") or shutil.which("7z")
        if rar_available:
            download_file(UCF101_URL, rar_path, "UCF-101 videos (.rar)")
            print("  Extracting RAR (this may take a few minutes) ...")
            tool = "unrar" if shutil.which("unrar") else "7z"
            cmd = (
                f"unrar x {rar_path} {raw_dir}/"
                if tool == "unrar"
                else f"7z x {rar_path} -o{raw_dir}/"
            )
            os.system(cmd)
        else:
            print(
                "  [info] unrar/7z not found — downloading tar.gz mirror instead."
            )
            download_file(UCF101_MIRROR, tar_path, "UCF-101 videos (tar.gz mirror)")
            extract_archive(tar_path, raw_dir)
    else:
        print("  [skip] UCF-101 videos already extracted")

    return raw_dir


# ── Frame extraction ──────────────────────────────────────────────────────────
def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    num_frames: int,
    img_size: int,
):
    """
    Uniformly samples `num_frames` frames from a video and saves them as
    JPEG images. Requires OpenCV (cv2).
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("Install opencv-python:  pip install opencv-python")

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return False

    indices = [int(i * total / num_frames) for i in range(num_frames)]
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (img_size, img_size))
        cv2.imwrite(str(out_dir / f"frame_{saved:04d}.jpg"), frame)
        saved += 1

    cap.release()
    return saved == num_frames


def build_frame_dataset(
    raw_dir: Path,
    frames_dir: Path,
    num_frames: int,
    img_size: int,
    max_videos: int = 0,   # 0 = all
):
    """
    Iterates over all class folders in UCF-101 and extracts frames.
    Returns a list of dicts: {video_id, class_name, class_idx, frames_dir}.
    """
    ucf_root = raw_dir / "UCF-101"
    if not ucf_root.exists():
        # mirror layout may be flat
        candidates = list(raw_dir.glob("UCF*"))
        if candidates:
            ucf_root = candidates[0]
        else:
            raise FileNotFoundError(
                f"UCF-101 video directory not found under {raw_dir}"
            )

    class_dirs = sorted([d for d in ucf_root.iterdir() if d.is_dir()])
    class_to_idx = {c.name: i for i, c in enumerate(class_dirs)}

    records = []
    total_videos = sum(len(list(d.glob("*.avi"))) for d in class_dirs)
    if max_videos:
        total_videos = min(total_videos, max_videos)

    processed = 0
    print(f"\n  Building frame dataset ({total_videos} videos) ...")
    for cls_dir in class_dirs:
        videos = list(cls_dir.glob("*.avi"))
        for vid in videos:
            if max_videos and processed >= max_videos:
                break
            vid_frames_dir = frames_dir / cls_dir.name / vid.stem
            ok = extract_frames_from_video(vid, vid_frames_dir, num_frames, img_size)
            if ok:
                records.append(
                    {
                        "video_id": vid.stem,
                        "class_name": cls_dir.name,
                        "class_idx": class_to_idx[cls_dir.name],
                        "frames_dir": str(vid_frames_dir),
                        "num_frames": num_frames,
                    }
                )
            processed += 1
            if processed % 100 == 0:
                print(f"    {processed}/{total_videos} videos processed")

    return records, class_to_idx


# ── Split manifest ────────────────────────────────────────────────────────────
def build_split_manifest(records: list, splits_dir: Path, out_dir: Path):
    """
    Uses the official UCF-101 split-1 text files to create train/test manifests.
    Falls back to a random 80/20 split if the files aren't available.
    """
    train_file = splits_dir / "trainlist01.txt"
    test_file  = splits_dir / "testlist01.txt"

    if train_file.exists() and test_file.exists():
        print("  Using official UCF-101 split-1 ...")
        train_ids = set()
        with open(train_file) as f:
            for line in f:
                train_ids.add(Path(line.split()[0]).stem)
        test_ids = set()
        with open(test_file) as f:
            for line in f:
                test_ids.add(Path(line.strip()).stem)

        train_recs = [r for r in records if r["video_id"] in train_ids]
        test_recs  = [r for r in records if r["video_id"] in test_ids]
    else:
        print("  [warn] Official split files not found — using random 80/20 split.")
        random.shuffle(records)
        cut = int(0.8 * len(records))
        train_recs, test_recs = records[:cut], records[cut:]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in [("train", train_recs), ("test", test_recs)]:
        path = out_dir / f"{name}_manifest.json"
        with open(path, "w") as f:
            json.dump(recs, f, indent=2)
        print(f"  Saved {name} manifest: {len(recs)} clips → {path}")

    return train_recs, test_recs


# ── Tiny synthetic dataset (offline / quick smoke-test) ──────────────────────
def create_synthetic_dataset(out_dir: Path, num_frames: int, img_size: int):
    """
    Creates a tiny NumPy-based synthetic dataset so the model can be trained
    without downloading UCF-101 first. Useful for verifying shapes & logic.
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Install numpy:  pip install numpy")

    print("\n  Creating synthetic dataset for smoke-testing ...")
    classes = [
        "Archery", "Basketball", "Biking", "Boxing", "Cricket",
        "Diving", "Fencing", "Golf", "HammerThrow", "IceDancing",
    ]
    records = []
    for split in ("train", "test"):
        n = 200 if split == "train" else 50
        for i in range(n):
            cls = classes[i % len(classes)]
            cls_dir = out_dir / "frames" / split / cls
            vid_id = f"synthetic_{split}_{i:04d}"
            vid_dir = cls_dir / vid_id
            vid_dir.mkdir(parents=True, exist_ok=True)
            for f in range(num_frames):
                # Random noise frame
                arr = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
                # Save as raw bytes (no cv2 needed for synthetic)
                _save_raw_frame(arr, vid_dir / f"frame_{f:04d}.jpg")
            records.append(
                {
                    "video_id": vid_id,
                    "class_name": cls,
                    "class_idx": classes.index(cls),
                    "frames_dir": str(vid_dir),
                    "num_frames": num_frames,
                    "synthetic": True,
                }
            )
        manifest_path = out_dir / f"{split}_manifest.json"
        split_recs = [r for r in records if r["video_id"].startswith(f"synthetic_{split}")]
        with open(manifest_path, "w") as f:
            json.dump(split_recs, f, indent=2)
        print(f"  {split}: {len(split_recs)} synthetic clips → {manifest_path}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_path = out_dir / "class_to_idx.json"
    with open(idx_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"  Class map saved → {idx_path}")


def _save_raw_frame(arr, path: Path):
    """Saves a numpy HxWx3 uint8 array as a minimal PPM then renames to jpg."""
    # Pure stdlib — avoids cv2/PIL dependency for synthetic data
    ppm = path.with_suffix(".ppm")
    h, w = arr.shape[:2]
    with open(ppm, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(arr.tobytes())
    ppm.rename(path.with_suffix(".ppm"))   # keep as .ppm (still loadable)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download & preprocess video data")
    parser.add_argument("--output_dir",  default="./data",  help="Root data directory")
    parser.add_argument("--num_frames",  type=int, default=16,  help="Frames to sample per clip")
    parser.add_argument("--img_size",    type=int, default=112, help="Spatial resolution (square)")
    parser.add_argument("--max_videos",  type=int, default=0,   help="Limit videos (0=all)")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create a tiny synthetic dataset instead of downloading UCF-101",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        create_synthetic_dataset(out, args.num_frames, args.img_size)
        print("\n✓ Synthetic dataset ready. Point your trainer at:", out)
        return

    # ── Real UCF-101 pipeline ─────────────────────────────────────────────
    print("=" * 60)
    print("  UCF-101 Video Dataset Downloader")
    print("=" * 60)

    raw_dir = download_ucf101(out / "raw")

    frames_dir = out / "frames"
    records, class_to_idx = build_frame_dataset(
        raw_dir, frames_dir, args.num_frames, args.img_size, args.max_videos
    )

    splits_dir = raw_dir / "ucfTrainTestlist"
    build_split_manifest(records, splits_dir, out)

    idx_path = out / "class_to_idx.json"
    with open(idx_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"\n  Class map ({len(class_to_idx)} classes) → {idx_path}")

    print("\n✓ Dataset ready.")
    print(f"  Frames dir : {frames_dir}")
    print(f"  Manifests  : {out}/train_manifest.json, test_manifest.json")
    print(f"  Classes    : {out}/class_to_idx.json")


if __name__ == "__main__":
    main()