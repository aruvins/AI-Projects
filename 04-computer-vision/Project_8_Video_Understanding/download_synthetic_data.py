"""
Synthetic Video Dataset Generator
=================================
Creates a fully synthetic action-recognition dataset for testing
VideoTransformer pipelines without downloading UCF-101.

Usage:
    python download_data.py --output_dir ./data --num_frames 16 --img_size 112
"""

import argparse
import json
import random
from pathlib import Path

random.seed(42)


# ── Synthetic dataset generation ──────────────────────────────────────────────
def create_synthetic_dataset(
    out_dir: Path,
    num_frames: int,
    img_size: int,
    train_size: int = 200,
    test_size: int = 50,
):
    """
    Creates a synthetic video dataset using random image frames.

    Directory structure:
        data/
            frames/
                train/
                    ClassName/
                        video_xxxx/
                            frame_0000.jpg
                test/
                    ...
            train_manifest.json
            test_manifest.json
            class_to_idx.json
    """

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Install dependencies:\n"
            "pip install numpy pillow"
        )

    print("=" * 60)
    print("  Synthetic Video Dataset Generator")
    print("=" * 60)

    classes = [
        "Archery",
        "Basketball",
        "Biking",
        "Boxing",
        "Cricket",
        "Diving",
        "Fencing",
        "Golf",
        "HammerThrow",
        "IceDancing",
    ]

    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # Save class mapping
    idx_path = out_dir / "class_to_idx.json"
    with open(idx_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"  Saved class mapping → {idx_path}")

    splits = {
        "train": train_size,
        "test": test_size,
    }

    for split, num_videos in splits.items():

        records = []

        print(f"\n  Generating {split} set ({num_videos} videos)...")

        for i in range(num_videos):

            cls = classes[i % len(classes)]
            vid_id = f"{split}_{i:05d}"

            vid_dir = (
                out_dir
                / "frames"
                / split
                / cls
                / vid_id
            )

            vid_dir.mkdir(parents=True, exist_ok=True)

            # Generate synthetic frames
            for frame_idx in range(num_frames):

                arr = np.random.randint(
                    0,
                    256,
                    (img_size, img_size, 3),
                    dtype=np.uint8,
                )

                img = Image.fromarray(arr)

                frame_path = vid_dir / f"frame_{frame_idx:04d}.jpg"

                img.save(frame_path, quality=95)

            records.append(
                {
                    "video_id": vid_id,
                    "class_name": cls,
                    "class_idx": class_to_idx[cls],
                    "frames_dir": str(vid_dir),
                    "num_frames": num_frames,
                    "synthetic": True,
                }
            )

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{num_videos} videos generated")

        manifest_path = out_dir / f"{split}_manifest.json"

        with open(manifest_path, "w") as f:
            json.dump(records, f, indent=2)

        print(
            f"  Saved {split} manifest "
            f"({len(records)} videos) → {manifest_path}"
        )

    print("\n✓ Synthetic dataset ready.")
    print(f"  Dataset location: {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():

    parser = argparse.ArgumentParser(
        description="Synthetic video dataset generator"
    )

    parser.add_argument(
        "--output_dir",
        default="./data",
        help="Output dataset directory",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Frames per video clip",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=112,
        help="Frame size (square)",
    )

    parser.add_argument(
        "--train_size",
        type=int,
        default=200,
        help="Number of training videos",
    )

    parser.add_argument(
        "--test_size",
        type=int,
        default=50,
        help="Number of testing videos",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    create_synthetic_dataset(
        out_dir=out_dir,
        num_frames=args.num_frames,
        img_size=args.img_size,
        train_size=args.train_size,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()