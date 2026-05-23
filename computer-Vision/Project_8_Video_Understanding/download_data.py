"""
download_ucf101.py
==================

Downloads the UCF101 action recognition dataset (.rar archive)
from the official UCF website with a progress bar.

Usage:
    python download_data.py
    python download_data.py --output_dir ./data
"""

import argparse
import os
import ssl
import sys
import urllib.request
from pathlib import Path


UCF101_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"


def download_file(url: str, output_path: Path):
    """
    Download file with progress bar.
    """

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size

        if total_size > 0:
            percent = downloaded / total_size * 100
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)

            sys.stdout.write(
                f"\rDownloading: {percent:6.2f}% "
                f"({downloaded_mb:.2f} MB / {total_mb:.2f} MB)"
            )
            sys.stdout.flush()

    ssl_context = ssl._create_unverified_context()
    
    print(f"\nDownloading UCF101 dataset...")
    print(f"URL: {url}")
    print(f"Save path: {output_path}")

    
    with urllib.request.urlopen(url, context=ssl_context) as response:
        total_size = int(response.headers.get("Content-Length", 0))

        with open(output_path, "wb") as f:
            downloaded = 0
            block_size = 8192

            while True:
                buffer = response.read(block_size)

                if not buffer:
                    break

                f.write(buffer)
                downloaded += len(buffer)

                progress_hook(
                    downloaded // block_size,
                    block_size,
                    total_size,
                )

    print("\n✓ Download complete!")


def main():
    parser = argparse.ArgumentParser(description="Download UCF101 dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save the dataset",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "UCF101.rar"

    if output_file.exists():
        print(f"File already exists: {output_file}")
        return

    try:
        download_file(UCF101_URL, output_file)

        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"Saved to: {output_file}")
        print(f"File size: {file_size:.2f} MB")

    except Exception as e:
        print(f"\nError downloading dataset:")
        print(e)

        if output_file.exists():
            os.remove(output_file)

        sys.exit(1)


if __name__ == "__main__":
    main()