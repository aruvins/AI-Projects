import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse


# =========================
# DATASET CONFIG
# =========================

DATASETS = {
    # NeRF dataset (small, fast, essential for training)
    "tiny_nerf": {
        "url": "https://github.com/bmild/nerf/raw/master/data/tiny_nerf_data.npz",
        "path": "data/nerf/tiny_nerf_data.npz",
        "type": "file"
    },

    # SfM dataset (classic multiview reconstruction)
    "temple": {
        "url": "https://vision.middlebury.edu/mview/data/data/temple.zip",
        "path": "data/sfm/temple.zip",
        "type": "zip"
    },

    # Stereo dataset (KITTI scene flow subset)
    "kitti_stereo": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip",
        "path": "data/stereo/kitti.zip",
        "type": "zip"
    }
}


# =========================
# UTILITIES
# =========================

def create_dirs():
    """Create required dataset directories."""
    paths = [
        "data/stereo",
        "data/sfm",
        "data/nerf",
        "outputs"
    ]
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def download_file(url, output_path):
    """Download large files with progress bar (streaming)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    print(f"\nDownloading: {url}")
    print(f"Saving to: {output_path}")

    with open(output_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=os.path.basename(output_path)
    ) as bar:

        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print("Download complete.\n")


# =========================
# SAFE EXTRACTION
# =========================

def safe_extract(zip_path, extract_to):
    """
    Prevents zip path traversal attacks (zip-slip protection).
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            member_path = extract_to / member

            # Security check
            if not str(member_path.resolve()).startswith(str(extract_to.resolve())):
                raise Exception(f"Unsafe zip file detected: {member}")

        zip_ref.extractall(extract_to)

    print(f"Extracted to: {extract_to}\n")


# =========================
# MAIN PIPELINE
# =========================

def download_dataset(name, config):
    url = config["url"]
    path = config["path"]
    dtype = config["type"]

    if Path(path).exists():
        print(f"[SKIP] {name} already exists at {path}")
        return

    download_file(url, path)

    if dtype == "zip":
        extract_dir = Path(path).with_suffix("")  # remove .zip
        safe_extract(path, extract_dir)


def download_all():
    create_dirs()

    print("\n==============================")
    print("  3D CV DATASET DOWNLOADER")
    print("==============================\n")

    for name, config in DATASETS.items():
        print(f"==> Processing {name}")
        download_dataset(name, config)

    print("\nAll datasets downloaded successfully.")
    print("Ready for Stereo + SfM + NeRF pipeline.\n")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    download_all()