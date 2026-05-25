# download_data.py

"""
Titanic Dataset Downloader
==========================

Downloads the Titanic dataset and saves it locally.

Run:
    python download_data.py
"""

import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
OUTPUT_DIR = "data"
OUTPUT_FILE = "titanic.csv"


def download_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    print("Downloading Titanic dataset...")
    print(f"URL: {DATA_URL}")
    print(f"Saving to: {output_path}")

    try:
        urllib.request.urlretrieve(DATA_URL, output_path)

        print("\nDownload complete!")
        print(f"Dataset saved to: {output_path}")

    except Exception as e:
        print("\nFailed to download dataset.")
        print(f"Error: {e}")


if __name__ == "__main__":
    download_dataset()