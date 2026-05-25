import os
import urllib.request
import zipfile
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")
EXTRACT_PATH = os.path.join(DATA_DIR, "smsspamcollection")


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(ZIP_PATH):
        print("📦 Dataset already downloaded.")
        return

    print("⬇️ Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    print("✅ Download complete.")


def extract_dataset():
    print("📂 Extracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

    print("✅ Extraction complete.")


def build_csv():
    file_path = os.path.join(EXTRACT_PATH, "SMSSpamCollection")

    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset file not found after extraction.")

    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            label, text = line.strip().split("\t")
            data.append((label, text))

    df = pd.DataFrame(data, columns=["label", "text"])

    csv_path = os.path.join(DATA_DIR, "spam.csv")
    df.to_csv(csv_path, index=False)

    print(f"📊 CSV saved to: {csv_path}")
    print(df.head())


def fallback_synthetic():
    print("⚠️ Using synthetic dataset fallback...")

    data = [
        ("ham", "Hey are we still meeting today?"),
        ("spam", "WIN a FREE iphone now!!!"),
        ("ham", "Please review the report"),
        ("spam", "Congratulations you won a lottery prize"),
        ("ham", "Let's grab lunch tomorrow"),
        ("spam", "Urgent! Your account has been suspended"),
    ]

    df = pd.DataFrame(data, columns=["label", "text"])

    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "spam.csv")
    df.to_csv(csv_path, index=False)

    print(f"📊 Synthetic CSV saved to: {csv_path}")


def main():
    try:
        download_dataset()
        extract_dataset()
        build_csv()
    except Exception as e:
        print("❌ Error occurred:", e)
        fallback_synthetic()


if __name__ == "__main__":
    main()