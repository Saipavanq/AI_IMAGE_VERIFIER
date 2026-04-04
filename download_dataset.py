"""
download_dataset.py — Download the CIFAKE dataset from Kaggle or HuggingFace.

Usage (requires Kaggle API key at ~/.kaggle/kaggle.json):
  python download_dataset.py

Or set env vars:
  KAGGLE_USERNAME and KAGGLE_KEY
"""

import os
import subprocess
import sys
from pathlib import Path


DATA_DIR   = Path("data/raw")
KAGGLE_DS  = "birdy654/cifake-real-and-ai-generated-synthetic-images"


def check_structure():
    expected = [
        DATA_DIR / "train" / "REAL",
        DATA_DIR / "train" / "FAKE",
        DATA_DIR / "test"  / "REAL",
        DATA_DIR / "test"  / "FAKE",
    ]
    if all(p.exists() for p in expected):
        counts = {str(p): len(list(p.glob("*"))) for p in expected}
        print("[Check] Dataset already present:")
        for k, v in counts.items():
            print(f"  {k}: {v} images")
        return True
    return False


def download_kaggle():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if kaggle.json exists
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"\n[Error] Kaggle API key not found at: {kaggle_json}")
        print("\nTo get your API key:")
        print("  1. Go to https://www.kaggle.com/settings/account")
        print("  2. Click 'Create New API Token'")
        print(f"  3. Save kaggle.json to: C:\\Users\\malla\\.kaggle\\kaggle.json")
        return False

    print(f"[Download] Fetching CIFAKE from Kaggle → {DATA_DIR}/")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DS, path=str(DATA_DIR), unzip=True)
        print("[Download] Complete.")
        return True
    except Exception as e:
        print(f"[Error] Kaggle download failed: {e}")
        return False



def main():
    print("=" * 60)
    print("  CIFAKE Dataset Downloader")
    print("=" * 60)

    if check_structure():
        print("[Skip] Dataset already downloaded.")
        return

    print("[Info] Dataset not found. Attempting Kaggle download …")
    print("[Info] Make sure kaggle.json is in ~/.kaggle/ or set env vars:")
    print("       KAGGLE_USERNAME and KAGGLE_KEY\n")

    success = download_kaggle()
    if not success:
        print("\n[Manual] Download manually from:")
        print(f"  https://www.kaggle.com/datasets/{KAGGLE_DS}")
        print(f"\nExtract to: {DATA_DIR.resolve()}/")
        print("Structure should be:")
        print("  data/raw/train/REAL/   data/raw/train/FAKE/")
        print("  data/raw/test/REAL/    data/raw/test/FAKE/")
        sys.exit(1)

    check_structure()


if __name__ == "__main__":
    main()
