"""
download_dataset.py
Downloads and prepares the TrashNet dataset from GitHub/Kaggle.
TrashNet: https://github.com/garythung/trashnet
~2527 images across 6 categories: cardboard, glass, metal, paper, plastic, trash
"""

import os
import zipfile
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm

DATASET_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# TrashNet hosted on Kaggle (most accessible mirror)
# Users can also download manually from: https://github.com/garythung/trashnet
KAGGLE_DATASET = "asdasdasasdas/garbage-classification"  # Mirror on Kaggle
GITHUB_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_trashnet():
    """Download TrashNet dataset."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_DIR / "trashnet.zip"

    print("=" * 60)
    print("  EcoLens — TrashNet Dataset Downloader")
    print("=" * 60)
    print(f"\nDownloading TrashNet dataset (~150MB)...")
    print(f"Source: {GITHUB_URL}\n")

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                                  desc="Downloading") as t:
            urllib.request.urlretrieve(GITHUB_URL, zip_path, reporthook=t.update_to)
        print("\n✅ Download complete!")
        return zip_path
    except Exception as e:
        print(f"\n❌ Direct download failed: {e}")
        print("\n📋 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Visit: https://github.com/garythung/trashnet")
        print("2. Download 'dataset-resized.zip' from the repository")
        print("3. OR use Kaggle: kaggle datasets download -d asdasdasasdas/garbage-classification")
        print(f"4. Place the zip file at: {zip_path.absolute()}")
        print("\nAlternatively, run: python download_dataset.py --kaggle")
        return None


def extract_dataset(zip_path: Path):
    """Extract and organize dataset."""
    print(f"\n📦 Extracting dataset...")
    extract_dir = DATASET_DIR / "extracted"
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    print("✅ Extraction complete!")
    return extract_dir


def organize_dataset(extract_dir: Path, split_ratio=(0.7, 0.15, 0.15)):
    """Organize dataset into train/val/test splits."""
    import random
    from PIL import Image

    print(f"\n🗂️  Organizing dataset into train/val/test splits...")
    print(f"   Split ratio: train={split_ratio[0]}, val={split_ratio[1]}, test={split_ratio[2]}")

    # Find images directory
    img_root = None
    for path in extract_dir.rglob("*"):
        if path.is_dir() and path.name in CLASSES:
            img_root = path.parent
            break

    if not img_root:
        # Try common structure
        for subdir in extract_dir.iterdir():
            if subdir.is_dir():
                img_root = subdir
                break

    if not img_root:
        print("❌ Could not locate image directories. Please check dataset structure.")
        return

    stats = {}
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        cls_dir = img_root / cls
        if not cls_dir.exists():
            # Try uppercase or different naming
            for candidate in img_root.iterdir():
                if candidate.name.lower() == cls.lower():
                    cls_dir = candidate
                    break

        if not cls_dir.exists():
            print(f"  ⚠️  Class directory not found: {cls}")
            continue

        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + \
                 list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.JPG"))

        random.shuffle(images)
        n = len(images)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            for img_path in split_images:
                dst = PROCESSED_DIR / split_name / cls / img_path.name
                shutil.copy2(img_path, dst)

        stats[cls] = {"total": n, "train": len(splits["train"]),
                      "val": len(splits["val"]), "test": len(splits["test"])}
        print(f"  ✅ {cls:<12}: {n:>4} images → train:{len(splits['train'])}, "
              f"val:{len(splits['val'])}, test:{len(splits['test'])}")

    print(f"\n✅ Dataset organized at: {PROCESSED_DIR.absolute()}")
    return stats


def create_sample_dataset():
    """Create a small sample dataset for testing without real data."""
    import numpy as np
    from PIL import Image
    import random

    print("\n🔧 Creating sample dataset for testing (50 images per class)...")

    COLORS = {
        "cardboard": (139, 90, 43),
        "glass": (100, 180, 200),
        "metal": (160, 160, 175),
        "paper": (240, 230, 200),
        "plastic": (200, 100, 120),
        "trash": (80, 80, 80),
    }

    for split in ["train", "val", "test"]:
        counts = {"train": 40, "val": 5, "test": 5}
        for cls in CLASSES:
            out_dir = PROCESSED_DIR / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(counts[split]):
                base_color = COLORS[cls]
                noise = np.random.randint(-30, 30, (224, 224, 3), dtype=np.int16)
                img_array = np.clip(
                    np.full((224, 224, 3), base_color, dtype=np.int16) + noise,
                    0, 255
                ).astype(np.uint8)
                img = Image.fromarray(img_array)
                img.save(out_dir / f"{cls}_{i:03d}.jpg")

    print("✅ Sample dataset created. Replace with real TrashNet data for production.")


if __name__ == "__main__":
    import sys

    if "--sample" in sys.argv:
        create_sample_dataset()
    elif "--kaggle" in sys.argv:
        print("Downloading via Kaggle API...")
        os.system(f"kaggle datasets download -d asdasdasasdas/garbage-classification -p {DATASET_DIR}")
        zip_files = list(DATASET_DIR.glob("*.zip"))
        if zip_files:
            extract_dir = extract_dataset(zip_files[0])
            organize_dataset(extract_dir)
    else:
        zip_path = download_trashnet()
        if zip_path and zip_path.exists():
            extract_dir = extract_dataset(zip_path)
            organize_dataset(extract_dir)
        else:
            print("\nFalling back to sample dataset...")
            create_sample_dataset()
