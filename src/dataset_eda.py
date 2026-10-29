"""
Dataset Exploratory Data Analysis & Integrity Verification
Potato Leaf Disease Detection Project
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import Counter
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TARGET_CLASSES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)

# Redirect Matplotlib cache to writable workspace directory
mpl_cache_dir = PROCESSED_DATA_DIR / ".mplconfig"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

def inspect_and_verify_dataset():
    print("=" * 65)
    print("🔬 PHASE 2: DATASET INSPECTION & INTEGRITY CHECK")
    print("=" * 65)
    
    # 1. Verify existence of raw directory
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found at {RAW_DATA_DIR}")
        
    found_classes = sorted([d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"Found class folders ({len(found_classes)}): {found_classes}")
    
    # Check that our target classes exist
    for target in TARGET_CLASSES:
        if target not in found_classes:
            raise ValueError(f"Missing required class: {target}")
    print("[OK] All 3 required classes verified!\n")

    # 2. Inspect every single image
    records = []
    corrupted_files = []
    dimensions_set = set()
    modes_set = set()
    
    print("Scanning images, checking file integrity & collecting dimensions...")
    for class_name in TARGET_CLASSES:
        class_folder = RAW_DATA_DIR / class_name
        image_paths = sorted(list(class_folder.glob("*.*")))
        
        for p in image_paths:
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
                
            try:
                with Image.open(p) as img:
                    img.verify()  # Verify image integrity
                
                # Reopen to read dimensions and mode
                with Image.open(p) as img:
                    w, h = img.size
                    mode = img.mode
                    dimensions_set.add((w, h))
                    modes_set.add(mode)
                    
                records.append({
                    "file_path": str(p.relative_to(RAW_DATA_DIR.parent.parent)).replace("\\", "/"),
                    "filename": p.name,
                    "class_name": class_name,
                    "width": w,
                    "height": h,
                    "mode": mode
                })
            except Exception as e:
                corrupted_files.append((str(p), str(e)))

    total_images = len(records)
    print(f"Total Valid Images Verified: {total_images}")
    print(f"Corrupted Images Found: {len(corrupted_files)}")
    if corrupted_files:
        print(f"⚠️ Corrupted files details: {corrupted_files}")
    else:
        print("[OK] Zero corrupted images! 100% files are healthy and readable.")
        
    print(f"Image Dimensions Found: {dimensions_set}")
    print(f"Color Modes Found: {modes_set}")
    
    # 3. Class Imbalance Analysis
    class_counts = Counter(r["class_name"] for r in records)
    print("\n" + "-" * 50)
    print("📊 CLASS DISTRIBUTION & IMBALANCE ANALYSIS")
    print("-" * 50)
    for cls in TARGET_CLASSES:
        count = class_counts[cls]
        pct = (count / total_images) * 100
        print(f"{cls:25s} : {count:5d} images ({pct:5.2f}%)")
        
    imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    print(f"\nImbalance Ratio (Max / Min): {imbalance_ratio:.2f}:1")
    print("Observation: 'Potato___healthy' is a minority class (152 images, ~7.06%).")
    print("Action Required: Stratified splitting to ensure consistent class representation across train/val/test.")
    print("-" * 50)

    # 4. Leakage-Free Stratified Split (Deterministic with RANDOM_SEED = 42)
    # Group by class to ensure exact class proportions in all splits
    class_buckets = {cls: [] for cls in TARGET_CLASSES}
    for r in records:
        class_buckets[r["class_name"]].append(r)

    # Sort deterministically before shuffling
    train_records, val_records, test_records = [], [], []
    for cls in TARGET_CLASSES:
        items = sorted(class_buckets[cls], key=lambda x: x["filename"])
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(items)
        
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # Remainder goes to test to guarantee exact total
        cls_train = items[:n_train]
        cls_val = items[n_train:n_train + n_val]
        cls_test = items[n_train + n_val:]
        
        for item in cls_train:
            item["split"] = "train"
        for item in cls_val:
            item["split"] = "val"
        for item in cls_test:
            item["split"] = "test"
            
        train_records.extend(cls_train)
        val_records.extend(cls_val)
        test_records.extend(cls_test)

    all_split_records = train_records + val_records + test_records

    # 5. Split summaries
    splits_summary = {}
    for split_name, split_list in [("train", train_records), ("val", val_records), ("test", test_records)]:
        cnts = Counter(r["class_name"] for r in split_list)
        splits_summary[split_name] = {
            "total": len(split_list),
            "percentage": round((len(split_list) / total_images) * 100, 2),
            "class_counts": dict(cnts)
        }

    print("\n" + "=" * 50)
    print("✂️ STRATIFIED SPLIT BREAKDOWN (SEED = 42)")
    print("=" * 50)
    for s_name, data in splits_summary.items():
        print(f"Split [{s_name.upper():5s}]: Total {data['total']:4d} ({data['percentage']}%)")
        for cls in TARGET_CLASSES:
            print(f"   - {cls:23s}: {data['class_counts'].get(cls, 0)}")

    # 6. Save Manifests and Summary
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROCESSED_DATA_DIR / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_split_records, f, indent=2)
    print(f"\n[OK] Split manifest written to {manifest_path}")

    summary_path = PROCESSED_DATA_DIR / "dataset_summary.json"
    summary_data = {
        "dataset_name": "PlantVillage Potato Subset",
        "dataset_source": "David Hughes and Marcel Salathé, Penn State / EPFL (2015)",
        "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)",
        "total_images": total_images,
        "corrupted_images": len(corrupted_files),
        "dimensions": [list(d) for d in dimensions_set],
        "color_modes": list(modes_set),
        "class_counts": dict(class_counts),
        "splits": splits_summary
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[OK] Summary statistics saved to {summary_path}")

    # 7. Generate EDA Visualization Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    classes_short = [c.replace("Potato___", "") for c in TARGET_CLASSES]
    counts_plot = [class_counts[c] for c in TARGET_CLASSES]
    colors = ["#e74c3c", "#e67e22", "#2ecc71"]
    
    axes[0].bar(classes_short, counts_plot, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("PlantVillage Potato: Class Distribution", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Number of Images")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    for i, v in enumerate(counts_plot):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")
        
    x = np.arange(len(classes_short))
    width = 0.25
    train_counts = [splits_summary["train"]["class_counts"][c] for c in TARGET_CLASSES]
    val_counts = [splits_summary["val"]["class_counts"][c] for c in TARGET_CLASSES]
    test_counts = [splits_summary["test"]["class_counts"][c] for c in TARGET_CLASSES]
    
    axes[1].bar(x - width, train_counts, width, label="Train (80%)", color="#3498db")
    axes[1].bar(x, val_counts, width, label="Val (10%)", color="#f1c40f")
    axes[1].bar(x + width, test_counts, width, label="Test (10%)", color="#9b59b6")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes_short)
    axes[1].set_title("Stratified Splits per Class (Leakage-Free)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Image Count")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    chart_path = PROCESSED_DATA_DIR / "eda_class_distribution.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"[OK] EDA Distribution chart saved to {chart_path}")
    print("=" * 65)

    return summary_data

if __name__ == "__main__":
    inspect_and_verify_dataset()
