"""
Unit tests for Dataset Integrity and Stratified Splitting (Phase 2).
"""

import json
from pathlib import Path
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_CLASSES

def test_raw_dataset_classes_exist():
    assert RAW_DATA_DIR.exists(), "data/raw directory must exist"
    subdirs = [d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    for target_class in TARGET_CLASSES:
        assert target_class in subdirs, f"Class {target_class} must exist in data/raw"

def test_raw_dataset_image_counts():
    counts = {}
    for cls in TARGET_CLASSES:
        cls_dir = RAW_DATA_DIR / cls
        # Use case-insensitive extension check without duplicate globbing
        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        counts[cls] = len(images)
    
    assert counts["Potato___Early_blight"] == 1000
    assert counts["Potato___Late_blight"] == 1000
    assert counts["Potato___healthy"] == 152
    assert sum(counts.values()) == 2152

def test_manifest_and_no_data_leakage():
    manifest_path = PROCESSED_DATA_DIR / "split_manifest.json"
    assert manifest_path.exists(), "split_manifest.json must exist in data/processed"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    assert len(records) == 2152, "Manifest must contain all 2152 images"
    
    train_paths = {r["file_path"] for r in records if r["split"] == "train"}
    val_paths = {r["file_path"] for r in records if r["split"] == "val"}
    test_paths = {r["file_path"] for r in records if r["split"] == "test"}
    
    assert len(train_paths) == 1721
    assert len(val_paths) == 215
    assert len(test_paths) == 216
    
    # Critical test: ZERO DATA LEAKAGE (disjoint sets)
    assert len(train_paths.intersection(val_paths)) == 0, "Data leakage detected between train and val!"
    assert len(train_paths.intersection(test_paths)) == 0, "Data leakage detected between train and test!"
    assert len(val_paths.intersection(test_paths)) == 0, "Data leakage detected between val and test!"

def test_stratified_representation():
    manifest_path = PROCESSED_DATA_DIR / "split_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    for split_name in ["train", "val", "test"]:
        split_records = [r for r in records if r["split"] == split_name]
        split_classes = {r["class_name"] for r in split_records}
        # Ensure all 3 classes are represented in every split
        assert split_classes == set(TARGET_CLASSES), f"Split {split_name} is missing classes!"
