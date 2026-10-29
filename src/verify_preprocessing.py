"""
Preprocessing & Augmentation Pipeline Verification Script
Potato Leaf Disease Detection Project
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    PROCESSED_DATA_DIR,
)

# Redirect Matplotlib cache to writable workspace directory
mpl_cache_dir = PROCESSED_DATA_DIR / ".mplconfig"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import (
    get_train_val_test_datasets,
    get_data_augmentation_layers,
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
)

def verify_preprocessing():
    print("=" * 65)
    print("[PHASE 3: PREPROCESSING & PIPELINE VERIFICATION]")
    print("=" * 65)
    
    print("Building train, val, and test tf.data pipelines...")
    train_ds, val_ds, test_ds = get_train_val_test_datasets(batch_size=BATCH_SIZE)
    
    # 1. Inspect Train Batch
    print("\nFetching first batch from Training pipeline...")
    for images, labels in train_ds.take(1):
        print(f"  - Images Batch Shape: {images.shape}")
        print(f"  - Labels Batch Shape: {labels.shape}")
        print(f"  - Image Dtype: {images.dtype}")
        print(f"  - Pixel Range: [{images.numpy().min():.4f}, {images.numpy().max():.4f}]")
        
        assert images.shape == (BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), "Train image shape mismatch!"
        assert labels.shape == (BATCH_SIZE,), "Train label shape mismatch!"
        assert images.dtype == tf.float32, "Train image dtype must be float32!"
        assert 0.0 <= images.numpy().min() and images.numpy().max() <= 1.0, "Pixels must be normalized to [0, 1]!"
    print("[OK] Training batch dimensions and normalization verified.")

    # 2. Inspect Validation Batch
    print("\nFetching first batch from Validation pipeline...")
    for images, labels in val_ds.take(1):
        print(f"  - Images Batch Shape: {images.shape}")
        print(f"  - Labels Batch Shape: {labels.shape}")
        assert images.shape == (BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        assert 0.0 <= images.numpy().min() and images.numpy().max() <= 1.0
    print("[OK] Validation batch dimensions and normalization verified.")

    # 3. Inspect Test Batch
    print("\nFetching first batch from Test pipeline...")
    for images, labels in test_ds.take(1):
        print(f"  - Images Batch Shape: {images.shape}")
        print(f"  - Labels Batch Shape: {labels.shape}")
        assert images.shape == (BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        assert 0.0 <= images.numpy().min() and images.numpy().max() <= 1.0
    print("[OK] Test batch dimensions and normalization verified.")

    # 4. Demonstrate & Visualize Data Augmentation
    print("\nGenerating visual comparison of Data Augmentation...")
    aug_pipeline = get_data_augmentation_layers()
    
    # Get a sample image from val_ds (un-augmented original)
    sample_img = None
    sample_lbl = None
    for imgs, lbls in val_ds.take(1):
        sample_img = imgs[0]
        sample_lbl = int(lbls[0].numpy())
        break
        
    class_name = INDEX_TO_CLASS[sample_lbl]
    
    # Create 5 variations
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    axes[0].imshow(sample_img.numpy())
    axes[0].set_title(f"Original\n({class_name})", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    
    img_expanded = tf.expand_dims(sample_img, axis=0)
    for i in range(1, 5):
        aug_img = aug_pipeline(img_expanded, training=True)[0]
        axes[i].imshow(aug_img.numpy())
        axes[i].set_title(f"Augmented #{i}", fontsize=10)
        axes[i].axis("off")
        
    plt.tight_layout()
    aug_plot_path = PROCESSED_DATA_DIR / "augmentation_samples.png"
    plt.savefig(aug_plot_path, dpi=200)
    plt.close()
    print(f"[OK] Augmentation sample grid saved to {aug_plot_path}")

    print("\n" + "=" * 65)
    print("All preprocessing checks passed successfully!")
    print("=" * 65)

if __name__ == "__main__":
    verify_preprocessing()
