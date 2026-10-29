"""
Dataset Preprocessing and tf.data Input Pipeline
Potato Leaf Disease Detection Project
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from src.config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    TARGET_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
)

# Deterministic class mapping
CLASS_TO_INDEX: Dict[str, int] = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
INDEX_TO_CLASS: Dict[int, str] = {idx: cls for idx, cls in enumerate(TARGET_CLASSES)}

def get_data_augmentation_layers() -> tf.keras.Sequential:
    """
    Creates the Data Augmentation layer pipeline.
    Applied exclusively to training images to simulate real-world leaf orientations.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip(
            "horizontal_and_vertical",
            seed=RANDOM_SEED,
            name="aug_random_flip"
        ),
        tf.keras.layers.RandomRotation(
            0.1,  # Small rotation: +/- 10% of 2pi (~36 degrees)
            fill_mode="reflect",
            seed=RANDOM_SEED,
            name="aug_random_rotation"
        ),
        tf.keras.layers.RandomZoom(
            0.1,  # Moderate zoom: +/- 10%
            fill_mode="reflect",
            seed=RANDOM_SEED,
            name="aug_random_zoom"
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,  # Small translation: +/- 5% height
            width_factor=0.05,   # Small translation: +/- 5% width
            fill_mode="reflect",
            seed=RANDOM_SEED,
            name="aug_random_translation"
        ),
    ], name="data_augmentation_pipeline")

def parse_and_preprocess_image(file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Reads an image file from disk, decodes JPEG, resizes to target resolution (224, 224),
    and normalizes pixel intensities to [0.0, 1.0].
    """
    # 1. Read binary content from disk
    img_raw = tf.io.read_file(file_path)
    
    # 2. Decode JPEG to uint8 tensor with 3 color channels (RGB)
    img = tf.io.decode_jpeg(img_raw, channels=3)
    
    # 3. Resize to standard target resolution (224, 224) using bilinear interpolation
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    
    # 4. Normalize pixel values from [0, 255] to [0.0, 1.0]
    img = tf.cast(img, tf.float32) / 255.0
    
    return img, label

def load_split_records(split_name: str) -> Tuple[List[str], List[int]]:
    """
    Loads file paths and corresponding integer labels for a given split from split_manifest.json.
    """
    manifest_path = PROCESSED_DATA_DIR / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {manifest_path}. Please run dataset_eda.py first."
        )
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    split_items = [item for item in manifest if item["split"] == split_name]
    if not split_items:
        raise ValueError(f"No records found for split '{split_name}' in manifest.")

    file_paths = []
    labels = []
    for item in split_items:
        full_path = str(BASE_DIR / item["file_path"])
        file_paths.append(full_path)
        labels.append(CLASS_TO_INDEX[item["class_name"]])
        
    return file_paths, labels

def create_dataset_pipeline(
    split_name: str,
    batch_size: int = BATCH_SIZE,
    augment: bool = False,
    shuffle: bool = False,
    cache: bool = True
) -> tf.data.Dataset:
    """
    Constructs an optimized tf.data.Dataset pipeline.
    
    Args:
        split_name: 'train', 'val', or 'test'
        batch_size: Number of samples per batch
        augment: Whether to apply data augmentation (must be True ONLY for train)
        shuffle: Whether to shuffle elements (must be True for train)
        cache: Whether to cache elements in memory for faster epoch iterations
    """
    file_paths, labels = load_split_records(split_name)
    num_samples = len(file_paths)
    
    # 1. Create dataset from file paths and integer labels
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # 2. Shuffle paths if requested before loading images
    if shuffle:
        ds = ds.shuffle(buffer_size=num_samples, seed=RANDOM_SEED, reshuffle_each_iteration=True)
        
    # 3. Map preprocessing function across files (parallelized across CPU cores)
    ds = ds.map(parse_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 4. Cache decoded & resized images in memory to avoid repeated disk reads
    if cache:
        ds = ds.cache()
        
    # 5. Apply data augmentation if requested (TRAIN ONLY)
    if augment:
        aug_layers = get_data_augmentation_layers()
        ds = ds.map(
            lambda x, y: (aug_layers(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
    # 6. Batch the dataset
    ds = ds.batch(batch_size)
    
    # 7. Prefetch batches to overlap data preprocessing with model execution
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds

def get_train_val_test_datasets(
    batch_size: int = BATCH_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Convenience function returning (train_ds, val_ds, test_ds) with industry-standard configs:
    - Train: augmented=True, shuffled=True, cached=True
    - Val: augmented=False, shuffled=False, cached=True
    - Test: augmented=False, shuffled=False, cached=True
    """
    train_ds = create_dataset_pipeline("train", batch_size=batch_size, augment=True, shuffle=True)
    val_ds = create_dataset_pipeline("val", batch_size=batch_size, augment=False, shuffle=False)
    test_ds = create_dataset_pipeline("test", batch_size=batch_size, augment=False, shuffle=False)
    return train_ds, val_ds, test_ds
