"""
Unit tests for Image Preprocessing and Augmentation Pipelines (Phase 3).
"""

import tensorflow as tf
import numpy as np
from src.config import TARGET_CLASSES, IMAGE_SIZE, BATCH_SIZE
from src.dataset import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    parse_and_preprocess_image,
    get_data_augmentation_layers,
    create_dataset_pipeline,
    load_split_records,
)

def test_class_mappings():
    assert len(CLASS_TO_INDEX) == 3
    assert len(INDEX_TO_CLASS) == 3
    for idx, cls in enumerate(TARGET_CLASSES):
        assert CLASS_TO_INDEX[cls] == idx
        assert INDEX_TO_CLASS[idx] == cls

def test_single_image_preprocessing():
    file_paths, labels = load_split_records("test")
    assert len(file_paths) > 0
    
    img_tensor, label_tensor = parse_and_preprocess_image(
        tf.constant(file_paths[0]),
        tf.constant(labels[0])
    )
    
    assert img_tensor.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    assert img_tensor.dtype == tf.float32
    assert img_tensor.numpy().min() >= 0.0
    assert img_tensor.numpy().max() <= 1.0

def test_augmentation_layers():
    aug = get_data_augmentation_layers()
    dummy_input = tf.random.uniform((2, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), minval=0.0, maxval=1.0)
    augmented = aug(dummy_input, training=True)
    
    assert augmented.shape == dummy_input.shape
    assert augmented.dtype == tf.float32

def test_dataset_pipeline_batch_shapes():
    val_ds = create_dataset_pipeline("val", batch_size=16, augment=False, shuffle=False)
    for images, labels in val_ds.take(1):
        assert images.shape == (16, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        assert labels.shape == (16,)
        assert images.dtype == tf.float32
        assert images.numpy().min() >= 0.0
        assert images.numpy().max() <= 1.0
