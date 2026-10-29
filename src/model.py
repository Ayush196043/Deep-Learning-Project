"""
Deep Learning Model Architectures
Potato Leaf Disease Detection Project
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from src.config import IMAGE_SHAPE, NUM_CLASSES

def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = IMAGE_SHAPE,
    num_classes: int = NUM_CLASSES
) -> tf.keras.Model:
    """
    Builds a custom Baseline Convolutional Neural Network (CNN).
    
    Architecture:
    - Input: (224, 224, 3) normalized RGB image
    - 4 Convolutional blocks with increasing filter depths (32 -> 64 -> 128 -> 256)
    - ReLU activation for non-linearity
    - MaxPooling2D (2x2) for spatial downsampling and translation invariance
    - GlobalAveragePooling2D head to prevent parameter explosion and overfitting
    - Dense layer (128 units, ReLU)
    - Dropout (0.4) for regularization
    - Dense output layer with Softmax activation over 3 classes
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # Block 1: Low-level features (edges, color transitions)
    # Output: 224x224x32 -> 112x112x32
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    
    # Block 2: Textures & leaf surface patterns
    # Output: 112x112x64 -> 56x56x64
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    
    # Block 3: Complex lesion shapes & spot borders
    # Output: 56x56x128 -> 28x28x128
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    
    # Block 4: High-level semantic disease symptoms (early vs late blight necrosis)
    # Output: 28x28x256 -> 14x14x256
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    
    # Classifier Head: Global Average Pooling collapses 14x14 spatial grid to 1 vector of length 256
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout_regularization")(x)
    
    # Final Softmax Classification Layer (3 disease classes)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_potato_cnn")
    return model

def build_mobilenetv2_model(
    input_shape: Tuple[int, int, int] = IMAGE_SHAPE,
    num_classes: int = NUM_CLASSES,
    weights: Optional[str] = "imagenet"
) -> tf.keras.Model:
    """
    Builds a Transfer Learning model using MobileNetV2 pretrained on ImageNet.
    Ideal for lightweight, high-speed mobile and edge deployments.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # MobileNetV2 expects input in range [-1, 1], so transform from [0, 1]
    # x = (img * 2.0) - 1.0
    x = tf.keras.layers.Lambda(lambda img: (img * 2.0) - 1.0, name="mobilenet_scaling")(inputs)
    
    # Base MobileNetV2 backbone (without top classification head)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    # Freeze backbone weights for feature extraction
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="mobilenet_gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="mobilenet_dense")(x)
    x = tf.keras.layers.Dropout(0.3, name="mobilenet_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_potato_classifier")
    return model

if __name__ == "__main__":
    model = build_baseline_cnn()
    model.summary()
