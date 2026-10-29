"""
Training Pipeline for Potato Leaf Disease Detection Models
Potato Leaf Disease Detection Project
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import (
    SAVED_MODELS_DIR,
    PROCESSED_DATA_DIR,
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    BATCH_SIZE,
)

# Redirect Matplotlib cache to writable workspace directory
mpl_cache_dir = PROCESSED_DATA_DIR / ".mplconfig"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import get_train_val_test_datasets
from src.model import build_baseline_cnn, build_mobilenetv2_model

def train(
    model_type: str = "baseline",
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    use_class_weights: bool = False
):
    print("=" * 65)
    print(f"[START] INITIATING MODEL TRAINING: [{model_type.upper()}] (class_weights={use_class_weights})")
    print("=" * 65)
    
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data Pipelines (Train and Validation ONLY)
    print("Loading data pipelines (Test set is strictly held out)...")
    train_ds, val_ds, _ = get_train_val_test_datasets(batch_size=BATCH_SIZE)

    class_weights_dict = None
    if use_class_weights:
        manifest_path = PROCESSED_DATA_DIR / "split_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        train_items = [item for item in manifest if item["split"] == "train"]
        class_counts = {}
        for item in train_items:
            cls_name = item["class_name"]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        from src.dataset import CLASS_TO_INDEX
        total_train = len(train_items)
        n_classes = len(class_counts)
        class_weights_dict = {
            CLASS_TO_INDEX[c]: round(total_train / (n_classes * count), 4)
            for c, count in class_counts.items()
        }
        print(f"Computed Balanced Class Weights: {class_weights_dict}")
    
    # 2. Build Selected Architecture
    if model_type.lower() == "baseline":
        model = build_baseline_cnn()
    elif model_type.lower() == "mobilenetv2":
        model = build_mobilenetv2_model()
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'baseline' or 'mobilenetv2'.")

    # 3. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )
    print(f"\nModel compiled with Adam (lr={lr}) and SparseCategoricalCrossentropy.\n")

    # 4. Configure Callbacks
    best_model_path = SAVED_MODELS_DIR / f"{model_type}_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. Execute Training Loop
    print(f"Training for up to {epochs} epochs with EarlyStopping (patience={EARLY_STOPPING_PATIENCE})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    # 6. Save Final Model
    final_model_path = SAVED_MODELS_DIR / f"{model_type}_final.keras"
    model.save(str(final_model_path))
    print(f"\n[OK] Best model saved to:  {best_model_path}")
    print(f"[OK] Final model saved to: {final_model_path}")

    # 7. Persist Training History JSON
    raw_lr = history.history.get("learning_rate", history.history.get("lr", []))
    if isinstance(raw_lr, list) and len(raw_lr) > 0:
        lr_series = [float(x) for x in raw_lr]
    else:
        lr_series = [float(lr)] * len(history.history["loss"])

    history_dict = {
        "epoch": list(range(1, len(history.history["loss"]) + 1)),
        "loss": [float(x) for x in history.history["loss"]],
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "lr": lr_series
    }
    history_path = PROCESSED_DATA_DIR / f"training_history_{model_type}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[OK] Training history saved to: {history_path}")

    # 8. Plot and Save Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = history_dict["epoch"]

    # Accuracy subplot
    ax1.plot(epochs_range, history_dict["accuracy"], "o-", label="Train Accuracy", color="#2ecc71")
    ax1.plot(epochs_range, history_dict["val_accuracy"], "s--", label="Val Accuracy", color="#3498db")
    ax1.set_title(f"{model_type.capitalize()} CNN: Accuracy Curve", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Loss subplot
    ax2.plot(epochs_range, history_dict["loss"], "o-", label="Train Loss", color="#e74c3c")
    ax2.plot(epochs_range, history_dict["val_loss"], "s--", label="Val Loss", color="#e67e22")
    ax2.set_title(f"{model_type.capitalize()} CNN: Loss Curve", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    curves_path = PROCESSED_DATA_DIR / f"training_curves_{model_type}.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"[OK] Training curves plot saved to: {curves_path}")
    print("=" * 65)

    return model, history_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Potato Disease Classification Model")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "mobilenetv2"], help="Model type to train")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--use-class-weights", action="store_true", help="Apply inverse frequency class weights")
    args = parser.parse_args()

    train(model_type=args.model, epochs=args.epochs, lr=args.lr, use_class_weights=args.use_class_weights)
