"""
Evaluation Pipeline for Potato Leaf Disease Detection
Potato Leaf Disease Detection Project

Evaluates the trained model strictly on the held-out Test Dataset (216 images).
Computes:
- Overall Test Accuracy
- Per-class and Macro/Weighted Precision, Recall, and F1-score
- Confusion Matrix (numerical + visualization)
- Detailed Misclassification Analysis
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
    CLASS_NAMES,
    NUM_CLASSES,
    BATCH_SIZE,
)

# Redirect Matplotlib cache to writable workspace directory
mpl_cache_dir = PROCESSED_DATA_DIR / ".mplconfig"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import get_train_val_test_datasets

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Computes confusion matrix where C[i, j] is count of true class i predicted as j."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def compute_classification_metrics(cm: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """
    Computes Precision, Recall, F1-Score, and Support per class,
    along with Macro and Weighted averages.
    """
    num_classes = len(class_names)
    total_samples = int(np.sum(cm))
    total_correct = int(np.trace(cm))
    accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0

    per_class_metrics = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for i, cls in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        support = int(np.sum(cm[i, :]))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_metrics[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        weight = support / total_samples if total_samples > 0 else 0.0
        weighted_precision += precision * weight
        weighted_recall += recall * weight
        weighted_f1 += f1 * weight

    macro_precision /= num_classes
    macro_recall /= num_classes
    macro_f1 /= num_classes

    return {
        "accuracy": round(accuracy, 4),
        "total_test_samples": total_samples,
        "per_class": per_class_metrics,
        "macro_avg": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1_score": round(macro_f1, 4),
        },
        "weighted_avg": {
            "precision": round(weighted_precision, 4),
            "recall": round(weighted_recall, 4),
            "f1_score": round(weighted_f1, 4),
        },
    }

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Generates an annotated heatmap plot of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Custom color palette with imshow
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Sample Count", rotation=-90, va="bottom", fontsize=10)

    short_names = [name.replace("Potato___", "") for name in class_names]
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_names, fontsize=10, fontweight="bold")

    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Test Confusion Matrix (Held-out 216 Images)", fontsize=12, fontweight="bold", pad=12)

    # Annotate counts inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] Confusion matrix plot saved to: {save_path}")

def plot_misclassifications(
    misclassified_items: List[Dict[str, Any]],
    save_path: Path,
    max_images: int = 6
):
    """Plots sample misclassified images with True vs Predicted labels and confidences."""
    if not misclassified_items:
        print("[INFO] Zero misclassifications found! No plot needed.")
        return

    num_samples = min(len(misclassified_items), max_images)
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(num_samples):
        item = misclassified_items[idx]
        img_path = Path(item["image_path"])
        ax = axes[idx]

        if img_path.exists():
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Image Not Found", ha="center", va="center")

        true_lbl = item["true_class"].replace("Potato___", "")
        pred_lbl = item["predicted_class"].replace("Potato___", "")
        conf = item["confidence"] * 100

        ax.set_title(
            f"True: {true_lbl}\nPred: {pred_lbl} ({conf:.1f}%)",
            fontsize=10,
            fontweight="bold",
            color="#c0392b"
        )
        ax.axis("off")

    # Turn off any unused axes
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] Misclassified samples plot saved to: {save_path}")

def evaluate_model(model_path: Path = None) -> Dict[str, Any]:
    """Runs complete evaluation pipeline on the held-out test dataset."""
    if model_path is None:
        model_path = SAVED_MODELS_DIR / "baseline_best.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please train model first.")

    print("=" * 65)
    print(f"[START] EVALUATION ON TEST DATASET: {model_path.name}")
    print("=" * 65)

    # 1. Load trained model
    print(f"Loading model weights from: {model_path} ...")
    model = tf.keras.models.load_model(str(model_path))

    # 2. Load held-out test pipeline and manifest
    print("Loading test dataset (batch_size=32)...")
    _, _, test_ds = get_train_val_test_datasets(batch_size=BATCH_SIZE)

    manifest_path = PROCESSED_DATA_DIR / "split_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    test_manifest_items = [item for item in manifest if item["split"] == "test"]

    # 3. Collect true labels and predictions
    all_y_true = []
    all_y_prob = []

    print("Running inference across test batches...")
    for images, labels in test_ds:
        probs = model(images, training=False).numpy()
        all_y_prob.append(probs)
        all_y_true.append(labels.numpy())

    y_prob = np.concatenate(all_y_prob, axis=0)
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.argmax(y_prob, axis=-1)

    # 4. Compute metrics
    cm = compute_confusion_matrix(y_true, y_pred, NUM_CLASSES)
    metrics = compute_classification_metrics(cm, CLASS_NAMES)

    # 5. Identify misclassified samples
    misclassified_items = []
    for idx in range(len(y_true)):
        if y_true[idx] != y_pred[idx]:
            manifest_info = test_manifest_items[idx]
            full_img_path = str(PROJECT_ROOT / manifest_info["file_path"])
            misclassified_items.append({
                "sample_index": idx,
                "image_path": full_img_path,
                "relative_path": manifest_info["file_path"],
                "filename": manifest_info["filename"],
                "true_class": CLASS_NAMES[y_true[idx]],
                "true_index": int(y_true[idx]),
                "predicted_class": CLASS_NAMES[y_pred[idx]],
                "predicted_index": int(y_pred[idx]),
                "confidence": float(y_prob[idx, y_pred[idx]]),
                "probabilities": {
                    CLASS_NAMES[c]: float(y_prob[idx, c]) for c in range(NUM_CLASSES)
                }
            })

    metrics["misclassified_count"] = len(misclassified_items)
    metrics["misclassified_percentage"] = round((len(misclassified_items) / len(y_true)) * 100, 2)

    # 6. Save Artifacts
    metrics_path = PROCESSED_DATA_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Evaluation metrics saved to: {metrics_path}")

    misclass_path = PROCESSED_DATA_DIR / "misclassifications.json"
    with open(misclass_path, "w", encoding="utf-8") as f:
        json.dump(misclassified_items, f, indent=2)
    print(f"[OK] Misclassified items log saved to: {misclass_path}")

    cm_plot_path = PROCESSED_DATA_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_plot_path)

    misclass_plot_path = PROCESSED_DATA_DIR / "misclassified_samples.png"
    plot_misclassifications(misclassified_items, misclass_plot_path)

    # 7. Print Console Summary Table
    print("\n" + "=" * 65)
    print(f"TEST ACCURACY: {metrics['accuracy'] * 100:.2f}% ({np.trace(cm)} / {len(y_true)} samples)")
    print("=" * 65)
    print(f"{'Class Name':<28} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Support'}")
    print("-" * 65)
    for cls, val in metrics["per_class"].items():
        short_cls = cls.replace("Potato___", "")
        print(f"{short_cls:<28} | {val['precision']:<9.4f} | {val['recall']:<9.4f} | {val['f1_score']:<9.4f} | {val['support']}")
    print("-" * 65)
    print(f"{'Macro Average':<28} | {metrics['macro_avg']['precision']:<9.4f} | {metrics['macro_avg']['recall']:<9.4f} | {metrics['macro_avg']['f1_score']:<9.4f} | {len(y_true)}")
    print(f"{'Weighted Average':<28} | {metrics['weighted_avg']['precision']:<9.4f} | {metrics['weighted_avg']['recall']:<9.4f} | {metrics['weighted_avg']['f1_score']:<9.4f} | {len(y_true)}")
    print("=" * 65)
    print(f"Total Misclassified: {metrics['misclassified_count']} / {len(y_true)} ({metrics['misclassified_percentage']}%)")
    print("=" * 65)

    return metrics

if __name__ == "__main__":
    evaluate_model()
