#!/usr/bin/env python3
"""
train.py — Training script for Bangla OCR (BanglaLekha-Isolated dataset).

Supports two model architectures:
  1. Custom CNN (4-block ConvNet with BatchNorm)
  2. MobileNetV2 (ImageNet transfer learning with fine-tuning)

Features:
  - GPU auto-detection with memory growth
  - Data augmentation (rotation, translation, zoom)
  - MLflow experiment tracking (params, metrics per epoch, artifacts)
  - Early stopping, LR reduction, model checkpointing
  - Post-training evaluation: confusion matrix + classification report

Usage examples:
  python train.py --model cnn --epochs 30 --batch-size 64
  python train.py --model mobilenet --epochs 20 --learning-rate 0.0001
  python train.py --model both --subset 100
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

try:
    import mlflow
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARNING] mlflow not installed. Experiment tracking will be disabled.")


# Bangla character mapping (84 classes)
# Vowels (indices 0–10)
# Consonants (indices 11–49)
# Numerals (indices 50–59)
# Compound characters (indices 60–83) — placeholders
# Explicit folder-name → character mapping for BanglaLekha-Isolated.
# Folder names are the dataset's numeric class IDs (1-indexed strings).
# Using an explicit dict means we never depend on filesystem sort order.
FOLDER_TO_CHAR = {
    # Vowels
    "1":  "অ", "2":  "আ", "3":  "ই", "4":  "ঈ", "5":  "উ",
    "6":  "ঊ", "7":  "ঋ", "8":  "এ", "9":  "ঐ", "10": "ও",
    "11": "ঔ",
    # Consonants
    "12": "ক", "13": "খ", "14": "গ", "15": "ঘ", "16": "ঙ",
    "17": "চ", "18": "ছ", "19": "জ", "20": "ঝ", "21": "ঞ",
    "22": "ট", "23": "ঠ", "24": "ড", "25": "ঢ", "26": "ণ",
    "27": "ত", "28": "থ", "29": "দ", "30": "ধ", "31": "ন",
    "32": "প", "33": "ফ", "34": "ব", "35": "ভ", "36": "ম",
    "37": "য", "38": "র", "39": "ল",
    "40": "শ", "41": "ষ", "42": "স", "43": "হ",
    "44": "ড়", "45": "ঢ়", "46": "য়", "47": "ৎ",
    "48": "ং", "49": "ঃ", "50": "ঁ",
    # Numerals
    "51": "০", "52": "১", "53": "২", "54": "৩", "55": "৪",
    "56": "৫", "57": "৬", "58": "৭", "59": "৮", "60": "৯",
    # Compound characters (61-84) — not yet labelled in this dataset
    "61": "compound_61", "62": "compound_62", "63": "compound_63",
    "64": "compound_64", "65": "compound_65", "66": "compound_66",
    "67": "compound_67", "68": "compound_68", "69": "compound_69",
    "70": "compound_70", "71": "compound_71", "72": "compound_72",
    "73": "compound_73", "74": "compound_74", "75": "compound_75",
    "76": "compound_76", "77": "compound_77", "78": "compound_78",
    "79": "compound_79", "80": "compound_80", "81": "compound_81",
    "82": "compound_82", "83": "compound_83", "84": "compound_84",
}

IMAGE_SIZE = (64, 64)
NUM_CLASSES = 84


# CLI argument parsing
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train Bangla OCR models (CNN / MobileNetV2) on BanglaLekha-Isolated.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "mobilenet", "both"],
        default="both",
        help="Which model architecture(s) to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="If provided, use only N batches from each dataset (for quick experiments).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/BanglaLekha-Isolated/Images",
        help="Path to the BanglaLekha-Isolated image directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save trained models and artifacts.",
    )
    return parser.parse_args()


# GPU setup
def setup_gpu() -> None:
    """Detect and configure GPU with memory growth to avoid OOM errors."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" Using GPU: {gpus[0].name}")
    else:
        print(" No GPU found, using CPU")


# Data loading
def load_data(args: argparse.Namespace):
    """
    Load images from the data directory using tf.keras.utils.image_dataset_from_directory.

    Returns:
        train_ds:    tf.data.Dataset for training (augmented, normalized)
        val_ds:      tf.data.Dataset for validation (normalized only)
        class_names: list of class folder names as returned by Keras
    """
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f" Data directory not found: {data_dir}")
        print("   Please download the BanglaLekha-Isolated dataset first.")
        sys.exit(1)

    print(f"\nLoading data from: {data_dir}")
    print(f"   Image size : {IMAGE_SIZE}")
    print(f"   Batch size : {args.batch_size}")

    #Training split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMAGE_SIZE,
        color_mode="grayscale",
        label_mode="int",
        batch_size=args.batch_size,
        validation_split=0.2,
        subset="training",
        seed=42,
    )

    # Validation split
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMAGE_SIZE,
        color_mode="grayscale",
        label_mode="int",
        batch_size=args.batch_size,
        validation_split=0.2,
        subset="validation",
        seed=42,
    )

    class_names = train_ds.class_names
    print(f"   Classes     : {len(class_names)}")

    # Flag to imit the dataset size for quick experiments
    if args.subset is not None:
        print(f"   Using only {args.subset} batches per split (--subset)")
        train_ds = train_ds.take(args.subset)
        val_ds = val_ds.take(args.subset)

    #Normalisation (0-255 → 0-1)
    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)

    #Data augmentation (training only)
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    # Apply augmentation + normalisation to training set
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(normalization(x), training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Apply normalisation only to validation set
    val_ds = val_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


# Labels
def save_labels(class_names: list, output_dir: str) -> str:
    """
    Generate and save labels.json mapping model class indices to Bangla characters.

    Uses FOLDER_TO_CHAR — an explicit folder-name → character dict — so the
    mapping is never sensitive to filesystem sort order or list position.

    image_dataset_from_directory assigns class index `idx` to whichever folder
    it happens to enumerate at position `idx` (alphabetical order). We look up
    the character by the folder's *name*, not its position.

    Saves two copies:
      1. {output_dir}/labels.json — alongside the model
      2. ./labels.json            — project root (for app.py / Docker)

    Returns:
        Path to the project-root labels.json file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Map model class index → character via folder name (order-independent)
    labels_map = {}
    for idx, folder_name in enumerate(class_names):
        labels_map[str(idx)] = FOLDER_TO_CHAR.get(folder_name, f"unknown_{folder_name}")

    # Save to models directory
    models_labels_path = os.path.join(output_dir, "labels.json")
    with open(models_labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_map, f, ensure_ascii=False, indent=2)

    # Also save to project root (app.py and Docker expect it here)
    root_labels_path = "labels.json"
    with open(root_labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_map, f, ensure_ascii=False, indent=2)

    print(f"\nSaved labels.json → {models_labels_path}")
    print(f"   Also saved to project root → {root_labels_path}")
    return root_labels_path


# Model builders
def build_cnn(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Build a custom 4-block CNN with BatchNorm and GlobalAveragePooling.

    Architecture:
        Conv2D(32) → BN → Pool
        Conv2D(64) → BN → Pool
        Conv2D(128) → BN → Pool
        Conv2D(256) → BN → Pool
        GAP → Dense(256) → Dropout → Dense(num_classes, softmax)
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(64, 64, 1)),
            # Block 1
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            # Block 2
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            # Block 3
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            # Block 4
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            # Head
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="bangla_cnn",
    )
    return model


def build_mobilenet(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Build a MobileNetV2-based model for transfer learning.

    A Lambda layer converts single-channel grayscale input to 3-channel RGB
    so we can leverage ImageNet-pretrained weights.

    Architecture:
        Input(64,64,1) → grayscale_to_rgb → MobileNetV2(frozen) → GAP
        → Dense(256) → Dropout → Dense(num_classes, softmax)
    """
    inputs = tf.keras.layers.Input(shape=(64, 64, 1), name="grayscale_input")

    # Convert grayscale → RGB (3-channel) for MobileNetV2
    x = tf.keras.layers.Lambda(
        lambda img: tf.image.grayscale_to_rgb(img),
        name="grayscale_to_rgb",
    )(inputs)

    # MobileNetV2 backbone (frozen initially)
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(64, 64, 3),
    )
    base_model.trainable = False  # Freeze during initial training

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bangla_mobilenet")

    # Store reference to the base model so we can unfreeze later
    model._base_model = base_model

    return model


# MLflow callback
class MLflowCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback that logs per-epoch metrics to MLflow.

    Logs:
        - train_loss, train_accuracy   (from training metrics)
        - val_loss, val_accuracy       (from validation metrics)
    """

    def on_epoch_end(self, epoch: int, logs: dict = None):
        if not MLFLOW_AVAILABLE or logs is None:
            return
        mlflow.log_metric("train_loss", logs.get("loss", 0), step=epoch)
        mlflow.log_metric("train_accuracy", logs.get("accuracy", 0), step=epoch)
        mlflow.log_metric("val_loss", logs.get("val_loss", 0), step=epoch)
        mlflow.log_metric("val_accuracy", logs.get("val_accuracy", 0), step=epoch)


# Callbacks builder
def get_callbacks(model_type: str, output_dir: str) -> list:
    """
    Create training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, MLflow.

    Args:
        model_type: 'cnn' or 'mobilenet'
        output_dir: directory to save model checkpoints

    Returns:
        List of Keras callback instances.
    """
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, f"bangla_{model_type}_best.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        MLflowCallback(),
    ]
    return callbacks


# Evaluation helpers
def evaluate_model(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    class_names: list,
    model_type: str,
    output_dir: str,
) -> dict:
    """
    Run post-training evaluation on the validation set.

    Generates:
        1. Confusion matrix heatmap (saved as PNG)
        2. Classification report (saved as text)

    Returns:
        Dictionary with paths to generated artifacts.
    """
    print(f"\nEvaluating {model_type} model on validation set...")

    # Collect all predictions and true labels
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,  # Too many classes for annotations
        fmt="d",
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title(f"Confusion Matrix — {model_type.upper()}", fontsize=16)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_type}.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"   Saved confusion matrix → {cm_path}")

    # Classification Report
    # Build label names for the report (use Bangla chars if available)
    bangla_chars = [FOLDER_TO_CHAR[k] for k in sorted(FOLDER_TO_CHAR, key=lambda x: int(x))]
    num_unique = len(set(y_true) | set(y_pred))
    target_names = None
    if num_unique <= len(bangla_chars):
        unique_labels = sorted(set(y_true) | set(y_pred))
        target_names = [
            bangla_chars[i] if i < len(bangla_chars) else f"class_{i}"
            for i in unique_labels
        ]

    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    report_path = os.path.join(output_dir, f"classification_report_{model_type}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"   Saved classification report → {report_path}")

    # Compute overall accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"   Validation accuracy: {accuracy:.4f}")

    return {
        "confusion_matrix_path": cm_path,
        "classification_report_path": report_path,
        "accuracy": accuracy,
    }


# Training orchestration
def train_and_evaluate(
    args: argparse.Namespace,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: list,
    model_type: str,
) -> None:
    """
    Full training + evaluation pipeline for a single model type.

    Steps:
        1. Build the model (CNN or MobileNetV2)
        2. Compile and train (with optional fine-tuning for MobileNetV2)
        3. Evaluate and generate artifacts
        4. Log everything to MLflow

    Args:
        args:        Parsed CLI arguments
        train_ds:    Preprocessed training dataset
        val_ds:      Preprocessed validation dataset
        class_names: List of class folder names
        model_type:  'cnn' or 'mobilenet'
    """
    print("\n" + "=" * 70)
    print(f"Training model: {model_type.upper()}")
    print("=" * 70)

    # ---- Build model ----
    if model_type == "cnn":
        model = build_cnn(num_classes=len(class_names))
        learning_rate = args.learning_rate
    elif model_type == "mobilenet":
        model = build_mobilenet(num_classes=len(class_names))
        learning_rate = 0.0001  # Lower LR for transfer learning
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    #Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary()

    #Output paths
    model_output_dir = os.path.join(args.output_dir, model_type)
    os.makedirs(model_output_dir, exist_ok=True)

    labels_path = os.path.join(args.output_dir, "labels.json")

    #MLflow setup
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        os.makedirs("./artifacts", exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
        mlflow.set_experiment("bangla-ocr-training")
        mlflow_run = mlflow.start_run(run_name=f"{model_type}-training")

        mlflow.set_tags(
            {
                "dataset": "BanglaLekha-Isolated",
                "framework": "tensorflow",
                "model_type": model_type,
            }
        )
        mlflow.log_params(
            {
                "model_type": model_type,
                "learning_rate": learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
                "num_classes": len(class_names),
                "optimizer": "Adam",
            }
        )

    # Callbacks
    callbacks = get_callbacks(model_type, model_output_dir)

    # Train (phase 1)
    print(f"\n Phase 1: Training for {args.epochs} epochs (lr={learning_rate})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Fine-tuning for MobileNetV2 (phase 2)
    if model_type == "mobilenet":
        print("\n Phase 2: Fine-tuning MobileNetV2 (unfreezing last 20 layers)...")
        base_model = model._base_model

        # Unfreeze the base model
        base_model.trainable = True

        # Freeze all layers except the last 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Re-compile with a much lower learning rate
        fine_tune_lr = 0.00001
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        if MLFLOW_AVAILABLE:
            mlflow.log_param("fine_tune_lr", fine_tune_lr)
            mlflow.log_param("fine_tune_epochs", 10)
            mlflow.log_param("fine_tune_unfrozen_layers", 20)

        # Update callbacks for fine-tuning (new checkpoint path)
        ft_callbacks = get_callbacks(f"{model_type}_finetuned", model_output_dir)

        fine_tune_epochs = 10
        total_epochs = args.epochs + fine_tune_epochs

        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            initial_epoch=len(history.history["loss"]),
            callbacks=ft_callbacks,
            verbose=1,
        )

    # Save final model
    model_save_path = os.path.join(model_output_dir, f"bangla_{model_type}_final.keras")
    model.save(model_save_path)
    print(f"\n Saved model → {model_save_path}")

    # Also save to the default app.py location
    default_model_path = os.path.join(args.output_dir, "model.keras")
    model.save(default_model_path)
    print(f"   Also saved to → {default_model_path} (used by app.py)")

    # Evaluate
    eval_results = evaluate_model(
        model, val_ds, class_names, model_type, model_output_dir
    )

    # Log artifacts to MLflow
    if MLFLOW_AVAILABLE and mlflow_run is not None:
        try:
            mlflow.log_artifact(model_save_path)
            mlflow.log_artifact(labels_path)
            mlflow.log_artifact(eval_results["confusion_matrix_path"])
            mlflow.log_artifact(eval_results["classification_report_path"])
            mlflow.log_metric("final_val_accuracy", eval_results["accuracy"])
        except Exception as e:
            print(f"[WARNING] Failed to log some MLflow artifacts: {e}")
        finally:
            mlflow.end_run()

    # Print summary
    print(f" {model_type.upper()} Training Complete!")
    print(f"   Final Validation Accuracy : {eval_results['accuracy']:.4f}")
    print(f"   Model saved to            : {model_save_path}")
    print(f"   Confusion matrix          : {eval_results['confusion_matrix_path']}")
    print(f"   Classification report     : {eval_results['classification_report_path']}")


# Main entry point
def main() -> None:
    """Main entry point: parse args, setup GPU, load data, train models."""
    args = parse_args()

    print(" Bangla OCR Training Pipeline")
    print(f"  Model(s)      : {args.model}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Learning rate : {args.learning_rate}")
    print(f"  Data dir      : {args.data_dir}")
    print(f"  Output dir    : {args.output_dir}")
    if args.subset:
        print(f"  Subset        : {args.subset} batches")

    # 1. GPU configuration
    setup_gpu()

    # 2. Load and preprocess data
    train_ds, val_ds, class_names = load_data(args)

    # 3. Save labels mapping
    save_labels(class_names, args.output_dir)

    # 4. Train selected model(s)
    if args.model in ("cnn", "both"):
        train_and_evaluate(args, train_ds, val_ds, class_names, model_type="cnn")

    if args.model in ("mobilenet", "both"):
        train_and_evaluate(args, train_ds, val_ds, class_names, model_type="mobilenet")

    print("\nAll training runs completed!!!")


if __name__ == "__main__":
    main()