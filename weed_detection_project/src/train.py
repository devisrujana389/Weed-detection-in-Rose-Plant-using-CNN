"""
Train the weed detection CNN.

Usage:
    python src/train.py --data_dir data/raw --model_type mobilenet --epochs 20
"""

import argparse
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from dataset import get_data_generators
from model import get_model


def plot_history(history, out_path="outputs/training_curves.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train acc")
    axes[0].plot(history.history["val_accuracy"], label="val acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train loss")
    axes[1].plot(history.history["val_loss"], label="val loss")
    axes[1].set_title("Loss (Binary Cross-Entropy)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--model_type", type=str, default="mobilenet",
                         choices=["custom", "mobilenet", "vgg16", "resnet50"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_model", type=str, default="models/weed_detector.h5")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)

    print(f"Loading data from: {args.data_dir}")
    train_gen, val_gen = get_data_generators(args.data_dir, batch_size=args.batch_size)
    print("Class indices:", train_gen.class_indices)
    print(f"Train samples: {train_gen.samples} | Val samples: {val_gen.samples}")

    if train_gen.samples == 0:
        raise RuntimeError(
            "No training images found. Make sure data_dir contains subfolders "
            "like 'rose/' and 'weed/' with images inside them."
        )

    print(f"Building model: {args.model_type}")
    model = get_model(args.model_type)
    model.summary()

    callbacks = [
        ModelCheckpoint(args.out_model, save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss"),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history)
    print(f"\nTraining complete. Best model saved to: {args.out_model}")


if __name__ == "__main__":
    main()
