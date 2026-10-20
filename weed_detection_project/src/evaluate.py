"""
Evaluate a trained model: accuracy, precision, recall, F1, confusion matrix.

Usage:
    python src/evaluate.py --data_dir data/raw --model_path models/weed_detector.h5
"""

import argparse
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

from dataset import get_data_generators


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--model_path", type=str, default="models/weed_detector.h5")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)

    print(f"Loading validation data from {args.data_dir}")
    _, val_gen = get_data_generators(args.data_dir)
    class_names = {v: k for k, v in val_gen.class_indices.items()}
    print("Class indices:", val_gen.class_indices)

    val_gen.reset()
    y_true = val_gen.classes
    y_prob = model.predict(val_gen, verbose=1).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n===== Evaluation Metrics =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=[class_names[0], class_names[1]]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=[class_names[0], class_names[1]],
                yticklabels=[class_names[0], class_names[1]])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    out_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\nSaved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
