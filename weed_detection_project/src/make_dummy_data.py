"""
Generates small placeholder images so you can verify the ENTIRE pipeline
runs without errors before you plug in your real rose/weed dataset.
NOT a substitute for a real dataset.

Usage:
    python src/make_dummy_data.py
"""

import os
import numpy as np
from PIL import Image

OUT_DIR = "data/raw"
N_PER_CLASS = 60
IMG_SIZE = (224, 224)


def make_class_images(class_name: str, color_bias: tuple, n: int):
    out_path = os.path.join(OUT_DIR, class_name)
    os.makedirs(out_path, exist_ok=True)
    rng = np.random.default_rng(42)

    for i in range(n):
        noise = rng.integers(0, 80, size=(*IMG_SIZE, 3), dtype=np.uint8)
        base = np.array(color_bias, dtype=np.uint8)
        img_arr = np.clip(noise + base, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="RGB")
        img.save(os.path.join(out_path, f"{class_name}_{i:03d}.jpg"))

    print(f"Created {n} dummy images in {out_path}")


if __name__ == "__main__":
    make_class_images("rose", color_bias=(120, 60, 90), n=N_PER_CLASS)
    make_class_images("weed", color_bias=(60, 120, 60), n=N_PER_CLASS)
    print("\nDummy dataset ready. You can now run:")
    print("  python src/train.py --data_dir data/raw --model_type custom --epochs 3")
