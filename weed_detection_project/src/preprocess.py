"""
Image preprocessing utilities: resizing, denoising, contrast enhancement,
and pixel normalization before images are fed to the CNN.
"""

import cv2
import numpy as np

IMG_SIZE = (224, 224)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_image(img: np.ndarray, size=IMG_SIZE) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def denoise_image(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype("float32")
    x_min, x_max = img.min(), img.max()
    if x_max - x_min == 0:
        return img / 255.0
    return (img - x_min) / (x_max - x_min)


def preprocess_image(path: str, denoise: bool = True, contrast: bool = True) -> np.ndarray:
    img = load_image(path)
    img = resize_image(img)
    if denoise:
        img = denoise_image(img)
    if contrast:
        img = enhance_contrast(img)
    img = normalize_image(img)
    return img


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        out = preprocess_image(sys.argv[1])
        print("Processed image shape:", out.shape, "| value range:", out.min(), "-", out.max())
    else:
        print("Usage: python preprocess.py <image_path>")
