"""
Tests the running Python API on many images at once.

Usage (API must be running on port 5000):
    python test_api_folder.py "PATH_TO_ROSE_FOLDER" "PATH_TO_WEED_FOLDER"

Example:
    python test_api_folder.py data/raw/rose data/raw/weed
"""
import os
import random
import sys

import requests

URL = "http://localhost:5000/predict"
SAMPLES = 30  # images tested per folder
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def test_folder(folder, true_is_weed):
    files = [f for f in os.listdir(folder) if f.lower().endswith(EXT)]
    if not files:
        print(f"No images found in {folder}")
        return []
    random.seed(1)
    files = random.sample(files, min(SAMPLES, len(files)))
    probs, correct = [], 0
    for name in files:
        path = os.path.join(folder, name)
        with open(path, "rb") as f:
            r = requests.post(URL, files={"file": (name, f, "image/jpeg")})
        if r.status_code != 200:
            print("  error on", name, r.text)
            continue
        p = r.json()["weed_probability"]
        probs.append(p)
        if (p >= 0.5) == true_is_weed:
            correct += 1
    label = "WEED" if true_is_weed else "ROSE"
    print(f"\n{label} folder: {len(probs)} images tested")
    print(f"  correct        : {correct}/{len(probs)}")
    if probs:
        print(f"  weed_prob min  : {min(probs):.3f}")
        print(f"  weed_prob avg  : {sum(probs)/len(probs):.3f}")
        print(f"  weed_prob max  : {max(probs):.3f}")
    return probs


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    rose_probs = test_folder(sys.argv[1], true_is_weed=False)
    weed_probs = test_folder(sys.argv[2], true_is_weed=True)
    print("\nDone. Send me this whole output.")