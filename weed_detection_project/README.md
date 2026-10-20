# Weed Detection in Rose Plants using CNN

A complete, ready-to-run implementation of the CNN-based weed detection system.
Classifies plant images as **"Weed Detected"** or **"No Weed Detected"** using
a Convolutional Neural Network, with support for both a custom CNN and
transfer learning (MobileNetV2 / VGG16 / ResNet50).

---

## Project structure

```
weed_detection_project/
│
├── data/
│   ├── raw/
│   │   ├── rose/          <- put rose plant images here
│   │   └── weed/          <- put weed images here
│   └── sample_test/
│
├── src/
│   ├── preprocess.py       # resizing, normalization, noise removal helpers
│   ├── dataset.py          # data loading + augmentation pipeline
│   ├── model.py             # custom CNN + transfer-learning model builders
│   ├── train.py             # trains the model end-to-end
│   ├── evaluate.py          # accuracy / precision / recall / F1 + confusion matrix
│   ├── predict.py           # run inference on a single image
│   └── make_dummy_data.py   # generates fake images to smoke-test the pipeline
│
├── models/                  # trained .h5 model gets saved here
├── outputs/                 # training curves, confusion matrix images
├── backend_api/              # Python REST API that serves the trained model
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
├── app.py                   # Streamlit web demo
├── requirements.txt
└── README.md                 (this file)
```

## Step 0 — Install requirements

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1 — Get a dataset

You need two folders of images:
- `data/raw/rose/`  → images of rose plants/leaves (no weeds)
- `data/raw/weed/`  → images of weeds

**Quick test option (no real photos needed):**
```bash
python src/make_dummy_data.py
```
Generates placeholder images so you can confirm the whole pipeline runs
end-to-end before you plug in real photos.

**Real data:** download from Kaggle (rose-leaf-disease-dataset for rose,
v2-plant-seedlings-dataset for weeds — pick "Healthy" folder for rose,
weed-species folders for weed) and drop images into the folders above.

## Step 2 — Train

```bash
python src/train.py --data_dir data/raw --model_type mobilenet --epochs 20
```

Options for `--model_type`: `custom`, `mobilenet`, `vgg16`, `resnet50`.

## Step 3 — Evaluate

```bash
python src/evaluate.py --data_dir data/raw --model_path models/weed_detector.h5
```

## Step 4 — Predict on a single image

```bash
python src/predict.py --image path/to/your/photo.jpg --model_path models/weed_detector.h5
```

## Step 5 — Run the interactive demo app

```bash
streamlit run app.py
```

## Step 6 — Run the backend REST API (for Java to call)

```bash
cd backend_api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```
Then test at `http://localhost:5000/docs`. See `backend_api/README.md` for
full details.

## Notes for your report / viva

- The pipeline implements every step of a standard CNN methodology:
  preprocessing → augmentation → CNN feature extraction → fully connected
  layers → sigmoid output → Adam optimizer → Binary Cross-Entropy loss →
  accuracy/precision/recall/F1 evaluation.
- After training, copy the real numbers (accuracy, precision, recall, F1,
  confusion matrix and training curve images from `outputs/`) into your
  report.
- If your dataset is small, keep `--model_type mobilenet` (transfer learning)
  — it needs far less data than a from-scratch CNN to get good accuracy.
