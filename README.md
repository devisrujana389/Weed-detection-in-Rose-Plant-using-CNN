# Weed Detection in Rose Plant using CNN

A computer vision-based Weed Detection in Rose Plants project designed to identify and distinguish unwanted weeds from rose plants using image processing and machine learning techniques. The system helps improve plant monitoring, supports efficient weed management, and reduces manual effort in rose cultivation.

## Live Demo
🔗 https://rosarium-web.onrender.com

> The demo is hosted on Render's free plan, so the first load may take 30–60 seconds while the server wakes up.

## Project Overview
A web-based weed detection system that uses a Convolutional Neural Network (CNN) to identify weeds in rose plant images. The user uploads an image, the trained model classifies it as **Rose** or **Weed**, and the result is shown along with a confidence score.

The project is purely software-based. It does not use any hardware such as Arduino, Raspberry Pi, IoT sensors or robotics.

## Technologies Used

| Layer | Technology |
|-------|-----------|
| Deep learning | Python 3.11, TensorFlow / Keras, MobileNetV2 (transfer learning) |
| Model service | FastAPI, Uvicorn |
| Application backend | Java, Spring Boot 3.2.5, H2 database, Maven |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render |
| Tools | VS Code, IntelliJ IDEA, Postman, Git, GitHub |

## Features
- Upload plant images
- Detect weeds using the trained CNN model
- Display prediction results (class, confidence and weed probability)
- REST API that can be tested with Postman or curl

## System Architecture

```
Frontend (index.html)
        │  image upload
        ▼
Spring Boot backend  :8080  /api/predict
        │  multipart (key: file)
        ▼
FastAPI service      :5000  /predict
        │
        ▼
CNN model (.h5)  →  JSON { prediction, confidence, weed_probability }
```

## Project Structure

```
weed_detection_project/
├── src/              # Training and evaluation scripts
├── backend_api/      # Python FastAPI service (serves the CNN model)
├── java_backend/     # Java Spring Boot application
├── frontend/         # Web interface (index.html)
├── models/           # Trained model (.h5)
├── data/raw/
│   ├── rose/         # Rose plant images
│   └── weed/         # Weed images
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.11
- JDK 17 or 21 (Spring Boot 3.2.5 does not work with JDK 25/26)
- Maven
- Git

### Step 1: Clone the repository
```bash
git clone https://github.com/<your-username>/Weed-detection-in-Rose-Plant-using-CNN.git
cd Weed-detection-in-Rose-Plant-using-CNN
```

### Step 2: Create a virtual environment and install Python packages (Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Add the dataset (only needed if you want to train the model)
Place your images in these folders:
```
data/raw/rose/    -> rose plant images
data/raw/weed/    -> weed images
```
Use a balanced dataset (200+ images per class is recommended).

### Step 4: Train the model (optional)
```bash
python src/train.py mobilenet 20
python src/evaluate.py
```
The trained model is saved in the `models/` folder.

### Step 5: Start the Python model service (port 5000)
```bash
cd backend_api
uvicorn main:app --host 0.0.0.0 --port 5000
```

### Step 6: Start the Java backend (port 8080)
Open a new terminal:
```bash
cd java_backend
mvn spring-boot:run
```

### Step 7: Open the frontend
Open `frontend/index.html` in your browser, upload a rose plant image and view the prediction.

## API Usage

**Endpoint:** `POST http://localhost:8080/api/predict`
**Body:** `multipart/form-data` with key `file`

```bash
curl -X POST -F "file=@sample.jpg" http://localhost:8080/api/predict
```

**Sample response:**
```json
{
  "prediction": "Weed Detected",
  "confidence": 0.97,
  "weed_probability": 0.97
}
```

## Results

| Metric | Value |
|--------|-------|
| Training accuracy | _to be added_ |
| Validation accuracy | _to be added_ |
| Test accuracy | _to be added_ |

## Future
