# 👗 Fashion Item Classifier

A deep learning application that classifies clothing images into 10 fashion categories using a Convolutional Neural Network (CNN) trained on the Fashion MNIST dataset. Built with a decoupled architecture — a **FastAPI** backend serving the model via REST API, and a **Streamlit** frontend with a custom dark theme consuming it.

🔗 **Live Demo:** https://fashion-classification-using-cnn-dl-project-app.streamlit.app/
🔗 **API Docs:** https://fashion-classification-using-cnn-dl-project-production.up.railway.app/docs

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Limitations & Future Improvements](#limitations--future-improvements)

---

## 🔍 Overview

This project implements an end-to-end image classification system that identifies clothing items — T-shirts, trousers, dresses, shoes, bags, and more — from a single image. Like the project's companion Cat vs Dog Classifier, it separates the model-serving backend from the presentation layer, allowing the same API to power multiple frontends or client applications.

The Streamlit frontend supports both direct image upload and image URL input, returning the predicted category along with a confidence score and the top-3 most likely classes.

---

## 🏗 Architecture

```
┌────────────────────┐        HTTP/REST        ┌──────────────────────┐
│  Streamlit Frontend │  ───────────────────►   │   FastAPI Backend     │
│  (Streamlit Cloud)  │  ◄───────────────────   │   (Railway)           │
└────────────────────┘        JSON Response      └──────────────────────┘
                                                          │
                                                          ▼
                                                  ┌─────────────────┐
                                                  │  CNN Model (.h5) │
                                                  └─────────────────┘
```

The frontend sends the image to the backend, which handles preprocessing (grayscale conversion, resizing, normalization), runs inference, and returns a structured JSON response.

---

## ✨ Features

- Image classification via **file upload** or **image URL**
- Returns **top-3 predictions** with confidence scores, not just the top class
- REST API with interactive **Swagger documentation** (`/docs`)
- Backend health check endpoint for monitoring
- Input validation — file type, file size, and corrupt/invalid image handling
- Automatic **prediction logging** to CSV for basic analytics
- Custom dark-themed, professional frontend UI
- CORS-enabled API, allowing any frontend to consume it
- Independently deployable backend and frontend services

---

## 🛠 Tech Stack

| Category            | Tools/Libraries                          |
|---------------------|-------------------------------------------|
| Language            | Python                                    |
| Deep Learning       | TensorFlow / Keras                        |
| Backend API         | FastAPI, Uvicorn                          |
| Frontend            | Streamlit                                 |
| Image Processing    | Pillow, NumPy                             |
| Development         | Jupyter Notebook / Google Colab           |
| Deployment          | Railway (backend), Streamlit Cloud (frontend) |

---

## 📂 Project Structure

```
Fashion-Classification-Using-CNN-DL-Project/
│
├── backend/
│   ├── main.py                  # FastAPI application (routes, inference, logging)
│   ├── requirements.txt         # Backend dependencies (FastAPI + TensorFlow)
│   ├── model/
│   │   └── fashion_model.h5     # Trained CNN model
│   └── data/
│       ├── class_labels.json    # Class index → label mapping
│       └── prediction_logs.csv  # Auto-generated prediction history
│
├── frontend/
│   ├── app.py                   # Streamlit UI (dark theme)
│   └── requirements.txt         # Frontend dependencies (lightweight, no TensorFlow)
│
├── notebook/
│   └── fashion_mnist_classification.ipynb   # Full data prep, training & evaluation notebook
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset

Trained on the **Fashion MNIST** dataset (via `keras.datasets`), consisting of:

| Split       | Images  |
|-------------|---------|
| Training    | 60,000  |
| Test        | 10,000  |

Each image is a `28x28` grayscale image belonging to one of 10 categories:

| Label | Category      | Label | Category    |
|-------|---------------|-------|-------------|
| 0     | T-shirt/top   | 5     | Sandal      |
| 1     | Trouser       | 6     | Shirt       |
| 2     | Pullover      | 7     | Sneaker     |
| 3     | Dress         | 8     | Bag         |
| 4     | Coat          | 9     | Ankle boot  |

---

## ⚙️ Methodology

1. **Data Loading** — Loaded the Fashion MNIST dataset directly via Keras' built-in dataset loader.
2. **Exploratory Data Analysis** — Visualized sample images and inspected pixel-level data and class distributions.
3. **Preprocessing** — Reshaped images to include a single grayscale channel `(28, 28, 1)`, normalized pixel values to `[0, 1]`, and cast to `float32`.
4. **Model Architecture** — Built a custom 3-block CNN:
   - Each block: `Conv2D → MaxPooling2D → BatchNormalization → Dropout`
   - Followed by a `Flatten` layer and a `Dense(10, softmax)` output layer for multi-class classification.
5. **Training** — Trained for 5 epochs using the Adam optimizer and sparse categorical cross-entropy loss, with an 80/20 train-validation split.
6. **Evaluation** — Assessed performance on the held-out 10,000-image test set.
7. **Serialization** — Saved the trained model in HDF5 (`.h5`) format.
8. **API Development** — Wrapped the model in a FastAPI service with endpoints for file-based and URL-based predictions, returning top-3 predictions, confidence scores, validation, and error handling.
9. **Frontend Development** — Built a custom dark-themed Streamlit interface that communicates with the API over HTTP.

---

## 📈 Model Performance

| Metric              | Value   |
|-----------------------|---------|
| Training Accuracy    | ~87.9%  |
| Validation Accuracy   | ~88.3%  |
| **Test Accuracy**     | **87.7%** |
| Test Loss             | 0.329   |

---

## 🔌 API Reference

### `GET /health`
Returns backend and model status.
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict/upload`
Accepts a multipart file upload (`jpg`/`png`, max 5MB).
```json
{
  "predicted_label": "Sneaker",
  "confidence": 0.94,
  "top3": [
    { "label": "Sneaker", "confidence": 0.94 },
    { "label": "Ankle boot", "confidence": 0.04 },
    { "label": "Sandal", "confidence": 0.01 }
  ]
}
```

### `POST /predict/url`
Accepts a JSON body with an image URL.
```json
// Request
{ "url": "https://example.com/shoe.jpg" }

// Response — same structure as /predict/upload
```

Full interactive API documentation is available at `/docs` once the backend is running.

---

## 🚀 Installation

### Backend

```bash
git clone https://github.com/samichohan/Fashion-Classification-Using-CNN-DL-Project.git
cd Fashion-Classification-Using-CNN-DL-Project/backend

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000` (interactive docs at `http://localhost:8000/docs`).

### Frontend

```bash
cd ../frontend

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

Frontend runs at `http://localhost:8501` and connects to the backend at `http://localhost:8000` by default.

---

## ▶️ Usage

1. Start the backend and frontend as described above.
2. Open the Streamlit app in your browser.
3. Choose either **Upload Image** or **Image URL** tab.
4. Provide a clothing image and click **Classify Item**.
5. View the predicted category, confidence score, and top-3 predictions.

---

## ☁️ Deployment

This project uses a **two-service deployment**:

| Service   | Platform          | Notes                                            |
|-----------|-------------------|---------------------------------------------------|
| Backend   | Railway           | Deployed as a Python web service running Uvicorn  |
| Frontend  | Streamlit Cloud    | `BACKEND_URL` set via Streamlit Secrets to point to the deployed backend |

Once the backend is deployed, its public URL is added to the frontend's Streamlit Cloud secrets:
```toml
BACKEND_URL = "https://fashion-classification-using-cnn-dl-project-production.up.railway.app"
```

---

## ⚠️ Limitations & Future Improvements

- The model was trained on Fashion MNIST's simplified `28x28` grayscale, plain-background images. Performance on complex real-world photos (varied lighting, backgrounds, angles) is noticeably lower than the reported test accuracy — a natural limitation of the dataset rather than the system architecture.
- **Planned improvements:**
  - Fine-tune on a more realistic, higher-resolution clothing dataset (e.g., DeepFashion) for better real-world generalization
  - Apply data augmentation and transfer learning to improve robustness
  - Add Grad-CAM visualizations to explain predictions
  - Convert model to TensorFlow Lite for lighter, faster inference
  - Containerize the backend with Docker for consistent deployment
  - Add automated tests for API endpoints

---

## 👤 Author

**Sami Chohan**
GitHub: [@samichohan](https://github.com/samichohan)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
