

```markdown
# 🔬 PoreScope — Automated SEM Pore Detection & Analysis

**AI-Powered Biomaterial Characterization Tool** · By Somiya Khan

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/somiya-khan01/pore-detection)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-red)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org/)

> **⚠️ Note:** This tool is designed for **research and educational purposes only**. It is not intended for clinical diagnosis or regulatory decision-making.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Screenshots](#-screenshots)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [License](#-license)

---

## 🔬 Overview

**PoreScope** is an automated web application that detects and analyzes pores from **Scanning Electron Microscopy (SEM)** images of biomaterials (hydrogels, porous scaffolds, membranes). The system uses **computer vision** (OpenCV) and **Random Forest regression** to predict:

- ✅ **Pore Diameter** (pixels)
- ✅ **Porosity (%)**
- ✅ **Pore Count**
- ✅ **Intensity Statistics** (mean, std deviation)

The tool is designed for **high-throughput biomaterial characterization**, enabling rapid quality assessment of porous structures used in **tissue engineering, drug delivery, and wound healing applications**.

---

## 🌐 Live Demo

The application is live and accessible at:

### 🔗 [https://huggingface.co/spaces/somiya-khan01/pore-detection](https://huggingface.co/spaces/somiya-khan01/pore-detection)

---

## 📸 Screenshots

### Home Page


<img width="1920" height="810" alt="image" src="https://github.com/user-attachments/assets/20968717-a467-4b7c-b323-8d2e9153992e" />


### Analysis Results
<img width="1920" height="717" alt="image" src="https://github.com/user-attachments/assets/8f605f68-19e9-4cf5-bdd9-e02bb600f3eb" />


### Feature Table
<img width="1919" height="697" alt="image" src="https://github.com/user-attachments/assets/c3bbe0fa-a2dd-4e36-b007-8e6dd415ee41" />


*Extracted pore metrics and image statistics*

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Pore Detection Accuracy** | **98.6%** |
| **Pore Diameter Prediction Accuracy** | **91.0%** |
| **Porosity Prediction Accuracy** | **78.9%** |
| **Algorithm** | Random Forest Regression |
| **Input Features** | 5 (mean, std, min, max intensity + pore count) |
| **Training Images** | 1,000 SEM images |

### Confusion Matrix (Pore Detection)

| | Detected Pore | Detected Non-Pore |
|---|---|---|
| **Actual Pore** | 98.6% | 1.4% |
| **Actual Non-Pore** | 2.1% | 97.9% |

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | SEM Images of Nanoscale Features |
| **Source** | Kaggle |
| **Total Images** | 18,577 SEM images |
| **Categories** | 10 material types (Porous_Sponge, Fibres, Particles, etc.) |
| **Training Set** | 1,000 images |
| **Image Formats** | JPG, PNG, TIF |

### Material Categories in Dataset
- Porous_Sponge
- Fibres
- Particles
- Biological
- Nanowires
- Patterned_surface
- Tips
- Films_Coated_Surface
- MEMS_devices_and_electrodes
- Powder

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INPUT                                │
│   User uploads SEM image (JPG / PNG) via drag-and-drop         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   IMAGE PREPROCESSING                          │
│   • Convert to grayscale                                       │
│   • Resize to max 1024px (preserve aspect ratio)               │
│   • Otsu thresholding for pore/background segmentation         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CONTOUR DETECTION                            │
│   • Find all contours using OpenCV                            │
│   • Filter by area > 20 px² (remove noise)                    │
│   • Draw pore boundaries on original image                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION                           │
│   Extract 5 features:                                          │
│   • mean_intensity   • std_intensity                          │
│   • min_intensity    • max_intensity                          │
│   • pore_count                                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RANDOM FOREST PREDICTION                     │
│   Three RF regressors predict:                                │
│   • Avg Pore Diameter (91% accuracy)                          │
│   • Porosity (78.9% accuracy)                                 │
│   • Pore Count (98.6% accuracy)                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RESULTS DISPLAY                           │
│   • KPI cards with animated progress bars                     │
│   • Side-by-side image comparison (original vs annotated)     │
│   • Interactive slider view                                   │
│   • Download CSV / PNG / JSON reports                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| **Machine Learning** | scikit-learn (Random Forest Regressor) |
| **Image Processing** | OpenCV 4.8 (contour detection, thresholding) |
| **Backend** | Flask 3.0 |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Deployment** | Hugging Face Spaces (Docker) |
| **Data Processing** | NumPy, PIL |

---

## 📁 Project Structure

```
pore-detection/
│
├── app.py                 # Flask application + API routes
├── image_processing.py    # Core pore detection logic
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── README.md              # Documentation
│
├── templates/
│   └── index.html         # Main UI template
│
├── static/
│   ├── style.css          # Stylesheet
│   └── script.js          # Frontend logic (drag-drop, API calls, charts)
│
└── models/                # Trained Random Forest models
    ├── model_diameter.pkl
    ├── model_porosity.pkl
    ├── model_count.pkl
    └── scaler.pkl
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Local Setup

```bash
# Clone the repository
git clone https://huggingface.co/spaces/somiya-khan01/pore-detection

# Navigate to project directory
cd pore-detection

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Open browser and go to http://localhost:7860
```

### Docker Setup

```bash
# Build Docker image
docker build -t porescope .

# Run container
docker run -p 7860:7860 porescope
```

### Dependencies (`requirements.txt`)

```txt
flask>=2.3.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
Pillow>=10.0.0
pandas>=2.0.0
gunicorn>=21.2.0
```

---

## 📡 API Reference

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": ["diameter", "porosity", "count", "scaler"]
}
```

### POST `/predict`
Upload an SEM image for pore analysis.

**Request:** `multipart/form-data` with field `image`

**Response:**
```json
{
  "pore_diameter": 40.62,
  "porosity": 68.01,
  "pore_count": 27,
  "mean_intensity": 140.51,
  "std_intensity": 41.63,
  "annotated_image": "base64_encoded_png",
  "original_image": "base64_encoded_png",
  "method": "Random Forest ML Model",
  "image_size": "1024×620"
}
```

---

## 📊 Output Features

| Feature | Description |
|---------|-------------|
| **Average Pore Diameter** | Mean equivalent circular diameter of detected pores (pixels) |
| **Porosity (%)** | Fraction of image area classified as pore space |
| **Pore Count** | Number of valid contours (area > 20 px²) |
| **Mean Pixel Intensity** | Average grayscale value (0–255) |
| **Std Dev Intensity** | Contrast / texture measure |
| **Image Size** | Width × Height after preprocessing |

---

## 👩‍🔬 Author

**Somiya Khan**  
Biomedical Engineer | AI/ML Enthusiast

- [GitHub](https://github.com/somiya-khan01)
- [Hugging Face](https://huggingface.co/somiya-khan01)

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgments

- **Dataset:** SEM Images of Nanoscale Features — [Kaggle](https://www.kaggle.com/datasets/adrianacosta0/sem-images-of-nanoscale-features)
- **Inspiration:** High-throughput biomaterial characterization for tissue engineering applications
- **Built with:** OpenCV, scikit-learn, Flask, Hugging Face Spaces

---

## ⭐ Star this Project

If you find this project useful for your research or learning, please consider giving it a star on Hugging Face!

---

**Built with ❤️ by Somiya Khan**
```

This README is formatted for your PoreScope project and matches the style of your breast cancer classification README. Just replace the placeholder image names (`Analysisreport.png`, `Prediction.png`, `Features.png`) with your actual screenshot filenames.
