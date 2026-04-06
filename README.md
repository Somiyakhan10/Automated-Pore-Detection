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
- [Overview](#overview)
- [Live Demo](#live-demo)
- [Screenshots](#screenshots)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Author](#author)
- [License](#license)

# 🔬 PoreScope — Automated SEM Pore Detection & Analysis

**AI-Powered Biomaterial Characterization Tool** · By Somiya Khan

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/somiya-khan01/pore-detection)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-red)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org/)

> **⚠️ Note:** This tool is designed for **research and educational purposes only**. It is not intended for clinical diagnosis.


---

## 📸 Screenshots

### Home Page
<img width="1913" height="796" alt="image" src="https://github.com/user-attachments/assets/13565b87-16c7-4680-befe-fdcda1e8928e" />


*Upload interface with drag-and-drop functionality*

### Analysis Results
<img width="897" height="729" alt="image" src="https://github.com/user-attachments/assets/22159d18-7017-4f7c-a1bb-61617cacb127" />


*Side-by-side comparison: Original grayscale vs. Annotated image with detected pore boundaries*

### Feature Table


<img width="1408" height="664" alt="image" src="https://github.com/user-attachments/assets/83d253df-3810-4f49-b706-cabf05093476" />


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
