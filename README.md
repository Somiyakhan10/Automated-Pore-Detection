<div align="center">
  
# 🔬 SEM Image Analysis Dashboard

**Hybrid Multi-Scale Feature Learning for SEM Microstructure Analysis**

<br>

<a href="https://huggingface.co/spaces/somiya-khan01/SEM_analysis" target="_blank">
  <img src="https://img.shields.io/badge/🚀_LAUNCH_DEMO_-TRY_NOW-FF5722?style=for-the-badge&logo=huggingface&logoColor=white" alt="Launch Demo" width="300">
</a>

<br>

</div>



## Overview

This project presents a comprehensive computational pipeline for analyzing Scanning Electron Microscope (SEM) images of nanomaterials. The system extracts hybrid features combining morphological measurements, deep learning embeddings, and graph-based structural metrics to enable unsupervised clustering, anomaly detection, and similarity-based image retrieval across 10 nanomaterial categories.

The pipeline is designed for materials scientists and biomedical engineers working with porous scaffolds, nanoparticles, nanowires, and other microstructured materials where quantitative characterization is essential for quality control and research.


## Key Features

| Feature | Description |
|---------|-------------|
| Multi-Scale Feature Extraction | Combines morphological (pore size, porosity, fractal dimension), deep learning (ResNet-50/DINOv2), and graph-based (Delaunay triangulation) features |
| Unsupervised Clustering | UMAP dimensionality reduction with HDBSCAN and K-Means clustering for automatic material categorization |
| Anomaly Detection | Dual detection using Isolation Forest and Mahalanobis distance to identify rare or defective structures |
| Similarity Search | Cosine similarity-based image retrieval using morphology fingerprints |
| Interactive Visualizations | Dark-mode plots including UMAP projections, category distributions, and similarity matrices |


## Dataset

The system analyzes SEM images across **10 nanomaterial categories**:

| Category | Description |
|----------|-------------|
| Biological | Biological samples and tissues |
| Fibres | Fibrous structures and mats |
| Films Coated Surface | Thin films and coated surfaces |
| MEMS Devices | Micro-electromechanical systems |
| Nanowires | Wire-like nanostructures |
| Particles | Nanoparticles and microparticles |
| Patterned Surface | Lithographically patterned surfaces |
| Porous Sponge | Porous foam and sponge structures |
| Powder | Powdered materials |
| Tips | Sharp tips and probes |

Dataset Source: [SEM Images with Nanoscale Features](https://www.kaggle.com/datasets/adrianacosta0/sem-images-with-nanoscale-features)


## Methodology

## Methodology Flowchart
## Results Gallery

### Preprocessing Pipeline

<div align="center">
  <img width="4369" height="2955" alt="03_preprocessing_pipeline" src="https://github.com/user-attachments/assets/93255644-1f13-46e0-bc7f-9560617c68e2" />

  Preprocessing pipeline showing original image, segmentation mask, overlay, and contour detection for three material categories</em></p>
</div>
      

### Anomaly Detection

<div align="center">
<img width="5370" height="1485" alt="08b_anomaly_detection" src="https://github.com/user-attachments/assets/3da6f2cc-6c46-40a6-b3df-ade3b1aef393" />

Anomaly detection using Isolation Forest (left, 5% contamination), Mahalanobis distance (center, p<0.01), and combined anomalies (right)</em></p>
</div>


### Segmentation Results

<div align="center">
   <img width="4463" height="5307" alt="09_segmentation_results" src="https://github.com/user-attachments/assets/80a525fb-f758-4454-bbc8-c86ba220b32f" />

Segmentation masks with Dice scores for multiple material categories </em></p>
</div>
      

