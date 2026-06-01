# SEM Image Analysis System

Multi-modal feature extraction and unsupervised clustering for nanomaterial classification using Scanning Electron Microscope images.

<div align="center">
    <h1>SEM Image Analysis System</h1>
    
    <a href="https://github.com/yourusername/sem-image-analysis" target="_blank">
        <button style="background-color: #3b82f6; color: white; font-size: 16px; font-weight: bold; padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-family: sans-serif;">
            View on GitHub
        </button>
    </a>
</div>

---

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

### 1. Preprocessing
- CLAHE contrast enhancement
- Bilateral filtering for noise reduction
- Otsu thresholding for segmentation
- Morphological cleaning (opening/closing)

### 2. Feature Extraction

| Feature Type | Dimensions | Description |
|--------------|------------|-------------|
| Morphological | 11 | Pore count, porosity, fractal dimension, circularity, solidity, entropy, skeleton fraction |
| Deep Learning | 384 or 2048 | ResNet-50 or DINOv2 embeddings (ImageNet pretrained) |
| SimCLR-style | 384 or 2048 | Contrastive augmented views for robust representations |
| Graph-based | 7 | Delaunay triangulation: nodes, edges, density, clustering coefficient, betweenness centrality |
| Hybrid Fingerprint | 139 | Concatenated and PCA-reduced (morphology + deep + SimCLR + graph) |

### 3. Dimensionality Reduction & Clustering
- PCA for initial variance reduction
- UMAP (or t-SNE fallback) for 2D visualization
- HDBSCAN for density-based clustering
- K-Means for centroid-based baseline
- Silhouette score and Davies-Bouldin index for validation

### 4. Anomaly Detection
- **Isolation Forest**: Tree-based outlier detection (5% contamination)
- **Mahalanobis Distance**: Multivariate Gaussian test (p < 0.01 threshold)
- Combined anomalies flagged by both methods

### 5. Similarity Search
- Cosine similarity on L2-normalized hybrid fingerprints
- k-NN retrieval (k=5) with category labels
- Category-level similarity matrix heatmap



## Results

### Sample Images by Category

The pipeline processes SEM images across all 10 categories, with sample outputs shown below:

| Category | Sample | Category | Sample |
|----------|--------|----------|--------|
| Biological | ✓ | Fibres | ✓ |
| Films Coated | ✓ | MEMS | ✓ |
| Nanowires | ✓ | Particles | ✓ |
| Patterned | ✓ | Porous | ✓ |
| Powder | ✓ | Tips | ✓ |

### Preprocessing Pipeline

Each image undergoes:
1. CLAHE contrast enhancement
2. Bilateral filtering
3. Otsu thresholding segmentation
4. Contour detection for morphological analysis

### Clustering Results

| Method | Silhouette Score | Davies-Bouldin Index |
|--------|------------------|----------------------|
| K-Means (optimal k) | >0.40 | <1.50 |
| HDBSCAN | >0.40 | <1.50 |

### Anomaly Detection

| Detector | Anomaly Rate |
|----------|--------------|
| Isolation Forest | 5% |
| Mahalanobis Distance | ~5% (p<0.01) |
| Combined (both) | ~2-3% |

### Similarity Search

The morphology fingerprint system achieves:
- Intra-category similarity >0.80 for most categories
- Cross-category discrimination visible in similarity matrix
- Top-5 retrieval accuracy demonstrating meaningful feature representations

## Results Gallery

### Preprocessing Pipeline

<div align="center">
   <img width="4369" height="2955" alt="03_preprocessing_pipeline" src="https://github.com/user-attachments/assets/28047f3a-d915-4f75-be15-259d86fefae2" />

    <p><em>Figure 1: Preprocessing pipeline showing original image, segmentation mask, overlay, and contour detection for three material categories</em></p>
</div>

### Similarity Search Results

The morphology fingerprint system retrieves the top 5 most similar images for a given query using cosine similarity.

<div align="center">
    <table>
        <tr>
            <td align="center" colspan="2">
                <strong>Query: Biological</strong>
            </td>
        </tr>
        <tr>
            <td align="center">
        <img width="2804" height="2358" alt="10b_category_similarity_matrix" src="https://github.com/user-attachments/assets/89a5c1eb-7d3b-4ca2-8e68-a59b436a1d9d" />

                <br>NN-1 | sim=0.804
                <br>Biological
            </td>
            <td align="center">
                <img src="images/similarity_biological_4.png" alt="Biological Result 4" width="150">
                <br>NN-4 | sim=0.763
                <br>Biological
            </td>
        </tr>
        <tr>
            <td align="center" colspan="2">
                <img src="images/similarity_biological_5.png" alt="Biological Result 5" width="150">
                <br>NN-5 | sim=0.724
                <br>Nanowires (cross-category match)
            </td>
        </tr>
    </table>
</div>

<div align="center">
    <table>
        <tr>
            <td align="center" colspan="2">
                <strong>Query: MEMS Devices and Electrodes</strong>
            </td>
        </tr>
        <tr>
            <td align="center">
                <img src="images/similarity_mems_1.png" alt="MEMS Result 1" width="150">
                <br>NN-1 | sim=0.941
                <br>MEMS Devices
            </td>
            <td align="center">
                <img src="images/similarity_mems_4.png" alt="MEMS Result 4" width="150">
                <br>NN-4 | sim=0.871
                <br>Biological
            </td>
        </tr>
        <tr>
            <td align="center" colspan="2">
                <img src="images/similarity_mems_5.png" alt="MEMS Result 5" width="150">
                <br>NN-5 | sim=0.869
                <br>Patterned Surface / Powder
            </td>
        </tr>
    </table>
    <p><em>Figure 2: Similarity search results showing top matches with similarity scores (0.72-0.94)</em></p>
</div>

### Category Similarity Matrix

<div align="center">
    <img src="images/10b_category_similarity_matrix.png" alt="Category Similarity Matrix" width="700">
    <p><em>Figure 3: Mean cosine similarity heatmap between 10 nanomaterial categories. Darker green indicates higher similarity.</em></p>
</div>

### Clustering Analysis

<div align="center">
    <img src="images/06_clustering_analysis.png" alt="Clustering Analysis" width="900">
    <p><em>Figure 4: UMAP projections showing K-Means clustering (top-right), HDBSCAN clustering (bottom-left), and true category labels (bottom-center)</em></p>
</div>

### Anomaly Detection

<div align="center">
    <img src="images/08b_anomaly_detection.png" alt="Anomaly Detection" width="900">
    <p><em>Figure 5: Anomaly detection using Isolation Forest (left, 5% contamination), Mahalanobis distance (center, p<0.01), and combined anomalies (right)</em></p>
</div>

### Category Distribution

<div align="center">
    <img src="images/02_category_distribution.png" alt="Category Distribution" width="700">
    <p><em>Figure 6: Distribution of SEM images across 10 nanomaterial categories</em></p>
</div>

### Morphology Feature Distributions

<div align="center">
    <img src="images/04_morphology_boxplots.png" alt="Morphology Box Plots" width="900">
    <p><em>Figure 7: Box plots showing morphological feature distributions (pore count, porosity, diameter, area, circularity, fractal dimension) across all categories</em></p>
</div>

### Graph Structures on SEM Images

<div align="center">
    <img src="images/05_graph_structures.png" alt="Graph Structures" width="900">
    <p><em>Figure 8: Delaunay triangulation graph structures overlaid on skeletonized SEM images for six material categories</em></p>
</div>

### Segmentation Results

<div align="center">
    <img src="images/09_segmentation_results.png" alt="Segmentation Results" width="900">
    <p><em>Figure 9: Segmentation masks with Dice scores for multiple material categories</em></p>
</div>


