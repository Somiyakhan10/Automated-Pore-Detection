# 🔬 SEM Image Analysis Dashboard

A production-ready Streamlit dashboard for comprehensive Scanning Electron Microscope (SEM) image analysis, featuring deep learning embeddings, clustering, anomaly detection, graph-based structural analysis, and morphology-fingerprint similarity search.

---

## Features

| Capability | Details |
|---|---|
| **Preprocessing** | CLAHE enhancement, bilateral filter, Otsu segmentation |
| **Morphology** | 17 features: porosity, fractal dimension, circularity, entropy, skeleton fraction… |
| **Deep Embeddings** | ResNet-50 (2048-D) or DINOv2 ViT-S/14 (384-D) |
| **SimCLR Embeddings** | Two-view augmentation contrastive embeddings |
| **Hybrid Features** | Morphology + Deep PCA-64 + SimCLR PCA-64 |
| **Clustering** | K-Means (elbow + silhouette), HDBSCAN, Ward dendrogram |
| **Dimensionality** | UMAP or t-SNE 2-D projection |
| **Anomaly Detection** | Isolation Forest + Mahalanobis chi-square test |
| **Graph Analysis** | Delaunay triangulation + k-NN, betweenness centrality, path length |
| **Similarity Search** | Cosine k-NN over hybrid fingerprint vectors |
| **Database** | SQLite with export to CSV / JSON / NPZ |

---

## Quick Start

### 1. Local (Python venv)

```bash
# Clone / copy files
cd sem_dashboard

# Create environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

### 2. Docker

```bash
# Build and run
docker compose up --build

# Or without GPU
docker compose up --build sem-dashboard
```

Open **http://localhost:8501**.

> **GPU support**: Docker Compose is configured for NVIDIA GPUs via the `nvidia` runtime. Remove the `deploy.resources` block in `docker-compose.yml` if running CPU-only.

---

## Project Structure

```
sem_dashboard/
├── app.py                      # Main Streamlit application
├── config.py                   # All configuration constants
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── models/
│   ├── __init__.py
│   └── model_manager.py        # Backbone loading (ResNet-50 / DINOv2)
└── utils/
    ├── __init__.py
    ├── feature_extractor.py    # Preprocessing, morphology, graph, embeddings
    ├── visualization.py        # Plotly / matplotlib figure factories
    └── database.py             # SQLite feature store
```

---

## Dashboard Tabs

### 🔬 Single Analysis
Upload one SEM image and get the full pipeline instantly:
- Preprocessing strip (original → CLAHE → mask → overlay → contours)
- 17 morphology metrics with key-metric cards
- Feature radar chart
- Delaunay graph overlaid on image
- Deep + SimCLR embedding bar charts

### 📦 Batch Analysis
Upload many images at once:
- Progress bar processing
- Summary morphology table
- Feature distribution box-plots
- Correlation heatmap
- CSV / JSON export

### 🗄️ Database
Browse and manage stored images:
- Category distribution charts
- Category-wise statistics table
- ANOVA significance tests
- Correlation and PCA loading heatmaps
- One-click CSV / NPZ / JSON exports

### 🔵 Clustering Explorer
Runs on all images stored in the database:
- Elbow / silhouette / Davies-Bouldin curves
- K-Means with automatic optimal-k selection
- HDBSCAN (optional, install `hdbscan`)
- UMAP or t-SNE 2-D scatter by cluster / category
- Isolation Forest + Mahalanobis anomaly maps
- Anomaly report CSV download

### 🔍 Similarity Search
- Upload a new query image **or** select one from the database
- Returns top-K most similar images with cosine similarity scores
- Category-level mean similarity heatmap

---

## Configuration

All tuneable parameters live in **`config.py`**:

```python
IMAGE_SIZE      = (224, 224)    # CNN input resolution
PREPROCESS_SIZE = (512, 512)    # Working resolution
CLAHE_CLIP_LIMIT = 3.0
KMEANS_K_RANGE  = (2, 12)
ISO_CONTAMINATION = 0.05        # Anomaly fraction
MAHA_ALPHA      = 0.01          # chi-square p-value threshold
PCA_COMPONENTS  = 64            # Embedding reduction
```

The sidebar also exposes runtime controls:
- UMAP `n_neighbors` / `min_dist`
- Anomaly contamination rate
- Max K for K-Means search
- Toggle HDBSCAN on/off
- Try DINOv2 (requires internet on first run)

---

## Optional Dependencies

| Package | Why | Install |
|---|---|---|
| `umap-learn` | Better topology-preserving 2-D projection | `pip install umap-learn` |
| `hdbscan` | Density-based clustering, auto-selects cluster count | `pip install hdbscan` |

Both fall back gracefully (t-SNE / DBSCAN) if not installed.

---

## Hardware Notes

- **CPU**: Fully supported. ResNet-50 inference is ~0.5 s/image.
- **GPU (CUDA)**: Detected automatically via `torch.cuda.is_available()`. Batch inference is 10–20× faster.
- **Memory**: Each image uses ~50 MB during feature extraction; released after processing.

---

## Supported Image Formats

`.jpg` · `.jpeg` · `.png` · `.tif` · `.tiff`

SEM images should be grayscale or RGB; the pipeline handles both automatically.

---

## Database

Features are stored in **`sem_database.db`** (SQLite, single file). Embeddings are serialised as base64-pickled numpy arrays. The file grows at roughly **~3 MB per 100 images** (2048-D ResNet embeddings + morphology + graph + thumbnails).

To start fresh: click **Clear Database** in the sidebar or `rm sem_database.db`.
