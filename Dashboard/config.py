"""
config.py – Central configuration for the SEM Analysis Dashboard.
"""
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Image
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE       = (224, 224)   # CNN input
PREPROCESS_SIZE  = (512, 512)   # working resolution
SUPPORTED_EXTS   = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT    = 3.0
CLAHE_GRID          = (8, 8)
BILATERAL_D         = 9
BILATERAL_SIGMA_CLR = 75
BILATERAL_SIGMA_SPC = 75

# ─────────────────────────────────────────────────────────────────────────────
# Morphology
# ─────────────────────────────────────────────────────────────────────────────
MORPH_MIN_SIZE   = 10           # minimum object size (pixels)
MORPH_KERNEL     = 5            # morphological kernel size
MORPH_FEATURE_COLS = [
    "pore_count", "porosity", "solid_fraction",
    "area_mean", "area_std", "area_min", "area_max",
    "diameter_mean", "diameter_std",
    "circularity_mean", "circularity_std",
    "eccentricity_mean", "solidity_mean", "perimeter_mean",
    "image_entropy", "skeleton_fraction", "fractal_dimension",
]

# ─────────────────────────────────────────────────────────────────────────────
# Deep Learning
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE        = 8
RESNET_EMBED_DIM  = 2048
DINO_EMBED_DIM    = 384
PCA_COMPONENTS    = 64          # reduce embeddings before hybrid concat
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────
KMEANS_K_RANGE    = (2, 12)
KMEANS_N_INIT     = 10
HDBSCAN_MIN_SIZE  = 5
HDBSCAN_MIN_SAMP  = 3
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.1
TSNE_PERPLEXITY   = 30
TSNE_N_ITER       = 1000

# ─────────────────────────────────────────────────────────────────────────────
# Anomaly detection
# ─────────────────────────────────────────────────────────────────────────────
ISO_CONTAMINATION  = 0.05
ISO_N_ESTIMATORS   = 200
MAHA_ALPHA         = 0.01       # chi-square significance level

# ─────────────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────────────
KNN_GRAPH_K = 5

# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH       = "sem_database.db"
EXPORT_DIR    = "exports"

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
PAGE_TITLE    = "SEM Image Analysis Dashboard"
PAGE_ICON     = "🔬"
LAYOUT        = "wide"

CATEGORY_COLORS = {
    "Biological"              : "#4CAF50",
    "Fibres"                  : "#2196F3",
    "Films_Coated_Surface"    : "#FF9800",
    "MEMS_devices_and_electrodes": "#9C27B0",
    "Nanowires"               : "#F44336",
    "Particles"               : "#00BCD4",
    "Patterned_surface"       : "#FF5722",
    "Porous_Sponge"           : "#607D8B",
    "Powder"                  : "#795548",
    "Tips"                    : "#009688",
    "Unknown"                 : "#9E9E9E",
}
