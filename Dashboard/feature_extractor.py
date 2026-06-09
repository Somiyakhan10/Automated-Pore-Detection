"""
utils/feature_extractor.py
All feature-extraction logic: preprocessing, morphology, graph metrics,
SimCLR augmentation embeddings, and hybrid vector construction.
"""
from __future__ import annotations

import warnings
import gc
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops, shannon_entropy

import torch
import torchvision.transforms as T
from PIL import Image as PILImage
from torch.utils.data import Dataset, DataLoader

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    CLAHE_CLIP_LIMIT, CLAHE_GRID, BILATERAL_D,
    BILATERAL_SIGMA_CLR, BILATERAL_SIGMA_SPC,
    MORPH_MIN_SIZE, MORPH_KERNEL,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, DEVICE, KNN_GRAPH_K, PCA_COMPONENTS,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(
    image_input,
    target_size: Tuple[int, int] = (512, 512),
) -> Optional[np.ndarray]:
    """
    Accept a file path (str), numpy array, or bytes.
    Returns float32 RGB [0,1] or None on failure.
    """
    try:
        if isinstance(image_input, (str, bytes, os.PathLike)):
            img = cv2.imread(str(image_input))
            if img is None:
                return None
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            # bytes from Streamlit uploader
            arr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

        # CLAHE on L channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Bilateral denoise
        img = cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SIGMA_CLR, BILATERAL_SIGMA_SPC)
        return img.astype(np.float32) / 255.0
    except Exception:
        return None


def generate_segmentation_mask(image: np.ndarray) -> np.ndarray:
    """Otsu + morphological cleanup → float32 mask [0,1]."""
    try:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = 255 - mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = (
            remove_small_objects((mask > 0).astype(bool), min_size=MORPH_MIN_SIZE)
            .astype(np.uint8) * 255
        )
        return mask.astype(np.float32) / 255.0
    except Exception:
        return np.zeros(image.shape[:2], dtype=np.float32)


def build_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask * 0.5, 0, 1)
    return overlay


def build_contour_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    cimg = image.copy()
    cv2.drawContours(cimg, contours, -1, (0, 1, 0), 2)
    return cimg


# ─────────────────────────────────────────────────────────────────────────────
# 2. MORPHOLOGY FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_morphology_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Extract 17 hand-crafted features at micro / meso / macro scale."""
    mask_binary = (mask > 0.5).astype(np.uint8)
    features: Dict[str, float] = {}

    # ── Micro: region props ──
    _REGION_KEYS = [
        "pore_count", "area_mean", "area_std", "area_min", "area_max",
        "diameter_mean", "diameter_std", "circularity_mean", "circularity_std",
        "eccentricity_mean", "solidity_mean", "perimeter_mean",
    ]
    try:
        labeled  = label(mask_binary)
        regions  = regionprops(labeled)
        if regions:
            areas  = [r.area       for r in regions]
            perims = [r.perimeter  for r in regions]
            diams  = [2 * np.sqrt(a / np.pi) for a in areas]
            circs  = [4 * np.pi * a / (p ** 2 + 1e-7) for a, p in zip(areas, perims)]
            eccens = [r.eccentricity for r in regions]
            solids = [r.solidity     for r in regions]
            features.update({
                "pore_count"       : float(len(regions)),
                "area_mean"        : float(np.mean(areas)),
                "area_std"         : float(np.std(areas)),
                "area_min"         : float(np.min(areas)),
                "area_max"         : float(np.max(areas)),
                "diameter_mean"    : float(np.mean(diams)),
                "diameter_std"     : float(np.std(diams)),
                "circularity_mean" : float(np.mean(circs)),
                "circularity_std"  : float(np.std(circs)),
                "eccentricity_mean": float(np.mean(eccens)),
                "solidity_mean"    : float(np.mean(solids)),
                "perimeter_mean"   : float(np.mean(perims)),
            })
        else:
            features.update({k: 0.0 for k in _REGION_KEYS})
    except Exception:
        features.update({k: 0.0 for k in _REGION_KEYS})

    # ── Meso: entropy + skeleton ──
    try:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        features["image_entropy"] = float(shannon_entropy(gray))
    except Exception:
        features["image_entropy"] = 0.0

    try:
        skel = skeletonize(mask_binary > 0)
        features["skeleton_fraction"] = float(skel.sum() / (mask_binary.sum() + 1e-7))
    except Exception:
        features["skeleton_fraction"] = 0.0

    # ── Macro: porosity + fractal ──
    porosity = mask_binary.sum() / (mask_binary.shape[0] * mask_binary.shape[1])
    features["porosity"]       = float(porosity)
    features["solid_fraction"] = float(1.0 - porosity)

    try:
        sizes = np.logspace(0.5, 4, num=10, base=2).astype(int)
        counts_fd = []
        for s in sizes:
            if s >= min(mask_binary.shape):
                break
            cnt = sum(
                1 for i in range(0, mask_binary.shape[0], s)
                  for j in range(0, mask_binary.shape[1], s)
                  if np.any(mask_binary[i : i + s, j : j + s])
            )
            counts_fd.append(cnt)
        if len(counts_fd) >= 2:
            slope = np.polyfit(np.log(sizes[: len(counts_fd)]), np.log(counts_fd), 1)[0]
            features["fractal_dimension"] = float(-slope)
        else:
            features["fractal_dimension"] = 2.0
    except Exception:
        features["fractal_dimension"] = 2.0

    return features


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def build_delaunay_graph(centroids: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    if len(centroids) == 0:
        return G
    for i, c in enumerate(centroids):
        G.add_node(i, pos=tuple(c))
    if len(centroids) < 3:
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                w = float(np.linalg.norm(centroids[i] - centroids[j]))
                G.add_edge(i, j, weight=w)
        return G
    tri = Delaunay(centroids)
    for simplex in tri.simplices:
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            u, v = int(simplex[a]), int(simplex[b])
            w = float(np.linalg.norm(centroids[u] - centroids[v]))
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=w)
    return G


def build_knn_graph(centroids: np.ndarray, k: int = KNN_GRAPH_K) -> nx.Graph:
    G = nx.Graph()
    if len(centroids) == 0:
        return G
    k = min(k, len(centroids) - 1)
    for i, c in enumerate(centroids):
        G.add_node(i, pos=tuple(c))
    if k < 1:
        return G
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(centroids)
    dists, indices = nbrs.kneighbors(centroids)
    for i in range(len(centroids)):
        for j_idx, dist in zip(indices[i, 1:], dists[i, 1:]):
            G.add_edge(i, int(j_idx), weight=float(dist))
    return G


def compute_graph_metrics(G: nx.Graph) -> Dict[str, float]:
    n, e = len(G), G.number_of_edges()
    metrics: Dict[str, float] = {
        "num_nodes"           : float(n),
        "num_edges"           : float(e),
        "density"             : float(nx.density(G)),
        "avg_degree"          : float(2 * e / n) if n > 0 else 0.0,
        "num_components"      : float(nx.number_connected_components(G)),
        "avg_clustering_coeff": float(nx.average_clustering(G)) if n > 1 else 0.0,
    }
    try:
        bc = nx.betweenness_centrality(G, normalized=True)
        metrics["mean_betweenness"] = float(np.mean(list(bc.values())))
    except Exception:
        metrics["mean_betweenness"] = 0.0
    try:
        lcc   = max(nx.connected_components(G), key=len)
        Gsub  = G.subgraph(lcc).copy()
        if len(Gsub) > 1:
            metrics["avg_path_length"] = float(nx.average_shortest_path_length(Gsub))
            metrics["diameter"]        = float(nx.diameter(Gsub))
        else:
            metrics["avg_path_length"] = 0.0
            metrics["diameter"]        = 0.0
    except Exception:
        metrics["avg_path_length"] = 0.0
        metrics["diameter"]        = 0.0
    return metrics


def extract_graph_features(mask: np.ndarray, mode: str = "delaunay") -> Tuple[nx.Graph, Dict[str, float]]:
    mask_binary = (mask > 0.5).astype(np.uint8)
    labeled  = label(mask_binary)
    regions  = regionprops(labeled)
    _ZERO = {k: 0.0 for k in [
        "num_nodes", "num_edges", "density", "avg_degree",
        "num_components", "avg_clustering_coeff",
        "mean_betweenness", "avg_path_length", "diameter"
    ]}
    if not regions:
        return nx.Graph(), _ZERO
    centroids = np.array([r.centroid for r in regions])
    G = build_knn_graph(centroids, k=KNN_GRAPH_K) if mode == "knn" else build_delaunay_graph(centroids)
    return G, compute_graph_metrics(G)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DEEP LEARNING EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

_imagenet_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_aug_tf_1 = T.Compose([
    T.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_aug_tf_2 = T.Compose([
    T.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.5, 0.9)),
    T.RandomHorizontalFlip(p=0.3),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.4),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class _SEMDataset(Dataset):
    def __init__(self, images: List[np.ndarray], transform=None):
        self.images    = images
        self.transform = transform or _imagenet_tf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = (self.images[idx] * 255).clip(0, 255).astype(np.uint8)
        pil = PILImage.fromarray(img).resize(IMAGE_SIZE, PILImage.BILINEAR)
        return self.transform(pil)


class _DualAugDataset(Dataset):
    def __init__(self, images: List[np.ndarray]):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = (self.images[idx] * 255).clip(0, 255).astype(np.uint8)
        pil = PILImage.fromarray(img).resize(IMAGE_SIZE, PILImage.BILINEAR)
        return _aug_tf_1(pil), _aug_tf_2(pil)


def extract_deep_embeddings(
    images: List[np.ndarray],
    backbone,
    backbone_name: str,
    batch_size: int = BATCH_SIZE,
    device: torch.device = DEVICE,
    progress_cb=None,
) -> np.ndarray:
    """Returns L2-normalised (N, D) embedding matrix."""
    dataset = _SEMDataset(images)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs: List[np.ndarray] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            if backbone_name == "dinov2":
                out = backbone(batch)
            else:
                out = backbone(batch).squeeze(-1).squeeze(-1)
            embs.append(out.cpu().numpy())
            if progress_cb:
                progress_cb(i + 1, len(loader))
    result = np.vstack(embs)
    return normalize(result, norm="l2")


def extract_simclr_embeddings(
    images: List[np.ndarray],
    backbone,
    backbone_name: str,
    batch_size: int = BATCH_SIZE,
    device: torch.device = DEVICE,
    progress_cb=None,
) -> np.ndarray:
    """Average of two augmented views, L2-normalised."""
    dataset = _DualAugDataset(images)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs: List[np.ndarray] = []
    with torch.no_grad():
        for i, (v1, v2) in enumerate(loader):
            v1, v2 = v1.to(device), v2.to(device)
            if backbone_name == "dinov2":
                e1, e2 = backbone(v1), backbone(v2)
            else:
                e1 = backbone(v1).squeeze(-1).squeeze(-1)
                e2 = backbone(v2).squeeze(-1).squeeze(-1)
            embs.append(((e1 + e2) / 2).cpu().numpy())
            if progress_cb:
                progress_cb(i + 1, len(loader))
    result = np.vstack(embs)
    return normalize(result, norm="l2")


# ─────────────────────────────────────────────────────────────────────────────
# 5. HYBRID FEATURE VECTOR
# ─────────────────────────────────────────────────────────────────────────────

MORPH_COLS_HYBRID = [
    "pore_count", "porosity", "diameter_mean", "area_mean",
    "circularity_mean", "fractal_dimension",
    "image_entropy", "skeleton_fraction", "solid_fraction",
    "eccentricity_mean", "solidity_mean",
]


def build_hybrid_features(
    morph_df,
    deep_emb: np.ndarray,
    simclr_emb: np.ndarray,
    pca_components: int = PCA_COMPONENTS,
) -> Tuple[np.ndarray, PCA, PCA, StandardScaler]:
    """
    Returns (X_hybrid_scaled, pca_deep, pca_simclr, scaler_hybrid).
    """
    import pandas as pd

    X_morph = morph_df[MORPH_COLS_HYBRID].fillna(0).values
    scaler_morph = StandardScaler()
    X_morph_sc   = scaler_morph.fit_transform(X_morph)

    n_comp = min(pca_components, deep_emb.shape[1])
    pca_deep = PCA(n_components=n_comp, random_state=42)
    X_deep_r = pca_deep.fit_transform(deep_emb)

    pca_sim = PCA(n_components=n_comp, random_state=42)
    X_sim_r = pca_sim.fit_transform(simclr_emb)

    X_hybrid = np.hstack([X_morph_sc, X_deep_r, X_sim_r])
    scaler_h = StandardScaler()
    X_hybrid_sc = scaler_h.fit_transform(X_hybrid)

    return X_hybrid_sc, pca_deep, pca_sim, scaler_hybrid_holder(scaler_morph, scaler_h)


class scaler_hybrid_holder:
    """Lightweight container so we can pass both scalers together."""
    def __init__(self, morph_scaler, hybrid_scaler):
        self.morph  = morph_scaler
        self.hybrid = hybrid_scaler


def build_single_hybrid_vector(
    morph_feats: Dict[str, float],
    deep_single: np.ndarray,
    simclr_single: np.ndarray,
    pca_deep: PCA,
    pca_simclr: PCA,
    scalers,
) -> np.ndarray:
    """
    Project a single image's features into the trained hybrid space.
    """
    import pandas as pd

    morph_vec = np.array([[morph_feats.get(c, 0.0) for c in MORPH_COLS_HYBRID]])
    morph_sc  = scalers.morph.transform(morph_vec)

    d_r = pca_deep.transform(deep_single.reshape(1, -1))
    s_r = pca_simclr.transform(simclr_single.reshape(1, -1))

    hybrid = np.hstack([morph_sc, d_r, s_r])
    return scalers.hybrid.transform(hybrid).squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# 6. SIMILARITY SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity_search(
    query_vec: np.ndarray,
    database_vecs: np.ndarray,
    k: int = 5,
    exclude_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (indices, similarities) of top-k results."""
    q    = normalize(query_vec.reshape(1, -1), norm="l2")
    db   = normalize(database_vecs, norm="l2")
    sims = (db @ q.T).flatten()
    if exclude_idx is not None and 0 <= exclude_idx < len(sims):
        sims[exclude_idx] = -np.inf
    order = np.argsort(sims)[::-1][:k]
    return order, sims[order]
