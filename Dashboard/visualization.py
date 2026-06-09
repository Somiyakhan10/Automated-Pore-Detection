"""
utils/visualization.py
All plotting helpers – returns Plotly figures or matplotlib figures.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import seaborn as sns

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CATEGORY_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cat_color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, CATEGORY_COLORS["Unknown"])


def _discrete_color_map(categories: List[str]):
    unique = sorted(set(categories))
    return {c: _cat_color(c) for c in unique}


# ─────────────────────────────────────────────────────────────────────────────
# 1. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def plot_preprocessing_pipeline(
    original: np.ndarray,
    preprocessed: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    contour_img: np.ndarray,
) -> go.Figure:
    """5-panel preprocessing pipeline figure."""
    titles = ["Original", "CLAHE Enhanced", "Segmentation Mask", "Overlay", "Contours"]
    imgs   = [original, preprocessed, mask, overlay, contour_img]

    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=titles,
        horizontal_spacing=0.02,
    )
    for i, (img, title) in enumerate(zip(imgs, titles), 1):
        if img.ndim == 2:
            rgb = np.stack([img] * 3, axis=-1)
        else:
            rgb = img
        rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        fig.add_trace(go.Image(z=rgb_u8), row=1, col=i)
        fig.update_xaxes(showticklabels=False, row=1, col=i)
        fig.update_yaxes(showticklabels=False, row=1, col=i)

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. MORPHOLOGY FEATURE RADAR
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_radar(features: Dict[str, float]) -> go.Figure:
    """Spider/radar chart of normalised morphology features."""
    keys = [
        "porosity", "fractal_dimension", "image_entropy",
        "circularity_mean", "solidity_mean", "eccentricity_mean",
        "skeleton_fraction",
    ]
    vals = [features.get(k, 0.0) for k in keys]
    # Normalise 0–1 per feature roughly
    norms = {
        "porosity": 1.0, "fractal_dimension": 3.0, "image_entropy": 8.0,
        "circularity_mean": 1.0, "solidity_mean": 1.0,
        "eccentricity_mean": 1.0, "skeleton_fraction": 1.0,
    }
    vals_n = [min(v / norms.get(k, 1.0), 1.0) for k, v in zip(keys, vals)]
    labels = [k.replace("_", " ").title() for k in keys]

    fig = go.Figure(go.Scatterpolar(
        r=vals_n + [vals_n[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(33,150,243,0.25)",
        line=dict(color="#2196F3", width=2),
        name="Feature Profile",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=340,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_graph_on_image(
    image: np.ndarray,
    G: nx.Graph,
    mask: np.ndarray,
) -> go.Figure:
    """Overlay skeleton + Delaunay graph on SEM image."""
    from skimage.morphology import skeletonize

    rgb_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb_u8))

    # Skeleton overlay using scatter
    skel = skeletonize((mask > 0.5).astype(np.uint8))
    ys, xs = np.where(skel)
    if len(xs):
        fig.add_trace(go.Scatter(
            x=xs[::3], y=ys[::3], mode="markers",
            marker=dict(color="rgba(255,100,100,0.6)", size=2),
            name="Skeleton", showlegend=True,
        ))

    # Graph edges
    if len(G) > 0:
        edge_x, edge_y = [], []
        for u, v in G.edges():
            pu = G.nodes[u].get("pos", (0, 0))
            pv = G.nodes[v].get("pos", (0, 0))
            edge_x += [pu[1], pv[1], None]
            edge_y += [pu[0], pv[0], None]
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color="rgba(0,220,255,0.5)", width=1),
            name="Graph Edges", showlegend=True,
        ))
        # Nodes
        node_x = [G.nodes[n]["pos"][1] for n in G.nodes()]
        node_y = [G.nodes[n]["pos"][0] for n in G.nodes()]
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(color="yellow", size=5, line=dict(color="black", width=1)),
            name="Region Centroids", showlegend=True,
        ))

    fig.update_layout(
        height=400,
        xaxis=dict(showticklabels=False, range=[0, image.shape[1]]),
        yaxis=dict(showticklabels=False, range=[image.shape[0], 0], scaleanchor="x"),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=10)),
        title=f"Delaunay Graph — {len(G)} nodes, {G.number_of_edges()} edges",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLUSTERING / EMBEDDING SCATTER
# ─────────────────────────────────────────────────────────────────────────────

def plot_embedding_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    categories: List[str],
    cluster_labels: Optional[np.ndarray] = None,
    method: str = "UMAP",
    title: str = "",
) -> go.Figure:
    df = pd.DataFrame({
        f"{method}-1": coords_2d[:, 0],
        f"{method}-2": coords_2d[:, 1],
        "Category"   : categories,
        "Cluster"    : [str(l) for l in (cluster_labels if cluster_labels is not None else labels)],
    })
    color_col = "Category"
    color_map = _discrete_color_map(categories)

    fig = px.scatter(
        df,
        x=f"{method}-1", y=f"{method}-2",
        color=color_col,
        color_discrete_map=color_map,
        symbol="Cluster",
        hover_data=["Category", "Cluster"],
        title=title or f"{method} Projection",
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,20,1)",
        xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
        legend=dict(font=dict(size=10)),
    )
    return fig


def plot_elbow_silhouette(
    k_range: range,
    inertias: List[float],
    silhouettes: List[float],
    dbi: List[float],
    optimal_k: int,
) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Elbow (Inertia)", "Silhouette Score ↑", "Davies-Bouldin ↓"],
    )
    ks = list(k_range)
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                             line=dict(color="#2196F3", width=2),
                             marker=dict(size=7), name="Inertia"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=silhouettes, mode="lines+markers",
                             line=dict(color="#4CAF50", width=2),
                             marker=dict(size=7), name="Silhouette"), row=1, col=2)
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_trace(go.Scatter(x=ks, y=dbi, mode="lines+markers",
                             line=dict(color="#FF9800", width=2),
                             marker=dict(size=7), name="DBI"), row=1, col=3)
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", row=1, col=3)
    fig.update_layout(
        height=320, showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_map(
    coords_2d: np.ndarray,
    iso_labels: np.ndarray,
    maha_scores: np.ndarray,
    combined: np.ndarray,
    categories: List[str],
    method: str = "UMAP",
) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Isolation Forest",
            "Mahalanobis Distance",
            "Combined Anomalies",
        ],
        horizontal_spacing=0.06,
    )
    x, y = coords_2d[:, 0], coords_2d[:, 1]

    # Panel 1 – ISO
    colors_iso = ["#F44336" if l == -1 else "#90CAF9" for l in iso_labels]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=colors_iso, size=6, opacity=0.7),
        text=categories, hovertemplate="%{text}<extra></extra>",
        name="ISO",
    ), row=1, col=1)

    # Panel 2 – Mahalanobis heat
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(
            color=maha_scores, colorscale="Hot_r",
            size=6, opacity=0.75,
            colorbar=dict(title="Maha dist", x=0.62, len=0.9, thickness=12),
        ),
        text=categories, hovertemplate="%{text}<extra></extra>",
        name="Maha",
    ), row=1, col=2)

    # Panel 3 – Combined
    colors_comb = []
    for i in range(len(combined)):
        if combined[i]:
            colors_comb.append("#FF1744")
        elif iso_labels[i] == -1:
            colors_comb.append("#FF9800")
        else:
            colors_comb.append("#B0BEC5")
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=colors_comb, size=6, opacity=0.75),
        text=categories, hovertemplate="%{text}<extra></extra>",
        name="Combined",
    ), row=1, col=3)

    fig.update_layout(
        height=380, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,15,1)",
    )
    for c in [1, 2, 3]:
        fig.update_xaxes(showticklabels=False, gridcolor="#1a1a2e", row=1, col=c)
        fig.update_yaxes(showticklabels=False, gridcolor="#1a1a2e", row=1, col=c)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame, features: List[str]) -> go.Figure:
    n = len(features)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features,
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    cats = sorted(df["category"].unique())
    palette = [_cat_color(c) for c in cats]

    for idx, feat in enumerate(features):
        r, c = divmod(idx, cols)
        for ci, (cat, col) in enumerate(zip(cats, palette)):
            vals = df[df["category"] == cat][feat].dropna().tolist()
            fig.add_trace(go.Box(
                y=vals, name=cat,
                marker_color=col, showlegend=(idx == 0),
                boxmean="sd",
            ), row=r + 1, col=c + 1)

    fig.update_layout(
        height=220 * rows,
        boxmode="group",
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=9), orientation="h", y=-0.05),
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: List[str]) -> go.Figure:
    corr = df[features].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=420,
        margin=dict(l=80, r=20, t=40, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_pca_loadings(df: pd.DataFrame, features: List[str]) -> go.Figure:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(df[features].fillna(0))
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    df_l = pd.DataFrame(loadings, columns=["PC1", "PC2"], index=features)
    fig = go.Figure(go.Heatmap(
        z=df_l.values,
        x=["PC1", "PC2"],
        y=df_l.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(df_l.values, 3),
        texttemplate="%{text}",
        colorbar=dict(title="Loading"),
    ))
    ev = pca.explained_variance_ratio_
    fig.update_layout(
        title=f"PCA Loadings  (PC1={ev[0]:.1%}, PC2={ev[1]:.1%})",
        height=340,
        margin=dict(l=100, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_anova_bar(anova_df: pd.DataFrame) -> go.Figure:
    df = anova_df.sort_values("f_statistic", ascending=True)
    colors = ["#F44336" if p < 0.05 else "#78909C" for p in df["p_value"]]
    fig = go.Figure(go.Bar(
        x=df["f_statistic"], y=df["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"p={p:.4f}" for p in df["p_value"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="ANOVA F-Statistics (red = significant p<0.05)",
        height=320,
        xaxis_title="F-statistic",
        margin=dict(l=120, r=80, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_category_similarity_heatmap(
    sim_matrix: np.ndarray, cat_order: List[str]
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=sim_matrix,
        x=cat_order, y=cat_order,
        colorscale="Viridis",
        zmin=0, zmax=1,
        text=np.round(sim_matrix, 3),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Cosine sim"),
    ))
    fig.update_layout(
        title="Mean Cosine Similarity Between Categories",
        height=440,
        margin=dict(l=120, r=20, t=40, b=120),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=45),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. SIMILARITY SEARCH RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_similarity_results(
    query_img: np.ndarray,
    query_cat: str,
    nn_imgs: List[np.ndarray],
    nn_cats: List[str],
    nn_sims: List[float],
) -> go.Figure:
    n = len(nn_imgs) + 1
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=["QUERY"] + [f"NN-{i+1}  {s:.3f}" for i, s in enumerate(nn_sims)],
        horizontal_spacing=0.02,
    )
    all_imgs = [query_img] + nn_imgs
    for i, img in enumerate(all_imgs, 1):
        rgb = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        fig.add_trace(go.Image(z=rgb), row=1, col=i)
        fig.update_xaxes(showticklabels=False, row=1, col=i)
        fig.update_yaxes(showticklabels=False, row=1, col=i)
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. CATEGORY DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_category_distribution(counts: Dict[str, int]) -> go.Figure:
    cats = sorted(counts.keys())
    vals = [counts[c] for c in cats]
    colors = [_cat_color(c) for c in cats]

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "bar"}, {"type": "pie"}]])
    fig.add_trace(go.Bar(
        x=cats, y=vals, marker_color=colors,
        text=vals, textposition="outside", name="Count",
    ), row=1, col=1)
    fig.add_trace(go.Pie(
        labels=cats, values=vals,
        marker=dict(colors=colors),
        hole=0.35, name="",
    ), row=1, col=2)
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=30, b=60),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=45),
    )
    return fig
