"""
app.py – SEM Image Analysis Dashboard
Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
import os
import gc
import warnings
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as PILImage
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ── Local imports ──────────────────────────────────────────────────────────
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT,
    KMEANS_K_RANGE, KMEANS_N_INIT, HDBSCAN_MIN_SIZE, HDBSCAN_MIN_SAMP,
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST, TSNE_PERPLEXITY, TSNE_N_ITER,
    ISO_CONTAMINATION, ISO_N_ESTIMATORS, MAHA_ALPHA,
    MORPH_FEATURE_COLS, DEVICE,
)
from models.model_manager import load_backbone, get_backbone_info
from utils.feature_extractor import (
    preprocess_image, generate_segmentation_mask,
    build_overlay, build_contour_image,
    extract_morphology_features, extract_graph_features,
    extract_deep_embeddings, extract_simclr_embeddings,
    cosine_similarity_search, MORPH_COLS_HYBRID,
)
from utils.visualization import (
    plot_preprocessing_pipeline, plot_feature_radar,
    plot_graph_on_image, plot_embedding_scatter,
    plot_elbow_silhouette, plot_anomaly_map,
    plot_boxplots, plot_correlation_heatmap,
    plot_pca_loadings, plot_anova_bar,
    plot_category_similarity_heatmap, plot_similarity_results,
    plot_category_distribution,
)
from utils.database import SEMDatabase

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base palette ── */
:root {
  --bg-deep:    #0a0c14;
  --bg-card:    #111827;
  --bg-widget:  #1a2035;
  --accent:     #2196F3;
  --accent2:    #00BCD4;
  --accent3:    #4CAF50;
  --warn:       #FF9800;
  --danger:     #F44336;
  --text-main:  #E8EAF6;
  --text-dim:   #90A4AE;
  --border:     #1E3A5F;
}

/* ── App background ── */
.stApp { background-color: var(--bg-deep); }
[data-testid="stSidebar"] { background-color: var(--bg-card); border-right: 1px solid var(--border); }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--bg-widget);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 16px;
}

/* ── Tab styling ── */
[data-baseweb="tab-list"] { gap: 4px; background: var(--bg-card); border-radius: 8px; padding: 4px; }
[data-baseweb="tab"] {
  border-radius: 6px;
  color: var(--text-dim) !important;
  font-weight: 500;
}
[aria-selected="true"][data-baseweb="tab"] {
  background: var(--accent) !important;
  color: white !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  padding: 0.4rem 1.2rem;
  transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
  background: var(--bg-widget);
  border: 2px dashed var(--border);
  border-radius: 10px;
}

/* ── Progress bar ── */
.stProgress > div > div { background-color: var(--accent); }

/* ── Selectbox / slider ── */
[data-baseweb="select"] > div { background: var(--bg-widget); border-color: var(--border); }
[data-testid="stSlider"] .stSlider { accent-color: var(--accent); }

/* ── Divider ── */
hr { border-color: var(--border); }

/* ── Custom badges ── */
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  margin-right: 4px;
}
.badge-blue  { background: #1565C0; color: #90CAF9; }
.badge-green { background: #1B5E20; color: #A5D6A7; }
.badge-red   { background: #B71C1C; color: #FFCDD2; }
.badge-orange{ background: #E65100; color: #FFE0B2; }

/* ── Section headers ── */
.section-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--accent2);
  letter-spacing: 0.04em;
  border-left: 4px solid var(--accent);
  padding-left: 10px;
  margin: 18px 0 10px 0;
}

/* ── Feature table ── */
.feat-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.feat-table th { background: var(--bg-widget); color: var(--accent2);
                 padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
.feat-table td { padding: 5px 10px; color: var(--text-main); border-bottom: 1px solid #1a1a2e; }
.feat-table tr:hover td { background: var(--bg-widget); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading backbone model…")
def _get_backbone(prefer_dino: bool):
    return load_backbone(prefer_dino=prefer_dino)


@st.cache_resource
def _get_db() -> SEMDatabase:
    return SEMDatabase()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/"
        "24701-nature-natural-beauty.jpg/320px-24701-nature-natural-beauty.jpg",
        use_column_width=True,
    ) if False else None  # placeholder logo disabled

    st.sidebar.markdown(f"""
    <div style="text-align:center;padding:8px 0 16px 0;">
      <span style="font-size:2rem;">🔬</span><br>
      <span style="font-size:1.2rem;font-weight:700;color:#2196F3;">SEM Analyser</span><br>
      <span style="font-size:0.75rem;color:#78909C;">Nanoscale Feature Intelligence</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.divider()

    # ── Model ──
    st.sidebar.markdown("### 🤖 Model")
    prefer_dino = st.sidebar.toggle("Try DINOv2 (needs internet)", value=False)

    # ── Clustering ──
    st.sidebar.markdown("### 📊 Clustering")
    max_k = st.sidebar.slider("Max K (K-Means search)", 3, 15, KMEANS_K_RANGE[1])
    use_hdbscan = st.sidebar.toggle("Use HDBSCAN", value=True)

    # ── UMAP ──
    st.sidebar.markdown("### 🗺️ Embedding")
    emb_method = st.sidebar.radio("Dim-reduction", ["UMAP", "t-SNE"], horizontal=True)
    n_neighbors = st.sidebar.slider("UMAP n_neighbors", 5, 50, UMAP_N_NEIGHBORS)
    min_dist    = st.sidebar.slider("UMAP min_dist", 0.0, 0.9, UMAP_MIN_DIST, step=0.05)

    # ── Anomaly ──
    st.sidebar.markdown("### 🔍 Anomaly Detection")
    contamination = st.sidebar.slider("ISO contamination", 0.01, 0.20, ISO_CONTAMINATION, step=0.01)

    # ── DB ──
    st.sidebar.markdown("### 🗄️ Database")
    db = _get_db()
    n_db = db.count()
    st.sidebar.metric("Images in DB", n_db)

    if st.sidebar.button("🗑️ Clear Database", use_container_width=True):
        db.clear()
        st.sidebar.success("Database cleared.")
        st.rerun()

    return {
        "prefer_dino"  : prefer_dino,
        "max_k"        : max_k,
        "use_hdbscan"  : use_hdbscan,
        "emb_method"   : emb_method,
        "n_neighbors"  : n_neighbors,
        "min_dist"     : min_dist,
        "contamination": contamination,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bytes_to_np(file_bytes: bytes) -> Optional[np.ndarray]:
    """Decode uploaded file bytes to RGB uint8."""
    import cv2
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def np_thumbnail(img: np.ndarray, size: int = 128) -> np.ndarray:
    pil = PILImage.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    pil.thumbnail((size, size))
    return np.array(pil).astype(np.float32) / 255.0


def feature_table_html(feats: Dict[str, float]) -> str:
    rows = ""
    for k, v in feats.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        rows += f"<tr><td>{k.replace('_',' ').title()}</td><td><b>{val}</b></td></tr>"
    return f"""
    <table class="feat-table">
      <thead><tr><th>Feature</th><th>Value</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def analyse_single(
    raw_bytes: bytes,
    filename: str,
    category: str,
    backbone,
    backbone_name: str,
    save_to_db: bool = True,
) -> Dict:
    """Full single-image analysis pipeline. Returns results dict."""
    # 1. Preprocess
    original_np = bytes_to_np(raw_bytes)
    processed   = preprocess_image(raw_bytes)
    if processed is None:
        st.error("Could not decode image.")
        return {}

    mask    = generate_segmentation_mask(processed)
    overlay = build_overlay(processed, mask)
    contour = build_contour_image(processed, mask)

    # 2. Morphology
    morph = extract_morphology_features(processed, mask)

    # 3. Graph
    G, graph_m = extract_graph_features(mask, mode="delaunay")

    # 4. Deep embeddings (single image as list)
    deep_emb  = extract_deep_embeddings([processed], backbone, backbone_name)
    sim_emb   = extract_simclr_embeddings([processed], backbone, backbone_name)

    # 5. Save to DB
    db = _get_db()
    if save_to_db:
        thumb = np_thumbnail(processed)
        db.insert_image(
            filename=filename,
            category=category,
            morph=morph,
            graph=graph_m,
            deep_emb=deep_emb[0],
            simclr_emb=sim_emb[0],
            thumbnail=thumb,
        )

    return {
        "original"  : original_np.astype(np.float32) / 255.0,
        "processed" : processed,
        "mask"      : mask,
        "overlay"   : overlay,
        "contour"   : contour,
        "morph"     : morph,
        "graph"     : graph_m,
        "G"         : G,
        "deep_emb"  : deep_emb[0],
        "simclr_emb": sim_emb[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – SINGLE IMAGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def tab_single_analysis(cfg: Dict, backbone, backbone_name: str):
    st.markdown('<div class="section-title">Single Image Analysis</div>', unsafe_allow_html=True)

    col_up, col_cat = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Drop a SEM image (.jpg .png .tif .tiff)",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
        )
    with col_cat:
        categories = [
            "Unknown", "Biological", "Fibres", "Films_Coated_Surface",
            "MEMS_devices_and_electrodes", "Nanowires", "Particles",
            "Patterned_surface", "Porous_Sponge", "Powder", "Tips",
        ]
        category = st.selectbox("Category", categories)
        save_db  = st.checkbox("Save to database", value=True)

    if uploaded is None:
        st.info("Upload a SEM image to begin analysis.")
        return

    raw_bytes = uploaded.read()

    with st.spinner("Analysing…"):
        t0      = time.time()
        results = analyse_single(
            raw_bytes, uploaded.name, category,
            backbone, backbone_name, save_to_db=save_db
        )
        elapsed = time.time() - t0

    if not results:
        return

    st.success(f"Analysis complete in {elapsed:.1f}s   |   Model: `{backbone_name}`")

    # ── Preprocessing pipeline ──
    st.markdown('<div class="section-title">Preprocessing Pipeline</div>', unsafe_allow_html=True)
    fig_pp = plot_preprocessing_pipeline(
        results["original"], results["processed"],
        results["mask"], results["overlay"], results["contour"]
    )
    st.plotly_chart(fig_pp, use_container_width=True)

    # ── Metrics row ──
    m = results["morph"]
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Pore Count",          f"{int(m.get('pore_count',0))}")
    c2.metric("Porosity",             f"{m.get('porosity',0):.3f}")
    c3.metric("Fractal Dim.",         f"{m.get('fractal_dimension',0):.3f}")
    c4.metric("Mean Circularity",     f"{m.get('circularity_mean',0):.3f}")
    c5.metric("Image Entropy",        f"{m.get('image_entropy',0):.2f}")
    c6.metric("Mean Diameter (px)",   f"{m.get('diameter_mean',0):.1f}")

    # ── Detailed features + radar ──
    col_feat, col_radar = st.columns([1, 1])
    with col_feat:
        st.markdown('<div class="section-title">All Morphology Features</div>', unsafe_allow_html=True)
        st.markdown(feature_table_html(results["morph"]), unsafe_allow_html=True)

    with col_radar:
        st.markdown('<div class="section-title">Feature Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_feature_radar(results["morph"]), use_container_width=True)

    # ── Graph analysis ──
    st.markdown('<div class="section-title">Graph-based Structural Analysis</div>', unsafe_allow_html=True)
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        st.plotly_chart(
            plot_graph_on_image(results["processed"], results["G"], results["mask"]),
            use_container_width=True
        )
    with col_g2:
        st.markdown("**Graph Metrics**")
        st.markdown(feature_table_html(results["graph"]), unsafe_allow_html=True)

    # ── Embedding info ──
    with st.expander("📐 Embedding Vectors"):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Deep ({backbone_name})** shape: `{results['deep_emb'].shape}`")
            st.bar_chart(pd.DataFrame({"value": results["deep_emb"][:64]}))
        with c2:
            st.write(f"**SimCLR** shape: `{results['simclr_emb'].shape}`")
            st.bar_chart(pd.DataFrame({"value": results["simclr_emb"][:64]}))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – BATCH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def tab_batch_analysis(cfg: Dict, backbone, backbone_name: str):
    st.markdown('<div class="section-title">Batch Image Analysis</div>', unsafe_allow_html=True)

    col_up, col_opts = st.columns([3, 1])
    with col_up:
        files = st.file_uploader(
            "Upload multiple SEM images",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            accept_multiple_files=True,
        )
    with col_opts:
        batch_cat = st.selectbox("Default category", [
            "Unknown", "Biological", "Fibres", "Films_Coated_Surface",
            "MEMS_devices_and_electrodes", "Nanowires", "Particles",
            "Patterned_surface", "Porous_Sponge", "Powder", "Tips",
        ], key="batch_cat")
        save_batch = st.checkbox("Save all to database", value=True)

    if not files:
        st.info("Upload one or more SEM images for batch processing.")
        return

    if st.button(f"▶ Run Batch Analysis ({len(files)} images)", use_container_width=True):
        all_morph, all_names = [], []
        prog = st.progress(0, text="Starting…")

        for i, f in enumerate(files):
            prog.progress((i) / len(files), text=f"Processing {f.name}…")
            raw = f.read()
            results = analyse_single(
                raw, f.name, batch_cat, backbone, backbone_name, save_to_db=save_batch
            )
            if results:
                rec = results["morph"].copy()
                rec["filename"] = f.name
                rec["category"] = batch_cat
                all_morph.append(rec)
                all_names.append(f.name)

        prog.progress(1.0, text="Done!")

        if not all_morph:
            st.error("No images could be processed.")
            return

        df = pd.DataFrame(all_morph)
        st.success(f"✅ Processed {len(df)} images.")

        # Show summary table
        st.markdown('<div class="section-title">Batch Results Summary</div>', unsafe_allow_html=True)
        st.dataframe(
            df[[c for c in MORPH_FEATURE_COLS if c in df.columns]
               + ["filename"]].round(4),
            use_container_width=True,
        )

        # Distribution plots
        st.markdown('<div class="section-title">Feature Distributions</div>', unsafe_allow_html=True)
        feat_cols = [c for c in ["pore_count","porosity","fractal_dimension",
                                  "circularity_mean","image_entropy","diameter_mean"]
                     if c in df.columns]
        fig_box = plot_boxplots(df, feat_cols)
        st.plotly_chart(fig_box, use_container_width=True)

        # Correlation
        fig_corr = plot_correlation_heatmap(df, feat_cols)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Export
        st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download CSV", csv_bytes,
            file_name=f"sem_batch_{batch_cat}.csv",
            mime="text/csv",
        )
        json_bytes = df.to_json(orient="records", indent=2).encode()
        st.download_button(
            "⬇ Download JSON", json_bytes,
            file_name=f"sem_batch_{batch_cat}.json",
            mime="application/json",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – DATABASE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def tab_database(cfg: Dict):
    st.markdown('<div class="section-title">Database Management</div>', unsafe_allow_html=True)
    db = _get_db()
    n  = db.count()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Images", n)
    df_meta = db.get_all_metadata()
    if not df_meta.empty:
        c2.metric("Categories", df_meta["category"].nunique())
        c3.metric("Latest Upload", df_meta["upload_time"].max()[:16])

    if n == 0:
        st.info("Database is empty. Analyse some images first.")
        return

    # Category distribution
    st.markdown('<div class="section-title">Category Distribution</div>', unsafe_allow_html=True)
    counts = df_meta["category"].value_counts().to_dict()
    st.plotly_chart(plot_category_distribution(counts), use_container_width=True)

    # Metadata table
    st.markdown('<div class="section-title">Image Registry</div>', unsafe_allow_html=True)
    st.dataframe(df_meta, use_container_width=True)

    # Morphology stats
    st.markdown('<div class="section-title">Category-wise Statistics</div>', unsafe_allow_html=True)
    df_morph = db.get_morph_dataframe()
    feat_cols = [c for c in MORPH_FEATURE_COLS if c in df_morph.columns]
    if feat_cols and "category" in df_morph.columns:
        stats = df_morph.groupby("category")[feat_cols].mean().round(4)
        st.dataframe(stats, use_container_width=True)

        fig_box = plot_boxplots(df_morph, feat_cols[:6])
        st.plotly_chart(fig_box, use_container_width=True)

        # ANOVA
        st.markdown('<div class="section-title">ANOVA — Inter-Category Differences</div>', unsafe_allow_html=True)
        anova_rows = []
        for feat in feat_cols:
            groups = [df_morph[df_morph["category"] == c][feat].dropna().values
                      for c in df_morph["category"].unique()]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                try:
                    f_stat, p_val = f_oneway(*groups)
                    anova_rows.append({"feature": feat, "f_statistic": f_stat, "p_value": p_val})
                except Exception:
                    pass
        if anova_rows:
            df_anova = pd.DataFrame(anova_rows)
            st.plotly_chart(plot_anova_bar(df_anova), use_container_width=True)
            st.dataframe(df_anova.round(5), use_container_width=True)

        # Correlation
        if len(df_morph) > 3:
            fig_corr = plot_correlation_heatmap(df_morph, feat_cols[:8])
            st.plotly_chart(fig_corr, use_container_width=True)
            fig_pca_l = plot_pca_loadings(df_morph, feat_cols[:8])
            st.plotly_chart(fig_pca_l, use_container_width=True)

    # Exports
    st.markdown('<div class="section-title">Export Database</div>', unsafe_allow_html=True)
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        if st.button("⬇ Export Morphology CSV"):
            path = db.export_morphology_csv()
            with open(path, "rb") as fh:
                st.download_button("Download CSV", fh.read(), file_name=os.path.basename(path), mime="text/csv")
    with col_e2:
        if st.button("⬇ Export Embeddings NPZ"):
            path = db.export_embeddings_npz("hybrid")
            with open(path, "rb") as fh:
                st.download_button("Download NPZ", fh.read(), file_name=os.path.basename(path))
    with col_e3:
        if st.button("⬇ Export Full JSON"):
            path = db.export_full_json()
            with open(path, "rb") as fh:
                st.download_button("Download JSON", fh.read(), file_name=os.path.basename(path), mime="application/json")

    # Delete individual
    with st.expander("🗑️ Delete Images"):
        del_id = st.number_input("Image ID to delete", min_value=1, step=1)
        if st.button("Delete"):
            db.delete_image(int(del_id))
            st.success(f"Deleted image ID {del_id}.")
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – CLUSTERING EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

def tab_clustering(cfg: Dict):
    st.markdown('<div class="section-title">Clustering Explorer</div>', unsafe_allow_html=True)

    db = _get_db()
    n  = db.count()
    if n < 4:
        st.warning(f"Need ≥4 images in the database (have {n}). Analyse more images first.")
        return

    emb_kind = st.radio("Feature space", ["hybrid", "deep", "simclr"], horizontal=True)
    vecs, cats, fns = db.get_all_embeddings(kind=emb_kind)

    if vecs.ndim < 2 or len(vecs) < 4:
        st.warning("Not enough embeddings stored yet.")
        return

    # PCA pre-reduction
    n_pca = min(50, vecs.shape[1], len(vecs) - 1)
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(vecs)

    # ── Dimensionality reduction ──
    with st.spinner(f"Running {cfg['emb_method']}…"):
        if cfg["emb_method"] == "UMAP":
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(cfg["n_neighbors"], len(X_pca) - 1),
                    min_dist=cfg["min_dist"],
                    random_state=42,
                )
                X_2d = reducer.fit_transform(X_pca)
                method_label = "UMAP"
            except ImportError:
                st.warning("umap-learn not installed, falling back to t-SNE.")
                X_2d = TSNE(n_components=2, random_state=42,
                            perplexity=min(TSNE_PERPLEXITY, len(X_pca) // 3),
                            n_iter=TSNE_N_ITER).fit_transform(X_pca)
                method_label = "t-SNE"
        else:
            X_2d = TSNE(n_components=2, random_state=42,
                        perplexity=min(TSNE_PERPLEXITY, len(X_pca) // 3),
                        n_iter=TSNE_N_ITER).fit_transform(X_pca)
            method_label = "t-SNE"

    # ── K-Means ──
    max_k    = min(cfg["max_k"], len(X_pca) - 1)
    k_range  = range(2, max_k + 1)
    inertias, silhs, dbis = [], [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=KMEANS_N_INIT)
        lbl = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhs.append(silhouette_score(X_pca, lbl))
        dbis.append(davies_bouldin_score(X_pca, lbl))

    optimal_k = list(k_range)[int(np.argmax(silhs))]
    km_final  = KMeans(n_clusters=optimal_k, random_state=42, n_init=KMEANS_N_INIT)
    km_labels = km_final.fit_predict(X_pca)

    st.markdown('<div class="section-title">Elbow / Silhouette / Davies-Bouldin</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_elbow_silhouette(k_range, inertias, silhs, dbis, optimal_k),
                    use_container_width=True)

    # ── HDBSCAN ──
    hdb_labels = None
    if cfg["use_hdbscan"]:
        try:
            import hdbscan
            hdb = hdbscan.HDBSCAN(
                min_cluster_size=max(HDBSCAN_MIN_SIZE, len(X_pca) // 20),
                min_samples=HDBSCAN_MIN_SAMP,
            )
            hdb_labels = hdb.fit_predict(X_pca)
            n_hdb    = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
            noise_r  = (hdb_labels == -1).sum() / len(hdb_labels)
            st.info(f"HDBSCAN → {n_hdb} clusters, {noise_r:.1%} noise")
        except ImportError:
            st.warning("hdbscan not installed — showing K-Means only.")

    # ── Scatter plots ──
    st.markdown('<div class="section-title">2-D Projections</div>', unsafe_allow_html=True)
    tab_km, tab_hdb, tab_cat = st.tabs(
        [f"K-Means (k={optimal_k})",
         "HDBSCAN" if hdb_labels is not None else "HDBSCAN (unavail.)",
         "By Category"]
    )
    with tab_km:
        st.plotly_chart(
            plot_embedding_scatter(X_2d, km_labels, cats, km_labels,
                                   method=method_label, title=f"{method_label} — K-Means"),
            use_container_width=True,
        )
    with tab_hdb:
        if hdb_labels is not None:
            st.plotly_chart(
                plot_embedding_scatter(X_2d, hdb_labels, cats, hdb_labels,
                                       method=method_label, title=f"{method_label} — HDBSCAN"),
                use_container_width=True,
            )
        else:
            st.info("Install hdbscan for density-based clustering.")
    with tab_cat:
        cat_ids = np.array([hash(c) for c in cats])
        st.plotly_chart(
            plot_embedding_scatter(X_2d, cat_ids, cats, km_labels,
                                   method=method_label, title=f"{method_label} — True Categories"),
            use_container_width=True,
        )

    # ── Quality table ──
    st.markdown("**Clustering Quality**")
    sil_km = silhs[optimal_k - 2]
    dbi_km = dbis[optimal_k - 2]
    rows = [{"Method": f"K-Means (k={optimal_k})", "Silhouette↑": round(sil_km, 4), "DBI↓": round(dbi_km, 4)}]
    if hdb_labels is not None:
        valid = hdb_labels != -1
        if valid.sum() > 1 and len(set(hdb_labels[valid])) > 1:
            rows.append({
                "Method": "HDBSCAN",
                "Silhouette↑": round(silhouette_score(X_pca[valid], hdb_labels[valid]), 4),
                "DBI↓": round(davies_bouldin_score(X_pca[valid], hdb_labels[valid]), 4),
            })
    st.table(pd.DataFrame(rows))

    # ── Anomaly detection ──
    st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)
    iso = IsolationForest(contamination=cfg["contamination"],
                          random_state=42, n_estimators=ISO_N_ESTIMATORS)
    iso_labels  = iso.fit_predict(X_pca)
    iso_scores  = iso.decision_function(X_pca)

    try:
        from scipy.stats import chi2
        mu  = X_pca.mean(axis=0)
        cov_inv = np.linalg.pinv(np.cov(X_pca.T))
        diff    = X_pca - mu
        maha2   = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
        p_vals  = 1.0 - chi2.cdf(maha2, df=X_pca.shape[1])
        maha_a  = p_vals < MAHA_ALPHA
        maha_d  = np.sqrt(np.maximum(maha2, 0))
    except Exception:
        maha_a = np.zeros(len(X_pca), dtype=bool)
        maha_d = np.zeros(len(X_pca))

    combined = (iso_labels == -1) & maha_a
    n_iso    = (iso_labels == -1).sum()
    n_maha   = maha_a.sum()
    n_comb   = combined.sum()

    ca, cb, cc = st.columns(3)
    ca.metric("Isolation Forest", f"{n_iso} anomalies ({n_iso/len(iso_labels):.1%})")
    cb.metric("Mahalanobis (p<0.01)", f"{n_maha} anomalies")
    cc.metric("Combined (both agree)", f"{n_comb} anomalies")

    st.plotly_chart(
        plot_anomaly_map(X_2d, iso_labels, maha_d, combined, cats, method=method_label),
        use_container_width=True,
    )

    # Anomaly CSV export
    df_anom = pd.DataFrame({
        "filename"      : fns,
        "category"      : cats,
        "iso_label"     : iso_labels,
        "maha_distance" : maha_d,
        "combined"      : combined.astype(int),
    })
    st.download_button(
        "⬇ Export Anomaly Report",
        df_anom.to_csv(index=False).encode(),
        file_name="anomaly_report.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 – SIMILARITY SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def tab_similarity(cfg: Dict, backbone, backbone_name: str):
    st.markdown('<div class="section-title">Morphology Fingerprint Similarity Search</div>', unsafe_allow_html=True)

    db = _get_db()
    n  = db.count()
    if n < 2:
        st.warning(f"Need ≥2 images in the database (have {n}).")
        return

    vecs, cats, fns = db.get_all_embeddings(kind="hybrid")
    if vecs.ndim < 2:
        vecs_fallback, cats, fns = db.get_all_embeddings(kind="deep")
        vecs = vecs_fallback
        if vecs.ndim < 2:
            st.error("No embedding vectors found. Re-run analysis.")
            return

    col_q, col_k = st.columns([3, 1])
    with col_k:
        k_val = st.slider("Top-K results", 1, min(10, n - 1), min(5, n - 1))
        mode  = st.radio("Query mode", ["Upload new image", "Select from DB"], horizontal=True)

    if mode == "Upload new image":
        with col_q:
            query_file = st.file_uploader("Query image", type=["jpg","jpeg","png","tif","tiff"], key="sim_q")
        if query_file is None:
            st.info("Upload a query image.")
            return
        raw = query_file.read()
        with st.spinner("Extracting query embedding…"):
            proc = preprocess_image(raw)
            if proc is None:
                st.error("Could not decode image.")
                return
            q_deep  = extract_deep_embeddings([proc], backbone, backbone_name)[0]
            q_sim   = extract_simclr_embeddings([proc], backbone, backbone_name)[0]
            q_morph = extract_morphology_features(proc, generate_segmentation_mask(proc))
            # Build hybrid via PCA on current DB
            n_pca = min(64, vecs.shape[1])
            pca_v = PCA(n_components=n_pca, random_state=42).fit(vecs)
            q_reduced = pca_v.transform(
                normalize(np.hstack([q_deep, q_sim]).reshape(1, -1), norm="l2")
                if vecs.shape[1] == len(q_deep) + len(q_sim)
                else normalize(q_deep.reshape(1, -1), norm="l2")
            )
            # Direct search in embedding space
            q_vec = normalize(q_deep.reshape(1, -1), norm="l2")[0]
            db_vecs_d, _, _ = db.get_all_embeddings(kind="deep")
            if db_vecs_d.ndim < 2:
                db_vecs_d = vecs
            nn_idx, nn_sims = cosine_similarity_search(q_vec, db_vecs_d, k=k_val)

        query_img = proc
        q_cat     = "Query"

    else:  # Select from DB
        with col_q:
            sel = st.selectbox("Select query image", options=list(range(n)),
                               format_func=lambda i: fns[i] if i < len(fns) else str(i))
        q_vec     = normalize(vecs[sel].reshape(1, -1), norm="l2")[0]
        nn_idx, nn_sims = cosine_similarity_search(q_vec, vecs, k=k_val, exclude_idx=sel)
        # Load thumbnail as query image
        thumbs = db.get_thumbnails()
        query_img = thumbs[sel][2] if sel < len(thumbs) and thumbs[sel][2] is not None \
                    else np.zeros((64, 64, 3), dtype=np.float32)
        q_cat = cats[sel] if sel < len(cats) else "?"

    # ── Results ──
    st.markdown('<div class="section-title">Search Results</div>', unsafe_allow_html=True)
    thumbs = db.get_thumbnails()
    nn_imgs = []
    nn_cats = []
    for idx in nn_idx:
        if idx < len(thumbs) and thumbs[idx][2] is not None:
            nn_imgs.append(thumbs[idx][2])
        else:
            nn_imgs.append(np.zeros((64, 64, 3), dtype=np.float32))
        nn_cats.append(cats[idx] if idx < len(cats) else "?")

    st.plotly_chart(
        plot_similarity_results(query_img, q_cat, nn_imgs, nn_cats, nn_sims.tolist()),
        use_container_width=True,
    )

    # Similarity table
    df_res = pd.DataFrame({
        "Rank"      : list(range(1, len(nn_idx) + 1)),
        "Filename"  : [fns[i] if i < len(fns) else "?" for i in nn_idx],
        "Category"  : nn_cats,
        "Similarity": [f"{s:.4f}" for s in nn_sims],
    })
    st.dataframe(df_res, use_container_width=True)

    # ── Category similarity matrix ──
    if n >= 4:
        st.markdown('<div class="section-title">Category-Level Similarity Matrix</div>', unsafe_allow_html=True)
        cat_order  = sorted(set(cats))
        cats_arr   = np.array(cats)
        sim_matrix = np.zeros((len(cat_order), len(cat_order)))
        vecs_norm  = normalize(vecs, norm="l2")
        for i, ci in enumerate(cat_order):
            for j, cj in enumerate(cat_order):
                fi = vecs_norm[cats_arr == ci]
                fj = vecs_norm[cats_arr == cj]
                if len(fi) > 0 and len(fj) > 0:
                    sim_matrix[i, j] = float((fi @ fj.T).mean())
        st.plotly_chart(
            plot_category_similarity_heatmap(sim_matrix, cat_order),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = render_sidebar()

    # Load backbone
    backbone, embed_dim, backbone_name = _get_backbone(cfg["prefer_dino"])

    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
      <span style="font-size:2.4rem;">🔬</span>
      <div>
        <h1 style="margin:0;font-size:1.8rem;font-weight:800;
                   background:linear-gradient(90deg,#2196F3,#00BCD4);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          SEM Image Analysis Dashboard
        </h1>
        <p style="margin:0;color:#78909C;font-size:0.85rem;">
          Nanoscale Feature Intelligence · {backbone_name.upper()} · {embed_dim}-dim embeddings · {str(cfg.get('emb_method','UMAP'))} projection
        </p>
      </div>
    </div>
    <hr style="margin:8px 0 16px 0;">
    """, unsafe_allow_html=True)

    # Model info banner
    with st.expander(f"🤖 Active Model: {backbone_name}", expanded=False):
        st.info(get_backbone_info(backbone_name))

    # Tabs
    tabs = st.tabs([
        "🔬 Single Analysis",
        "📦 Batch Analysis",
        "🗄️ Database",
        "🔵 Clustering Explorer",
        "🔍 Similarity Search",
    ])

    with tabs[0]:
        tab_single_analysis(cfg, backbone, backbone_name)
    with tabs[1]:
        tab_batch_analysis(cfg, backbone, backbone_name)
    with tabs[2]:
        tab_database(cfg)
    with tabs[3]:
        tab_clustering(cfg)
    with tabs[4]:
        tab_similarity(cfg, backbone, backbone_name)

    # Footer
    st.markdown("""
    <div style="text-align:center;color:#37474F;font-size:0.75rem;margin-top:40px;padding:16px;
                border-top:1px solid #1E3A5F;">
      SEM Analysis Dashboard · ResNet-50 / DINOv2 · UMAP · HDBSCAN · Delaunay Graphs
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
