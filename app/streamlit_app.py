from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from PIL import Image

from src.config import cfg
from src.inference import MetalPredictor
from src.insights import get_texture_insights, get_top_activation_regions

st.set_page_config(
    page_title="Metal Texture Classifier",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {font-size:2.2rem; font-weight:700; color:#1E3A5F; text-align:center;}
    .insight-box {background:#F0F7FF; border-left:4px solid #1E88E5;
                  padding:1rem; border-radius:8px; margin:0.5rem 0;}
    .trait-chip  {display:inline-block; background:#E3F2FD; color:#1565C0;
                  padding:3px 10px; border-radius:12px; margin:3px;
                  font-size:0.85rem; font-weight:500;}
    .metric-card {background:#F8F9FA; border-radius:10px; padding:1rem;
                  text-align:center; border:1px solid #DEE2E6;}
</style>
""", unsafe_allow_html=True)

METAL_COLORS = {
    "aluminum"        : "#607D8B",
    "brass"           : "#FFC107",
    "Copper"          : "#FF5722",
    "Iron"            : "#455A64",
    "Lead"            : "#78909C",
    "Mild Steel"      : "#546E7A",
    "Stainless Steel" : "#B0BEC5",
    "Tool Steel EN 31": "#37474F",
    "Zinc"            : "#90A4AE",
}


@st.cache_resource
def load_predictor(model_path):
    return MetalPredictor(model_path=model_path)


def build_prob_chart(prob_dict):
    classes = list(prob_dict.keys())
    values  = [prob_dict[c] * 100 for c in classes]
    colors  = [METAL_COLORS.get(c, "#90A4AE") for c in classes]
    fig = go.Figure(go.Bar(
        x=values,
        y=classes,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 120], ticksuffix="%", showgrid=False),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#FAFAFA", paper_bgcolor="#FAFAFA",
        margin=dict(l=10, r=50, t=10, b=10),
        height=60 + 45 * len(classes),
        showlegend=False,
    )
    return fig


def build_similarity_matrix(prob_dict):
    classes = list(prob_dict.keys())
    values  = list(prob_dict.values())
    mat     = np.outer(values, values)
    np.fill_diagonal(mat, [v for v in values])
    mat = (mat / mat.max() * 100).round(1)
    fig = ff.create_annotated_heatmap(
        z=mat,
        x=classes,
        y=classes,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title="Feature Support Matrix",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
    )
    return fig


def main():
    st.markdown('<p class="main-header">🔩 Metal Texture Classifier</p>',
                unsafe_allow_html=True)
    st.markdown("<center>Upload a macro surface image to classify the metal and get texture insights</center>",
                unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Settings")
        model_path   = st.text_input("Model path", value=cfg.paths.best_model)
        show_gradcam = st.toggle("Grad-CAM Heatmap", value=True)
        show_matrix  = st.toggle("Feature Support Matrix", value=True)
        show_insights= st.toggle("Texture Insights", value=True)
        alpha        = st.slider("Heatmap blend", 0.2, 0.8, 0.45, step=0.05)
        st.markdown("---")
        st.markdown("**Model:** EfficientNet-B3")
        st.markdown("**Classes:** 9 metals")

    if not Path(model_path).exists():
        st.error("Model not found. Run: python -m src.train")
        st.stop()

    predictor = load_predictor(model_path)

    uploaded = st.file_uploader(
        "Upload metal surface image",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded is None:
        st.info("👆 Upload a metal surface image to begin analysis.")
        return

    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Analyzing metal texture ..."):
        result = predictor.predict(image, return_gradcam=show_gradcam)

    predicted   = result["predicted"]
    confidence  = result["confidence"]
    prob_dict   = result["probabilities"]
    infer_ms    = result["inference_ms"]

    # ── Top metrics ──────────────────────────────────────────────────
    st.markdown("### 🎯 Classification Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Metal", predicted.upper())
    c2.metric("Confidence",      f"{confidence*100:.1f}%")
    c3.metric("Inference Time",  f"{infer_ms:.0f} ms")
    c4.metric("Total Classes",   len(prob_dict))

    st.markdown("---")

    # ── Image + GradCAM ───────────────────────────────────────────────
    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### 📷 Input Image")
        st.image(image, use_column_width=True, caption=uploaded.name)

        if show_gradcam and "gradcam_image" in result:
            st.markdown("#### 🔥 Grad-CAM Attention Map")
            st.image(result["gradcam_image"],
                     use_column_width=True,
                     caption="Red = high texture influence")

            heatmap = result.get("raw_heatmap")
            if heatmap is not None:
                regions = get_top_activation_regions(heatmap)
                st.markdown("**Top activated regions:**")
                for r in regions:
                    st.markdown(
                        f"- `{r['region']}` — activation score: **{r['activation']:.3f}**")

    with col_results:
        st.markdown("#### 📊 Class Probabilities")
        st.plotly_chart(build_prob_chart(prob_dict), use_container_width=True)

        # ── Texture Insights ─────────────────────────────────────────
        if show_insights:
            st.markdown("#### 💡 Texture Insights")
            insight_data = get_texture_insights(predicted, confidence)
            st.markdown(
                f'<div class="insight-box">{insight_data["insight"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Supporting texture traits detected:**")
            chips = "".join(
                [f'<span class="trait-chip">✓ {t}</span>'
                 for t in insight_data["traits"]]
            )
            st.markdown(chips, unsafe_allow_html=True)

        # ── Raw probabilities ─────────────────────────────────────────
        with st.expander("📋 Full probability breakdown"):
            for cls, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
                st.text(f"  {cls:<20} {bar}  {prob*100:.2f}%")

    # ── Feature Support Matrix ────────────────────────────────────────
    if show_matrix:
        st.markdown("---")
        st.markdown("### 🧮 Feature Support Matrix")
        st.caption(
            "This matrix shows how model confidence scores relate across all metal classes. "
            "Brighter cells indicate stronger feature overlap or contrast between metals."
        )
        st.plotly_chart(build_similarity_matrix(prob_dict),
                        use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<center><small>Metal Texture Classifier · EfficientNet-B3 · PyTorch · 9 Classes</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
