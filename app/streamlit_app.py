import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from src.config import cfg
from src.inference import MetalPredictor, InvalidImageError, CONFIDENCE_THRESHOLD
from src.gradcam import GradCAM, overlay_heatmap
from src.class_info import validate_class_meta, get_meta

st.set_page_config(page_title="Metal Classifier", layout="wide")

st.markdown("""
<style>
.metric-card {background:white;border-radius:12px;padding:16px 20px;
  box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:12px;border:1px solid #E8EEF6;}
.result-badge {border-radius:12px;padding:20px 24px;margin-bottom:16px;border-left:6px solid;}
.bar-wrap {background:#EEF2F7;border-radius:6px;height:12px;overflow:hidden;margin-top:4px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def load_predictor(model_path):
    predictor = MetalPredictor(model_path=model_path)
    validate_class_meta(predictor.class_names)
    return predictor


with st.sidebar:
    st.markdown("## Metal Classifier")
    st.markdown("### Settings")
    model_path = st.text_input("Model path", value=cfg.paths.best_model)
    threshold = st.slider("Score threshold", 0.0, 1.0, float(CONFIDENCE_THRESHOLD), 0.05)
    st.markdown("### Display")
    show_scores = st.toggle("Class scores", value=True)
    show_info = st.toggle("Class description", value=True)
    show_cam = st.toggle("Grad-CAM explanation map", value=True)

st.markdown("# Metal Classifier")

if not Path(model_path).exists():
    st.error(f"Model not found at: `{model_path}`")
    st.info("Train first with: `python -m src.train`")
    st.stop()

try:
    predictor = load_predictor(model_path)
except ValueError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.markdown("### Supported classes")
    for name in predictor.class_names:
        c = get_meta(name)["color"]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
            f'<div style="width:12px;height:12px;border-radius:50%;background:{c}"></div>'
            f'<span style="font-size:14px">{name}</span></div>',
            unsafe_allow_html=True)
    st.caption(f"Model: {predictor.architecture} | {predictor.image_size[1]}x{predictor.image_size[0]} input")

st.markdown("Upload a macro surface image to classify it into one of: "
            + ", ".join(predictor.class_names) + ".")
st.divider()

uploaded = st.file_uploader("Upload metal surface image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded is None:
    st.info("Upload an image above to start.")
    st.stop()

try:
    image = Image.open(uploaded)
    image.load()
    image = image.convert("RGB")
except Exception as e:
    st.error(f"Could not read this file as an image: {e}")
    st.stop()

try:
    with st.spinner("Analysing..."):
        t0 = time.time()
        result = predictor.predict(image)
        tensor = predictor.preprocess(image)
        elapsed = (time.time() - t0) * 1000
except InvalidImageError as e:
    st.error(str(e))
    st.stop()

label = result["label"]
score = result["confidence"]
scores = result["scores"]
meta = get_meta(label)
color = meta["color"]

col_img, col_res = st.columns(2, gap="large")

with col_img:
    st.subheader("Input image")
    st.image(image, use_container_width=True, caption=uploaded.name)
    st.caption(f"Size: {image.width}x{image.height}px | Inference + preprocess: {elapsed:.0f} ms")

with col_res:
    st.subheader("Prediction")
    st.markdown(
        f'<div class="result-badge" style="background:{color}15;border-left-color:{color}">'
        f'<div style="font-size:32px;font-weight:800;color:{color};line-height:1.1">{label}</div>'
        f'<div style="font-size:15px;color:#555;margin-top:4px">'
        f'Model score: <b style="color:{color}">{score*100:.1f}%</b></div></div>',
        unsafe_allow_html=True)

    if score >= threshold:
        st.success("Score is above the threshold.")
    else:
        st.warning(f"Score {score*100:.1f}% is below the threshold ({threshold*100:.0f}%). "
                   "The model is unsure; try a clearer image.")
    st.caption("Scores are softmax outputs from a model trained with label smoothing, "
               "so even a certain prediction tops out near 91%. They are not calibrated "
               "probabilities of being correct.")

    if show_info and meta["description"]:
        st.markdown(
            f'<div class="metric-card"><div style="font-size:12px;font-weight:600;'
            f'color:#64748B;text-transform:uppercase;margin-bottom:6px">About {label}</div>'
            f'<div style="font-size:14px;color:#334155;line-height:1.6">{meta["description"]}</div></div>',
            unsafe_allow_html=True)

    if show_scores:
        st.markdown("**Class scores**")
        for cls, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            bc = get_meta(cls)["color"]
            is_top = cls == label
            st.markdown(
                f'<div style="margin-bottom:10px">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                f'<span style="font-size:14px;font-weight:{"700" if is_top else "400"}">{cls}</span>'
                f'<span style="font-size:14px;font-weight:700;color:{bc}">{s*100:.1f}%</span></div>'
                f'<div class="bar-wrap"><div style="width:{max(s*100,0.5):.1f}%;background:{bc};'
                f'height:12px;border-radius:6px"></div></div></div>',
                unsafe_allow_html=True)

if show_cam:
    st.divider()
    st.subheader("Grad-CAM explanation map")
    try:
        with GradCAM(predictor.model) as g:
            heatmap = g.generate(tensor, class_idx=result["index"])
            cam_empty = g.last_cam_empty
        if cam_empty:
            st.warning("Grad-CAM produced an empty map for this image, so no overlay is shown.")
        else:
            seen = image.resize((predictor.image_size[1], predictor.image_size[0]))
            overlay = overlay_heatmap(seen, heatmap)
            c1, c2 = st.columns(2, gap="large")
            c1.image(seen, use_container_width=True, caption="Image as seen by the model")
            c2.image(overlay, use_container_width=True, caption=f"Grad-CAM for: {label}")
        st.caption(
            "This is a visual explanation map, not proof of what the model relies on. "
            "Each map is normalised per image, so a bright spot always appears. "
            "On the validation set, map peaks concentrated near the top edge, but a simple "
            "band-masking test changed few predictions, so the reason is unresolved.")
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")
