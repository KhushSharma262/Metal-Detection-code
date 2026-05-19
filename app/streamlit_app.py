import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image
import time
from src.config import cfg
from src.inference import MetalPredictor, CONFIDENCE_THRESHOLD

CLASSES     = ["LM-9", "LM 24", "ADC 12"]
CLASS_COLOR = {"LM-9": "#1A73E8", "LM 24": "#00C6A7", "ADC 12": "#F4A623"}
CLASS_ICON  = {"LM-9": "🔵", "LM 24": "🟢", "ADC 12": "🟡"}
CLASS_INFO  = {
    "LM-9":   "General purpose aluminium casting alloy. Good fluidity and pressure tightness. Widely used in automotive and general engineering castings.",
    "LM 24":  "High pressure die casting aluminium alloy. Excellent castability and good mechanical properties. Common in complex thin-walled components.",
    "ADC 12": "Japanese standard aluminium die casting alloy (equivalent to A383). Excellent die castability, used widely in electronic housings and automotive parts.",
}
CLASS_SPECS = {
    "LM-9":   {"Si": "10-13%", "Cu": "<=0.1%",   "Mg": "0.2-0.6%", "Standard": "BS 1490"},
    "LM 24":  {"Si": "7.5-9.5%","Cu": "3-4%",    "Fe": "<=1.3%",   "Standard": "BS 1490"},
    "ADC 12": {"Si": "9.6-12%", "Cu": "1.5-3.5%","Fe": "<=1.3%",   "Standard": "JIS H5302"},
}

st.set_page_config(page_title="Metal Alloy Classifier", page_icon="🔩", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #F8FAFC; }
.block-container { padding-top: 1.5rem; }
.metric-card {
    background: white; border-radius: 12px; padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 12px;
    border: 1px solid #E8EEF6;
}
.result-badge {
    border-radius: 12px; padding: 20px 24px;
    margin-bottom: 16px; border-left: 6px solid;
}
.bar-wrap {
    background: #EEF2F7; border-radius: 6px;
    height: 12px; overflow: hidden; margin-top: 4px;
}
section[data-testid="stSidebar"] { background: #0A1628; }
section[data-testid="stSidebar"] * { color: #C8D8F0 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model...")
def load_predictor(model_path):
    return MetalPredictor(model_path=model_path)

with st.sidebar:
    st.markdown("## 🔩 Metal Classifier")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    model_path  = st.text_input("Model path", value=cfg.paths.best_model)
    threshold   = st.slider("Confidence threshold", 0.0, 1.0, float(CONFIDENCE_THRESHOLD), 0.05)
    st.markdown("---")
    st.markdown("### 🔘 Display Options")
    show_scores = st.toggle("Class confidence bars", value=True)
    show_info   = st.toggle("Alloy description",     value=True)
    show_specs  = st.toggle("Chemical composition",  value=False)
    st.markdown("---")
    st.markdown("### 📦 Supported Classes")
    for c in CLASSES:
        color = CLASS_COLOR[c]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0">'
            f'<div style="width:12px;height:12px;border-radius:50%;background:{color}"></div>'
            f'<span style="font-size:14px">{c}</span></div>',
            unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p style="font-size:11px;opacity:0.5;text-align:center">Metal Alloy Classifier v2.0<br>EfficientNet-B3</p>', unsafe_allow_html=True)

st.markdown("# 🔩 Metal Alloy Classifier")
st.markdown("Upload a macro surface image to identify: **LM-9** | **LM 24** | **ADC 12**")
st.divider()

if not Path(model_path).exists():
    st.error(f"Model not found at: `{model_path}`")
    st.info("Run training first:  `python -m src.train`")
    st.stop()

predictor = load_predictor(model_path)
uploaded  = st.file_uploader("Upload metal surface image", type=["jpg","jpeg","png","bmp","webp"])

if uploaded is None:
    c1, c2, c3 = st.columns(3)
    for col, cls in zip([c1,c2,c3], CLASSES):
        color = CLASS_COLOR[cls]
        with col:
            st.markdown(
                f'<div class="metric-card" style="text-align:center;border-top:4px solid {color}">'
                f'<div style="font-size:32px">{CLASS_ICON[cls]}</div>'
                f'<div style="font-size:18px;font-weight:700;color:{color};margin:8px 0">{cls}</div>'
                f'<div style="font-size:12px;color:#64748B">{CLASS_INFO[cls][:60]}...</div>'
                f'</div>', unsafe_allow_html=True)
    st.info("👆 Upload an image above to start classification")
    st.stop()

image = Image.open(uploaded).convert("RGB")
with st.spinner("🔍 Analysing metal surface..."):
    t_start = time.time()
    result  = predictor.predict(image)
    elapsed = (time.time() - t_start) * 1000

label  = result["label"]
conf   = result["confidence"]
scores = result["scores"]
color  = CLASS_COLOR.get(label, "#888")
icon   = CLASS_ICON.get(label, "🔩")
above  = conf >= threshold

col_img, col_res = st.columns([1,1], gap="large")

with col_img:
    st.subheader("📷 Input Image")
    st.image(image, use_column_width=True, caption=uploaded.name)
    st.caption(f"Size: {image.width}x{image.height}px  |  Inference: {elapsed:.0f}ms")

with col_res:
    st.subheader("🎯 Prediction Result")
    st.markdown(
        f'<div class="result-badge" style="background:{color}15;border-left-color:{color}">'
        f'<div style="display:flex;align-items:center;gap:12px">'
        f'<span style="font-size:40px">{icon}</span>'
        f'<div>'
        f'<div style="font-size:32px;font-weight:800;color:{color};line-height:1.1">{label}</div>'
        f'<div style="font-size:15px;color:#555;margin-top:4px">'
        f'Confidence: <b style="color:{color}">{conf*100:.1f}%</b>'
        f'&nbsp;|&nbsp;Inference: <b>{elapsed:.0f}ms</b>'
        f'</div></div></div></div>',
        unsafe_allow_html=True)

    if above:
        st.success("✅ High confidence — result is reliable")
    else:
        st.warning(f"⚠️ Confidence {conf*100:.1f}% is below threshold ({threshold*100:.0f}%) — try a clearer image")

    if show_info and label in CLASS_INFO:
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:12px;font-weight:600;color:#64748B;text-transform:uppercase;margin-bottom:6px">About {label}</div>'
            f'<div style="font-size:14px;color:#334155;line-height:1.6">{CLASS_INFO[label]}</div>'
            f'</div>', unsafe_allow_html=True)

    if show_specs and label in CLASS_SPECS:
        specs = CLASS_SPECS[label]
        rows  = "".join(
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #F0F4F8;font-size:13px">'
            f'<span style="color:#64748B">{k}</span>'
            f'<span style="font-weight:600;color:#1E293B">{v}</span></div>'
            for k, v in specs.items())
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:12px;font-weight:600;color:#64748B;text-transform:uppercase;margin-bottom:8px">Chemical Composition — {label}</div>'
            f'{rows}</div>', unsafe_allow_html=True)

    if show_scores:
        st.markdown("**Class Confidence Scores**")
        for cls, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            bar_color = CLASS_COLOR.get(cls, "#888")
            is_top    = cls == label
            st.markdown(
                f'<div style="margin-bottom:12px">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
                f'<span style="font-size:14px;font-weight:{"700" if is_top else "400"};color:{"#1E293B" if is_top else "#64748B"}">'
                f'{CLASS_ICON.get(cls,"")} {cls}{"  ✓" if is_top else ""}</span>'
                f'<span style="font-size:14px;font-weight:700;color:{bar_color}">{score*100:.1f}%</span></div>'
                f'<div class="bar-wrap">'
                f'<div style="width:{max(score*100,0.5):.1f}%;background:{bar_color};height:12px;border-radius:6px"></div>'
                f'</div></div>',
                unsafe_allow_html=True)
