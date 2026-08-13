import os
import sys
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from predict import load_model, preprocess_slice, predict as run_inference, CLASS_NAMES
from metrics import dice_coefficient, iou_score, pixel_accuracy


LOG_PATH = "outputs/logs.csv"
os.makedirs("outputs", exist_ok=True)


# -----------------------------
# Theme
# -----------------------------

BG = "#0B0F14"
PANEL = "#11171E"
BORDER = "#26313B"
TEXT = "#E8EDF2"
MUTED = "#8C99A6"

TEAL = "#2FD3C6"
RED = "#FF6B6B"
YELLOW = "#FFD166"
PURPLE = "#A78BFA"

REGION_COLORS = [BG, RED, YELLOW, PURPLE]
REGION_NAMES = ["Background", "NCR / NET", "Edema", "Enhancing Tumor"]


# -----------------------------
# Page
# -----------------------------

st.set_page_config(
    page_title="NeuroSeg",
    page_icon="🧠",
    layout="wide",
)


# -----------------------------
# CSS
# -----------------------------

st.markdown(
    f"""
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap'
);

html, body, [class*="css"] {{
    font-family: "Inter", sans-serif;
}}

.stApp {{
    background: {BG};
    color: {TEXT};
}}

.block-container {{
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}}

header {{
    background: transparent !important;
}}

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}


/* Typography */

h1, h2, h3, h4 {{
    font-family: "Space Grotesk", sans-serif !important;
    color: {TEXT} !important;
}}

p, span, label {{
    color: {TEXT};
}}


/* Sidebar */

section[data-testid="stSidebar"] {{
    background: #0D1218;
    border-right: 1px solid {BORDER};
}}

.sidebar-title {{
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: {TEXT};
}}

.sidebar-subtitle {{
    color: {MUTED};
    font-size: 0.78rem;
    line-height: 1.5;
    margin-top: 0.25rem;
    margin-bottom: 2rem;
}}

.sidebar-section {{
    color: {TEAL};
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 1.7rem 0 0.8rem 0;
}}

.sidebar-line {{
    border-top: 1px solid {BORDER};
    margin: 1.3rem 0;
}}

.sidebar-item {{
    display: flex;
    align-items: center;
    gap: 9px;
    color: {MUTED};
    font-size: 0.8rem;
    margin: 0.65rem 0;
}}

.dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}}


/* Hero */

.hero {{
    border-bottom: 1px solid {BORDER};
    padding: 1rem 0 2.2rem 0;
    margin-bottom: 2.5rem;
}}

.eyebrow {{
    color: {TEAL};
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}}

.hero-title {{
    font-family: "Space Grotesk", sans-serif;
    font-size: 3.2rem;
    line-height: 1.05;
    font-weight: 700;
    color: {TEXT};
    margin-bottom: 0.8rem;
}}

.hero-description {{
    color: {MUTED};
    max-width: 700px;
    font-size: 0.95rem;
    line-height: 1.7;
}}


/* Sections */

.section {{
    margin: 2.2rem 0 1rem 0;
}}

.section-number {{
    color: {TEAL};
    font-family: monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
}}

.section-title {{
    font-family: "Space Grotesk", sans-serif;
    color: {TEXT};
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 0.15rem;
}}


/* Upload */

div[data-testid="stFileUploaderDropzone"] {{
    background: {PANEL};
    border: 1px dashed #35424E;
    border-radius: 2px;
    min-height: 150px;
}}

div[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {TEAL};
}}


/* Image containers */

.image-label {{
    color: {MUTED};
    font-family: monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}}


/* Metrics */

.metric-box {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 2px;
    padding: 1.1rem 1.2rem;
}}

.metric-value {{
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: {TEXT};
}}

.metric-label {{
    color: {MUTED};
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}}


/* Region */

.region-box {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 2px;
    padding: 1rem 1.2rem;
}}

.region-row {{
    display: grid;
    grid-template-columns: 170px 1fr 55px;
    align-items: center;
    gap: 15px;
    margin: 0.9rem 0;
}}

.region-name {{
    color: {TEXT};
    font-size: 0.8rem;
}}

.region-bar {{
    height: 7px;
    background: #202932;
}}

.region-fill {{
    height: 100%;
}}

.region-value {{
    color: {MUTED};
    text-align: right;
    font-family: monospace;
    font-size: 0.75rem;
}}


/* Buttons */

.stDownloadButton > button {{
    background: {TEAL};
    color: {BG};
    border: none;
    border-radius: 2px;
    font-weight: 600;
}}

.stDownloadButton > button:hover {{
    background: #45E0D4;
    color: {BG};
}}


/* Alerts */

div[data-testid="stAlert"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 2px;
}}


/* Empty state */

.empty {{
    border: 1px dashed {BORDER};
    padding: 3.5rem 2rem;
    text-align: center;
    background: rgba(17, 23, 30, 0.35);
}}

.empty-title {{
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.15rem;
    color: {TEXT};
}}

.empty-text {{
    color: {MUTED};
    font-size: 0.82rem;
    margin-top: 0.5rem;
}}

</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------

def section_header(number, title):
    st.markdown(
        f"""
        <div class="section">
            <div class="section-number">{number}</div>
            <div class="section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_panel(figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    ax.axis("off")
    return fig, ax


@st.cache_resource
def get_model():
    return load_model()


# -----------------------------
# Model
# -----------------------------

model = get_model()

n_params = sum(p.numel() for p in model.parameters())


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.markdown('<div class="sidebar-title">NeuroSeg</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-subtitle">
            Brain MRI segmentation using a lightweight U-Net model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sidebar-item">Architecture: U-Net</div>
        <div class="sidebar-item">Parameters: {n_params / 1e6:.2f}M</div>
        <div class="sidebar-item">Input: 4 MRI modalities</div>
        <div class="sidebar-item">Output: 4 classes</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-line"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Tumor regions</div>', unsafe_allow_html=True)

    for name, color in zip(REGION_NAMES[1:], REGION_COLORS[1:]):
        st.markdown(
            f"""
            <div class="sidebar-item">
                <span class="dot" style="background:{color}"></span>
                {name}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-line"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-item">Dataset: BraTS2020</div>
        <div class="sidebar-item">Resolution: 128 × 128</div>
        <div class="sidebar-item">Modality: MRI</div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Hero
# -----------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">MRI ANALYSIS · BraTS U-NET</div>
        <div class="hero-title">Brain Tumor Segmentation</div>
        <div class="hero-description">
            Segment tumor sub-regions from multi-modal brain MRI slices.
            Upload a BraTS-compatible slice to generate an AI prediction
            and compare it with the ground-truth mask.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Upload
# -----------------------------

section_header("01", "Upload MRI slice")

uploaded = st.file_uploader("Upload a BraTS .h5 slice", type=["h5"])


# -----------------------------
# Empty state
# -----------------------------

if uploaded is None:

    st.markdown(
        """
        <div class="empty">
            <div class="empty-title">No MRI slice selected</div>
            <div class="empty-text">
                Upload a .h5 file containing an <code>image</code> dataset to begin.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Prediction
# -----------------------------

else:

    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        img_tensor, label, image_r = preprocess_slice(tmp_path)
        pred = run_inference(model, img_tensor)

    except Exception:
        st.error(
            "Unable to read this file. "
            "Please upload a BraTS-compatible .h5 slice containing an 'image' dataset."
        )
        st.stop()

    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    base = image_r[..., 1]
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    cmap = mcolors.ListedColormap(REGION_COLORS)

    section_header("02", "Segmentation")

    if label is not None:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)

    fig1, ax1 = styled_panel()
    ax1.imshow(base, cmap="gray")
    ax1.set_title("MRI SLICE", color=TEXT, fontsize=10, fontweight="bold", pad=10)
    col1.pyplot(fig1, width="stretch")

    fig2, ax2 = styled_panel()
    ax2.imshow(base, cmap="gray")
    ax2.imshow(pred, cmap=cmap, alpha=0.52, vmin=0, vmax=3)
    ax2.set_title("AI PREDICTION", color=TEAL, fontsize=10, fontweight="bold", pad=10)
    col2.pyplot(fig2, width="stretch")

    if label is not None:
        fig3, ax3 = styled_panel()
        ax3.imshow(base, cmap="gray")
        ax3.imshow(label, cmap=cmap, alpha=0.52, vmin=0, vmax=3)
        ax3.set_title("GROUND TRUTH", color=TEXT, fontsize=10, fontweight="bold", pad=10)
        col3.pyplot(fig3, width="stretch")

    # export happens before we close the figure, so the saved PNG isn't blank
    fig2.savefig("outputs/last_prediction.png", dpi=150, bbox_inches="tight", facecolor=PANEL)

    plt.close(fig1)
    plt.close(fig2)
    if label is not None:
        plt.close(fig3)

    section_header("03", "Segmentation metrics")

    dice_val = None
    iou_val = None
    acc_val = None

    if label is not None:
        dice_val, _ = dice_coefficient(torch.from_numpy(pred), torch.from_numpy(label))
        iou_val, _ = iou_score(torch.from_numpy(pred), torch.from_numpy(label))
        acc_val = pixel_accuracy(torch.from_numpy(pred), torch.from_numpy(label))

        m1, m2, m3 = st.columns(3)
        values = [
            (m1, dice_val, "Dice coefficient"),
            (m2, iou_val, "IoU"),
            (m3, acc_val, "Pixel accuracy"),
        ]
        for col, value, name in values:
            col.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-value">{value:.3f}</div>
                    <div class="metric-label">{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info(
            "No ground-truth mask was found. "
            "Prediction is available, but metrics cannot be calculated."
        )

    section_header("04", "Tumor region distribution")

    total_px = pred.size
    rows_html = ""
    for c in range(1, 4):
        pct = (pred == c).sum() / total_px * 100
        rows_html += f"""
        <div class="region-row">
            <div class="region-name">{REGION_NAMES[c]}</div>
            <div class="region-bar">
                <div class="region-fill" style="width:{min(pct, 100):.2f}%; background:{REGION_COLORS[c]};"></div>
            </div>
            <div class="region-value">{pct:.1f}%</div>
        </div>
        """
    st.markdown(f'<div class="region-box">{rows_html}</div>', unsafe_allow_html=True)

    section_header("05", "Export result")

    with open("outputs/last_prediction.png", "rb") as f:
        st.download_button(
            "Download prediction",
            f,
            file_name=f"{uploaded.name}_prediction.png",
            mime="image/png",
        )

    log_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": uploaded.name,
        "dice": dice_val,
        "iou": iou_val,
        "pixel_accuracy": acc_val,
        **{f"{name}_pixels": int((pred == c).sum()) for c, name in enumerate(CLASS_NAMES)},
    }
    pd.DataFrame([log_row]).to_csv(
        LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False
    )


if os.path.exists(LOG_PATH):
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Segmentation history"):
        st.dataframe(pd.read_csv(LOG_PATH), width="stretch", hide_index=True)