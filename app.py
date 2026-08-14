import os
import sys
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from predict import load_model, preprocess_slice, predict as run_inference, CLASS_NAMES
from metrics import dice_coefficient, iou_score, pixel_accuracy


st.set_page_config(page_title="NeuroSeg", page_icon="🧠", layout="wide")

REGION_NAMES = ["Background", "NCR / NET", "Edema", "Enhancing Tumor"]
REGION_COLORS = ["#0B0F14", "#FF6B6B", "#FFD166", "#A78BFA"]

SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "last_prediction.png")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@st.cache_resource
def get_model():
    return load_model()


def show_image(image, title, overlay=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    if overlay is not None:
        cmap = mcolors.ListedColormap(REGION_COLORS)
        ax.imshow(overlay, cmap=cmap, alpha=0.5, vmin=0, vmax=3)
    ax.set_title(title)
    ax.axis("off")
    return fig


# Sidebar

with st.sidebar:
    st.title("🧠 NeuroSeg")
    st.caption("Brain MRI Tumor Segmentation")
    st.divider()

    model = get_model()
    parameters = sum(p.numel() for p in model.parameters())

    st.subheader("Model")
    st.write("Architecture: U-Net")
    st.write(f"Parameters: {parameters / 1e6:.2f}M")
    st.write("Input: 4 MRI modalities")
    st.write("Output: 4 classes")
    st.divider()

    st.subheader("Tumor regions")
    st.write("🔴 NCR / NET")
    st.write("🟡 Edema")
    st.write("🟣 Enhancing Tumor")
    st.divider()

    st.caption("Dataset: BraTS2020")
    st.caption("Input: .h5")
    st.caption("Original size: 240 × 240 × 4")
    st.caption("Model input: 128 × 128")


# Main

st.title("Brain Tumor Segmentation")
st.write("Upload a BraTS MRI slice or try one of the validation samples.")
st.divider()

input_mode = st.radio("Choose input", ["Upload .h5", "Try Sample"], horizontal=True)

selected_file = None

if input_mode == "Try Sample":
    all_samples = {
        "Sample 1": "sample_001.h5",
        "Sample 2": "sample_002.h5",
        "Sample 3": "sample_003.h5",
    }
    sample_files = {
        name: filename for name, filename in all_samples.items()
        if os.path.exists(os.path.join(SAMPLE_DIR, filename))
    }
    if not sample_files:
        st.error(f"No sample files found in '{SAMPLE_DIR}'.")
        st.stop()

    selected_name = st.selectbox("Validation sample", list(sample_files.keys()))
    selected_file = os.path.join(SAMPLE_DIR, sample_files[selected_name])
else:
    uploaded = st.file_uploader("Upload MRI slice", type=["h5"])
    if uploaded is None:
        st.info("Upload a .h5 BraTS slice containing the image dataset.")
        st.stop()


# Load and predict

try:
    if input_mode == "Try Sample":
        image_tensor, label, image = preprocess_slice(selected_file)
    else:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        image_tensor, label, image = preprocess_slice(tmp_path)

    prediction = run_inference(model, image_tensor)
finally:
    if input_mode == "Upload .h5" and "tmp_path" in locals() and os.path.exists(tmp_path):
        os.remove(tmp_path)


base = image[..., 1]
base = (base - base.min()) / (base.max() - base.min() + 1e-8)


# Results

st.subheader("Segmentation")

if label is not None:
    col1, col2, col3 = st.columns(3)
else:
    col1, col2 = st.columns(2)

with col1:
    st.caption("MRI SLICE")
    fig = show_image(base, "MRI Slice")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with col2:
    st.caption("AI PREDICTION")
    fig = show_image(base, "Predicted Segmentation", prediction)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

if label is not None:
    with col3:
        st.caption("GROUND TRUTH")
        fig = show_image(base, "Ground Truth", label)
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# Metrics

dice = iou = accuracy = None

if label is not None:
    st.subheader("Segmentation Metrics")
    dice, _ = dice_coefficient(torch.from_numpy(prediction), torch.from_numpy(label))
    iou, _ = iou_score(torch.from_numpy(prediction), torch.from_numpy(label))
    accuracy = pixel_accuracy(torch.from_numpy(prediction), torch.from_numpy(label))

    m1, m2, m3 = st.columns(3)
    m1.metric("Dice Coefficient", f"{dice:.3f}")
    m2.metric("IoU", f"{iou:.3f}")
    m3.metric("Pixel Accuracy", f"{accuracy:.3f}")
else:
    st.info("No ground-truth mask found. Metrics are unavailable for this file.")


# Region distribution — native st.progress instead of custom HTML,
# since indented multi-line HTML gets misread by Markdown as a code block.

st.subheader("Tumor Region Distribution")

total_pixels = prediction.size
percentages = {
    class_id: float((prediction == class_id).sum() / total_pixels * 100)
    for class_id in range(1, 4)
}

for class_id in range(1, 4):
    pct = percentages[class_id]
    st.write(f"{REGION_NAMES[class_id]} — {pct:.2f}%")
    st.progress(min(int(round(pct)), 100))

if all(p == 0 for p in percentages.values()):
    st.caption(
        "All regions show 0% — this slice may genuinely contain no tumor, "
        "or `prediction` isn't class-labeled yet (check predict.py does argmax)."
    )


# Logging

log_filename = selected_name if input_mode == "Try Sample" else uploaded.name

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "filename": log_filename,
    "dice": dice,
    "iou": iou,
    "pixel_accuracy": accuracy,
    **{f"{name}_pixels": int((prediction == c).sum()) for c, name in enumerate(CLASS_NAMES)},
}
pd.DataFrame([log_row]).to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

if os.path.exists(LOG_PATH):
    with st.expander("Segmentation history"):
        st.dataframe(pd.read_csv(LOG_PATH))


# Download

st.subheader("Export")

fig = show_image(base, "Predicted Segmentation", prediction)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

with open(OUTPUT_PATH, "rb") as file:
    st.download_button("Download Prediction", file, file_name="prediction.png", mime="image/png")