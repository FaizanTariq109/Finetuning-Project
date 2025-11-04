import streamlit as st

st.set_page_config(page_title="🍽️ Food41 Image Classifier", layout="centered")

from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# -----------------------------
# ✅ Load model and processor
# -----------------------------
MODEL_DIR = "./ViT-finetuned"  # same folder as app.py

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return processor, model

processor, model = load_model()

# -----------------------------
# ✅ Streamlit UI
# -----------------------------
st.title("🍽️ Food41 Vision Transformer (ViT) Classifier")
st.write("Upload an image of food and the model will predict its category!")

uploaded_file = st.file_uploader("Upload a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        # Preprocess
        inputs = processor(images=image, return_tensors="pt")

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.nn.functional.softmax(logits, dim=-1)
            topk = torch.topk(preds, 5)

        # Decode top 5 predictions
        labels = [model.config.id2label[i] for i in topk.indices[0].tolist()]
        scores = [f"{s*100:.2f}%" for s in topk.values[0].tolist()]

    st.success("✅ Classification Complete!")
    st.subheader("🍴 Top Predictions:")
    for lbl, score in zip(labels, scores):
        st.write(f"**{lbl}** — {score}")