# app.py
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
MODEL_DIR = "./t5-small-finetuned"  # path to your downloaded folder

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📰 News Article Summarizer (T5)")

article_text = st.text_area("Paste your article here:", height=300)

max_len = st.slider("Maximum summary length:", 32, 256, 128)
num_beams = st.slider("Number of beams (for beam search):", 1, 8, 4)

if st.button("Generate Summary"):
    if article_text.strip() == "":
        st.warning("Please enter an article to summarize!")
    else:
        with st.spinner("Generating summary..."):
            inputs = tokenizer("summarize: " + article_text,
                               return_tensors="pt",
                               max_length=512,
                               truncation=True).to(device)

            summary_ids = model.generate(
                **inputs,
                max_length=max_len,
                num_beams=num_beams,
                length_penalty=2.0,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        st.subheader("Summary:")
        st.write(summary)