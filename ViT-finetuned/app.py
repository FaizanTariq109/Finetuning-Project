import logging

import streamlit as st
from vit_runtime import load_model, predict, read_image

st.set_page_config(page_title="ViT Food Image Classifier", layout="centered")
st.title("ViT Food Image Classifier")
st.write("Classify a food photo among 101 Food-101 labels using the recovered fine-tuned Vision Transformer.")
st.caption("Portfolio demonstration. Scores are model probabilities, not guarantees of correctness. The model always chooses from its food labels, even for non-food images.")

@st.cache_resource(show_spinner=False)
def cached_model():
    return load_model()

uploaded = st.file_uploader("Choose a food photo", type=['jpg', 'jpeg', 'png'])
st.caption("JPG or PNG · up to 10 MB and 16 million pixels. Images are processed in server memory and are not saved by this app.")
if uploaded is not None:
    try:
        image = read_image(uploaded.getvalue())
    except ValueError as exc:
        st.warning(str(exc))
    else:
        st.image(image, caption="Uploaded food photo", use_column_width=True)
        try:
            with st.spinner("Loading model and classifying… The first request may download 344 MB."):
                processor, model = cached_model()
                predictions = predict(image, processor, model)
        except Exception as exc:
            logging.getLogger(__name__).warning("Classification unavailable (%s)", type(exc).__name__)
            st.error("The classifier is temporarily unavailable. Please try again shortly.")
            if st.button("Retry classification"):
                st.rerun()
        else:
            st.subheader("Top 5 predictions")
            for rank, prediction in enumerate(predictions, start=1):
                score = '<0.01%' if 0 < prediction['probability'] < 0.0001 else f"{prediction['probability']:.2%}"
                st.write(f"**{rank}. {prediction['label']}** — {score}")
                st.progress(prediction['probability'])
            st.caption("Labels are displayed exactly as stored in the recovered model configuration. The top five scores need not sum to 100%.")

with st.expander("About this model"):
    st.write("ViT-Base-sized architecture: 12 transformer layers, 768 hidden dimensions, 12 attention heads, 16×16 image patches and a 101-class head. Inputs are resized to 224×224 and normalized with the saved image processor.")
    st.write("The 101 labels match Food-101 exactly. The owner identified the dataset source as Kaggle kmader/food41 (Food Images / Food-101). The original training notebook, data split and evaluation report were not recovered; no benchmark accuracy is claimed.")
    st.markdown("[Source and documentation](https://github.com/FaizanTariq109/Finetuning-Project/tree/main/ViT-finetuned) · [Model weights](https://huggingface.co/FaizanTariq109/ViTFinetuned)")
