# app.py

import streamlit as st
from PIL import Image
from predict import predict_tumor

# Page setup
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered",
    initial_sidebar_state="auto"
)

# Title and Description
st.title("🧠 Brain Tumor Detection Web App")
st.markdown(
    """
    Upload a brain MRI image, and the AI model will predict whether it shows signs of a brain tumor.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

# Prediction Section
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("🔍 Predict"):
        try:
            result = predict_tumor(uploaded_file)
            st.success(f"🧠 **Prediction Result:** {result}")
        except Exception as e:
            st.error(f"Prediction failed. Error: {str(e)}")
