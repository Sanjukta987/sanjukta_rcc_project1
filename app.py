import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load your classifier model
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "tumor_classifier.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()

# Prediction function
def predict_tumor(image):
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Streamlit App Interface
st.set_page_config(page_title="🧠 Brain Tumor Detection Web App")

st.title("🧠 Brain Tumor Detection Web App")
st.markdown("Upload a brain MRI image, and the AI model will predict whether it shows signs of a brain tumor.")

# Image Upload
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            prediction, confidence = predict_tumor(image)

        # Display Prediction
        st.success(f"🎯 **Prediction Result:** `{prediction}` ({confidence * 100:.2f}% confidence)")
        
        # Recommendations
        if prediction == "No Tumor":
            st.markdown("✅ **No tumor detected.** You appear to have a normal brain scan.")
        else:
            st.markdown("⚠️ **Tumor Detected:**")
            st.markdown(f"- **Type:** {prediction}")
            st.markdown("- 📞 Please consult a neurologist or oncologist.")
            st.markdown("- 🧪 Further medical imaging and biopsy may be required.")

else:
    st.info("Please upload a brain MRI image to begin.")
