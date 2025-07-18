import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# Check if model exists
print("Checking model path:", os.path.exists("model/brain_tumor_model.h5"))

# ✅ Load the model correctly
model = load_model("model/brain_tumor_model.h5")  # model is inside /model/

def predict_tumor(uploaded_file):
    try:
        # ✅ Load image, convert to grayscale
        image = Image.open(uploaded_file).convert('L')

        # ✅ Resize to 100x100 (matches your training model)
        image = image.resize((100, 100))

        # Convert to array and normalize
        image = img_to_array(image)
        image = image / 255.0

        # ✅ Add dimensions to match input shape: (1, 100, 100, 1)
        image = np.expand_dims(image, axis=0)

        # ✅ Predict
        prediction = model.predict(image)[0][0]

        return "🧠 Tumor Detected" if prediction >= 0.5 else "✅ No Tumor Detected"

    except Exception as e:
        return f"❌ Prediction failed. Error: {str(e)}"
