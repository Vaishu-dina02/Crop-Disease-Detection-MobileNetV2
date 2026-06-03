import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# ---------------- MODEL ---------------- #

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')
])

# Load trained weights
model.load_weights("weights.weights.h5")

# ---------------- CLASS NAMES (CORRECT ORDER) ---------------- #

class_names = [
    "Carrot__Rotten",
    "Corn__northern_leaf_blight",
    "Tea__anthracnose",
    "Tomato__target_spot",
    "Wheat___Brown_Rust"
]

# ---------------- UI ---------------- #

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌿")

st.title("🌿 Crop Disease Detection")

file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

# ---------------- PREDICTION ---------------- #

if file is not None:
    img = Image.open(file).convert("RGB")
    img_resized = img.resize((224, 224))

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    result = class_names[np.argmax(pred)]
    confidence = float(np.max(pred))

    # Show image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Show result
    st.success(f"🌿 Disease: {result}")
    st.info(f"📊 Confidence: {confidence*100:.2f}%")

    # Alert
    if "healthy" not in result.lower():
        st.warning("⚠️ Disease detected! Take action immediately")

##streamlit run c:/Users/Vaishnavidina/OneDrive/Desktop/CropApp/app.py