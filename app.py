import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="Dog Breed Classification",
    page_icon="🐶",
    layout="wide"
)

# ======================================================
# Load Model
# ======================================================

@st.cache_resource
def load_cnn():
    return load_model("vgg16_dog breeds.h5")

model = load_cnn()

# Matches the exact alphabetical folder order from your dataset
class_names = [
    "Beagle",
    "Boxer",
    "Bulldog",
    "Dachshund",
    "German Shepherd",
    "Golden Retriever",
    "Labrador Retriever",
    "Poodle",
    "Rottweiler",
    "Yorkshire Terrier"
]

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("🐶 Model Information")

st.sidebar.markdown("### Architecture")
st.sidebar.write("✔ VGG16 Transfer Learning")
st.sidebar.write("✔ Dense & Dropout Layers")

st.sidebar.divider()

st.sidebar.metric("Input Size", "224 × 224")
st.sidebar.metric("Output Classes", "10")
st.sidebar.metric("Accuracy", "92.50 %")

st.sidebar.divider()

st.sidebar.info(
    "This application classifies dog images into 10 popular breeds using a VGG16 model."
)

# ======================================================
# Header
# ======================================================

st.title("🐶 Dog Breed Classification")

st.markdown(
"""
### Deep Learning based Dog Breed Identification

Upload an image of a dog and the trained VGG16 model will predict its breed.
"""
)
st.divider()

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Dog Image",
    type=["jpg", "jpeg", "png", "jfif"]
)

# -----------------------------
# If Image Uploaded
# -----------------------------
if uploaded_file is not None:

    image_file = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image_file, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        if st.button("Predict"):
            # 1. Resize to VGG16 input size
            img = image_file.resize((224, 224))
            
            # 2. Convert PIL image to numpy array
            img_array = image.img_to_array(img)
            
            # 3. Expand dimensions to shape (1, 224, 224, 3)
            img_batch = np.expand_dims(img_array, axis=0)
            
            # 4. Apply VGG16 preprocessing (Zero-centers pixels, converts RGB to BGR)
            # DO NOT divide by 255.0 when using VGG16 preprocess_input!
            img_preprocessed = preprocess_input(img_batch.copy())

            # 5. Run inference
            prediction = model.predict(img_preprocessed)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.success(f"Prediction : {class_names[predicted_class]}")
            st.info(f"Confidence : {confidence:.2f}%")

            st.divider()

            st.subheader("Class Probabilities")

            for i in range(len(class_names)):
                probability = float(prediction[0][i])
                st.write(f"**{class_names[i]}**")
                st.progress(probability)
                st.write(f"{probability * 100:.2f}%")

# ======================================================
# Bottom Section
# ======================================================

st.divider()

st.subheader("📋 Model Details")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Model", "VGG16")

with c2:
    st.metric("Input", "224×224")

with c3:
    st.metric("Classes", "10")

with c4:
    st.metric("Framework", "TensorFlow")

st.divider()

st.caption(
    "Developed by Tamil Arasan | Dog Breed Classification using Deep Learning"
)