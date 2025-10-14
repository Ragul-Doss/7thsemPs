import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model from your path
model = tf.keras.models.load_model("/content/brain_tumor_model.h5")

st.title("🧬 Brain Tumor Detection System")
st.write("Upload a brain MRI image to check for tumor presence.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess image
    img = image.resize((150,150))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    if st.button("Analyze MRI"):
        prediction = model.predict(img_array)[0][0]
        if prediction > 0.5:
            st.error("🚨 Tumor Detected! Please consult a specialist.")
        else:
            st.success("✅ No Tumor Detected.")
