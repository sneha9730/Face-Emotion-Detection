import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from face_detection import detect_face_and_preprocess
import os

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Facial Emotion Detection")
st.write("Upload a clear face image and select a model to predict the facial emotion using deep learning models trained on FER2013.")

use_sample = st.checkbox("Use sample image instead of uploading")

if use_sample:
    sample_dir = "sample_images"
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    selected_sample = st.selectbox("Select a sample image", sample_files)
    img = Image.open(os.path.join(sample_dir, selected_sample)).convert('RGB')
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
    else:
        img = None

if img is not None:
    img_array = np.array(img)

    model_choice = st.selectbox("Choose a model", ["MiniXception", "VGGNet", "SqueezeNet"])

    if model_choice == "MiniXception":
        model_path = "models/mini_xception_fer2013.h5"
        target_size = (64, 64)
    else:
        model_path = f"models/{model_choice.lower()}_fer2013.h5"
        target_size = (48, 48)

    face_input, box = detect_face_and_preprocess(img_array, target_size=target_size)

    if face_input is None:
        st.warning("No face detected in the image.")
    else:
        model = load_model(model_path)
        prediction = model.predict(face_input)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        st.success(f"Predicted Emotion: **{predicted_emotion}**")

        x, y, w, h = box
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        st.image(img_array, caption="Detected Face", use_container_width=True)
