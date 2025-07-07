import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from fer import FER
import os

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Facial Emotion Detection")
st.write("Upload a clear face image and select a model to predict the facial emotion using deep learning or the FER library.")

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

model_choice = st.selectbox("Choose a model", ["Mini - Xception", "VGGNet", "SqueezeNet", "FER Library"])

if img is not None:
    img_array = np.array(img)

    detector = FER(mtcnn=True)
    results = detector.detect_emotions(img_array)

    if not results:
        st.warning("No face detected.")
    else:
        box = results[0]["box"]
        x, y, w, h = box
        face = img_array[y:y+h, x:x+w]

        if model_choice == "FER Library":
            emotions = results[0]["emotions"]
            top_emotion = max(emotions, key=emotions.get)

            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.success(f"Predicted Emotion (FER): **{top_emotion.capitalize()}**")
            st.image(img_array, caption="Detected Face", use_container_width=True)
            st.write("All Detected Emotions:")
            st.json(emotions)

        else:
            if model_choice == "Mini - Xception":
                model_path = "models/mini_xception_fer2013.h5"
                target_size = (64, 64)
            else:
                model_path = f"models/{model_choice.lower()}_fer2013.h5"
                target_size = (48, 48)

            model = load_model(model_path)

            face_resized = cv2.resize(face, target_size)
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            face_input = face_gray.reshape(1, target_size[0], target_size[1], 1) / 255.0

            prediction = model.predict(face_input)
            predicted_emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.success(f"Predicted Emotion ({model_choice}): **{predicted_emotion}**")
            st.image(img_array, caption="Detected Face", use_container_width=True)
            
