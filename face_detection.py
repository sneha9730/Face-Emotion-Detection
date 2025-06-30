import cv2
import numpy as np
import os

def detect_face_and_preprocess(image, target_size=(48, 48)):
    cascade_path = os.path.join("models", "haarcascade_frontalface_alt2.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise ValueError("Failed to load Haar cascade. Check the XML file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, target_size)
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, axis=(0, -1))

    return face_resized, (x, y, w, h)
