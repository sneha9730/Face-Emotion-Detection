## Facial Emotion Detection

This is a deep learning project that detects facial emotions in uploaded images using pre-trained models like Mini Xception, VGGNet, SqueezeNet, and the FER library. The 7 facial emotions used are: **Angry**, **Sad**, **Happy**, **Surprise**, **Disgust**, **Fear**, and **Neutral**. 

Face detection is performed using the FER library with MTCNN, and depending on the selected model, emotions are predicted accordingly. The app is built using **Streamlit** and supports image input, face detection, and preprocessing.

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
