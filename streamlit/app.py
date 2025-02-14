import os
import streamlit as st
import cv2
import numpy as np
import imutils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import absl.logging
import time

# Desactivar warnings informativos de TensorFlow
absl.logging.set_verbosity(absl.logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Cargar imágenes de emociones y convertirlas a RGB
emotion_images = {emotion: cv2.cvtColor(cv2.imread(os.path.join(BASE_DIR, 'emociones', f'{emotion}.jpg')), cv2.COLOR_BGR2RGB) 
                  if cv2.imread(os.path.join(BASE_DIR, 'emociones', f'{emotion}.jpg')) is not None else None 
                  for emotion in classes}

prototxtPath = os.path.join(BASE_DIR, 'face_detector', 'deploy.prototxt')
weightsPath = os.path.join(BASE_DIR, 'face_detector', 'res10_300x300_ssd_iter_140000.caffemodel')
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

emotionModel = load_model(os.path.join(BASE_DIR, "modelFEC.h5"))

def predict_emotion(frame):
    try:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        current_emotion = None

        for i in range(0, detections.shape[2]):
            if detections[0, 0, i, 2] > 0.4:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (Xi, Yi, Xf, Yf) = box.astype("int")
                Xi, Yi = max(0, Xi), max(0, Yi)

                face = frame[Yi:Yf, Xi:Xf]
                if face.size == 0:
                    continue

                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_gray = cv2.resize(face_gray, (48, 48))
                face_array = img_to_array(face_gray)
                face_array = np.expand_dims(face_array, axis=0)

                pred = emotionModel.predict(face_array)
                label_index = np.argmax(pred)
                current_emotion = classes[label_index]
                prob = np.max(pred) * 100

                cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (0, 100, 0), -1)
                cv2.putText(frame, f"{current_emotion}: {prob:.0f}%", (Xi+5, Yi-15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 255, 50), 1)
                cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (0, 255, 0), 2)

        return frame, current_emotion

    except Exception as e:
        print("Error en predict_emotion:", e)
        return frame, None

st.title("MoodLens Traductor de emociones a pictogramas TEA")
st.write("Descubre el mundo de las emociones.")

col_buttons = st.columns(2)
with col_buttons[0]:
    start_button = st.button("▶️ Iniciar")
with col_buttons[1]:
    stop_button = st.button("⏹️ Detener")

col1, col2 = st.columns(2)
video_placeholder = col1.empty()
emotion_placeholder = col2.empty()

if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

if start_button:
    st.session_state.is_running = True
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(0)

if stop_button:
    st.session_state.is_running = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

while st.session_state.is_running and st.session_state.cap is not None:
    ret, frame = st.session_state.cap.read()
    if ret:
        frame = imutils.resize(frame, width=640)
        frame, detected_emotion = predict_emotion(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_placeholder.image(frame, channels="RGB", use_container_width=True)

        if detected_emotion and detected_emotion in emotion_images:
            emotion_img = emotion_images[detected_emotion]
            if emotion_img is not None:
                emotion_placeholder.image(emotion_img, caption=f"Emoción: {detected_emotion}", use_container_width=True)

        time.sleep(0.05)
    else:
        st.error("Error al capturar video")
        st.session_state.is_running = False
        st.session_state.cap.release()
        st.session_state.cap = None

if not st.session_state.is_running:
    video_placeholder.info("Presiona el botón 'Iniciar' para comenzar la detección")
    if emotion_placeholder:
        emotion_placeholder.empty()
