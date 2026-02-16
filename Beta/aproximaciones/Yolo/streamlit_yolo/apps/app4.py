import os
import streamlit as st
import cv2
import numpy as np
import imutils
import torch
import time
from ultralytics import YOLO

# Configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Cargar modelo YOLO para emociones
model_path = os.path.join(BASE_DIR, "best.pt")
emotionModel = YOLO(model_path)

# Descargar el clasificador de rostros de OpenCV si no está disponible
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(face_cascade_path):
    import urllib.request
    haarcascade_url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(haarcascade_url, face_cascade_path)
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Cargar imágenes de emociones y convertirlas a RGB
emotion_images = {}
for emotion in classes:
    img_path = os.path.join(BASE_DIR, 'emociones', f'{emotion}.jpg')
    img = cv2.imread(img_path)
    if img is not None:
        emotion_images[emotion] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return faces

def predict_emotion(frame):
    try:
        results = emotionModel(frame)
        detected_emotion = None
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_index = int(box.cls[0])
                detected_emotion = classes[label_index]
                prob = box.conf[0] * 100
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{detected_emotion}: {prob:.0f}%", (x1+5, y1-15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 255, 50), 1)
        
        return frame, detected_emotion
    except Exception as e:
        st.error(f"Error en predict_emotion: {e}")
        return frame, None

if start_button:
    st.session_state.is_running = True
    if st.session_state.cap is None:
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
        faces = detect_faces(frame)
        
        # Solo dibujar un cuadro verde en los rostros detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        frame, detected_emotion = predict_emotion(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB", use_container_width=True)

        if detected_emotion and detected_emotion in emotion_images:
            emotion_img = emotion_images.get(detected_emotion)
            if emotion_img is not None:
                emotion_placeholder.image(emotion_img, caption=f"Emoción: {detected_emotion}", use_container_width=True)
        
        time.sleep(0.05)
    else:
        st.error("Error al capturar video")
        st.session_state.is_running = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None

if not st.session_state.is_running:
    video_placeholder.info("Presiona el botón 'Iniciar' para comenzar la detección")
    emotion_placeholder.empty()
