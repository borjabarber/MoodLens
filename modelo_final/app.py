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

# Obtener la ruta absoluta del directorio actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------ CARGA DE MODELOS E IMÁGENES ------------------

# Lista de emociones
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Cargar imágenes de emociones y convertirlas a RGB
emotion_images = {}
for emotion in classes:
    img_path = os.path.join(BASE_DIR, 'emociones', f'{emotion}.jpg')
    img = cv2.imread(img_path)
    if img is not None:
        emotion_images[emotion] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir BGR a RGB
    else:
        emotion_images[emotion] = None  # Si la imagen no se carga, se asigna None

# Cargar el detector de rostros
prototxtPath = os.path.join(BASE_DIR, 'face_detector', 'deploy.prototxt')
weightsPath = os.path.join(BASE_DIR, 'face_detector', 'res10_300x300_ssd_iter_140000.caffemodel')
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Cargar el modelo de detección de emociones
emotionModel = load_model(os.path.join(BASE_DIR, "modelFEC.h5"))
emotionModel.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ------------------ FUNCIÓN DE PREDICCIÓN ------------------

def predict_emotion(frame):
    """
    Detecta rostros en el frame y predice la emoción.
    Dibuja la caja y etiqueta en la imagen.
    """
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

# ------------------ INTERFAZ DE STREAMLIT ------------------

st.title("MoodLens Traductor de emociones a pictogramas TEA")
st.write("Concede acceso a la cámara y descubre el mundo de la emociones.")

# Crear columnas para mostrar el video y la imagen al lado
col1, col2 = st.columns(2)

# Iniciar la webcam
cap = cv2.VideoCapture(0)

# Inicializamos los placeholders (vacíos) para actualizar solo cuando se necesite
video_placeholder = col1.empty()
emotion_placeholder = col2.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("No se pudo capturar el frame.")
        break

    frame = imutils.resize(frame, width=640)
    frame, detected_emotion = predict_emotion(frame)

    # Convertir BGR a RGB para Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Limpiar el contenido anterior antes de mostrar el nuevo fotograma
    video_placeholder.image(frame, channels="RGB", use_container_width=True)

    # Mostrar la imagen de la emoción detectada en la segunda columna si se ha detectado una emoción
    if detected_emotion and detected_emotion in emotion_images:
        emotion_img = emotion_images[detected_emotion]
        if emotion_img is not None:
            emotion_placeholder.image(emotion_img, caption=f"Emoción: {detected_emotion}", use_container_width=True)

    # Pausa para evitar que Streamlit se bloquee
    time.sleep(0.05)

cap.release()
