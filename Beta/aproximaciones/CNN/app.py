import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

#streamlit run app.py(para activarlo)

# Función para cargar el modelo y evitar recargas innecesarias
@st.cache_resource
def load_emotion_model():
    model = load_model('cnn_emotion_detection.h5')
    return model

model = load_emotion_model()

# Cargar el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Etiquetas de emociones
emotion_labels = ['enfado', 'Disgusto', 'Miedo', 'Felicidad', 'Neutral', 'Tristeza', 'Sorpresa']

# Función para preprocesar la imagen de la cara
def preprocess_image(face):
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    return face

# Función para detectar emociones en una imagen (entrada: imagen en formato BGR)
def detect_emotion(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) == 0:
        st.warning("No se detectaron rostros en la imagen.")
        return image
    
    # Para cada rostro detectado se realiza la predicción de la emoción
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]  # Recortar el rostro detectado
        processed_face = preprocess_image(face_region)
        
        # Predecir la emoción usando el modelo
        prediction = model.predict(processed_face)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        
        # Dibujar un rectángulo y escribir la emoción detectada en la imagen
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image

# Configuración de la interfaz de Streamlit
st.title("MOODLENS")
st.write("Sube una imagen para detectar emociones.")

# Cargar la imagen subida por el usuario
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convertir el archivo subido a un arreglo de bytes y luego a una imagen OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Realizar la detección de emociones sobre la imagen
    result_image = detect_emotion(image.copy())
    
    # Convertir la imagen de BGR a RGB para que se muestre correctamente en Streamlit
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    st.image(result_image_rgb, caption="Imagen procesada", use_column_width=True)
