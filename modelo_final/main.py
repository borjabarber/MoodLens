# Import de librerias
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import imutils
import cv2
import absl.logging

# Desactivar warnings de TensorFlow
absl.logging.set_verbosity(absl.logging.ERROR)

# Tipos de emociones del detector
classes = ['angry','disgust','fear','happy','neutral','sad','surprise']

# Cargamos imágenes de emociones
emotion_images = {
    'angry': cv2.imread('emociones/angry.jpg'),
    'disgust': cv2.imread('emociones/disgust.jpg'),
    'fear': cv2.imread('emociones/fear.jpg'),
    'happy': cv2.imread('emociones/happy.jpg'),
    'neutral': cv2.imread('emociones/neutral.jpg'),
    'sad': cv2.imread('emociones/sad.jpg'),
    'surprise': cv2.imread('emociones/surprise.jpg')
}

# Cargamos el modelo de detección de caras
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el detector de clasificación de emociones
emotionModel = load_model("modelFEC.h5")
emotionModel.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Configuración de cámara
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
last_emotion = None

def predict_emotion(frame, faceNet, emotionModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            if Xi < 0: Xi = 0
            if Yi < 0: Yi = 0
            
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face2 = img_to_array(face)
            face2 = np.expand_dims(face2, axis=0)

            faces.append(face2)
            locs.append((Xi, Yi, Xf, Yf))

            pred = emotionModel.predict(face2)
            preds.append(pred[0])

    return (locs, preds)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width=640)
    (locs, preds) = predict_emotion(frame, faceNet, emotionModel)
    
    current_emotion = None
    
    # Panel derecho para la imagen de la emoción
    right_panel = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)
    
    for (box, pred) in zip(locs, preds):
        (Xi, Yi, Xf, Yf) = box
        label_index = np.argmax(pred)
        label = classes[label_index]
        prob = np.max(pred) * 100

        # Dibujar cuadro y texto 
        cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (0, 100, 0), -1)  # Barra superior
        cv2.putText(frame, f"{label}: {prob:.0f}%", (Xi+5, Yi-15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 255, 50), 1)  # Texto verde 
        cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (0, 255, 0), 2)  # Borde verde 
    
        current_emotion = label

    # Mostrar imagen de emoción en el panel derecho
    if current_emotion:
        emotion_image = emotion_images.get(current_emotion, None)
        if emotion_image is not None:
            emotion_image = cv2.resize(emotion_image, (300, 300))
            y_start = (right_panel.shape[0] - 300) // 2
            right_panel[y_start:y_start+300, 0:300] = emotion_image
            cv2.putText(right_panel, current_emotion.upper(), (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2)  # Texto verde claro
    
    # Combinar los dos paneles
    composite_frame = cv2.hconcat([frame, right_panel])

    cv2.imshow("Deteccion de Emociones", composite_frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cam.release()
cv2.destroyAllWindows()
