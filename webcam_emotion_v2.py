"""
MoodLens v2 - Detección de Emociones con Pictogramas TEA (versión simplificada)
Pictograma solo al lado de la cara detectada.
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import os

print("=" * 60)
print("🎭 MoodLens v2 - Pictogramas TEA")
print("=" * 60)

# Verificar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📍 Dispositivo: {device}")

# Rutas
MODEL_PATH = 'models/emotion_cnn.pth'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
PICTOGRAMS_DIR = 'pictograms'

# Emociones
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_ES = ['Enojo', 'Asco', 'Miedo', 'Felicidad', 'Neutral', 'Tristeza', 'Sorpresa']
COLORS = [
    (0, 0, 255),      # Enojo - Rojo
    (0, 128, 0),      # Asco - Verde
    (128, 0, 128),    # Miedo - Púrpura
    (0, 255, 255),    # Felicidad - Amarillo
    (128, 128, 128),  # Neutral - Gris
    (255, 0, 0),      # Tristeza - Azul
    (0, 165, 255)     # Sorpresa - Naranja
]

# CNN
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                   nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                   nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                   nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                   nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                   nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                   nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                   nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                   nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512*3*3, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                nn.Dropout(0.5), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                nn.Dropout(0.5), nn.Linear(256, num_classes))
    
    def forward(self, x):
        return self.fc(self.conv4(self.conv3(self.conv2(self.conv1(x)))))


def load_pictograms(pictograms_dir, size=(100, 100)):
    """Carga pictogramas TEA."""
    pictograms = {}
    print(f"🖼️  Cargando pictogramas...")
    for emotion in EMOTIONS:
        for ext in ['.jpg', '.jpeg', '.png']:
            path = os.path.join(pictograms_dir, f"{emotion}{ext}")
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    pictograms[emotion] = cv2.resize(img, size)
                    print(f"   ✓ {emotion}")
                break
    return pictograms


def load_model(model_path, device):
    """Carga el modelo."""
    print(f"📂 Cargando modelo...")
    model = EmotionCNN(7).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ Accuracy: {checkpoint.get('accuracy', 0):.2f}%")
    return model


def preprocess_face(face_img):
    """Preprocesa cara."""
    transform = transforms.Compose([
        transforms.Grayscale(1), transforms.Resize((48, 48)),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return transform(Image.fromarray(face_rgb)).unsqueeze(0)


def predict_emotion(model, face_tensor, device):
    """Predice emoción."""
    with torch.no_grad():
        output = model(face_tensor.to(device))
        probs = torch.softmax(output, dim=1)
        pred_idx = output.argmax(1).item()
    return pred_idx, probs[0][pred_idx].item()


def overlay_pictogram(frame, pictogram, x, y):
    """Superpone pictograma."""
    h, w = pictogram.shape[:2]
    if y + h > frame.shape[0]: h = frame.shape[0] - y
    if x + w > frame.shape[1]: w = frame.shape[1] - x
    if y >= 0 and x >= 0:
        frame[y:y+h, x:x+w] = pictogram[:h, :w]


def main():
    model = load_model(MODEL_PATH, device)
    pictograms = load_pictograms(PICTOGRAMS_DIR, size=(150, 150))
    
    if not pictograms:
        print("❌ No se encontraron pictogramas")
        return
    
    print(f"📷 Cargando detector facial...")
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    print(f"🎥 Iniciando webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n✅ Listo! Presiona 'Q' para salir, 'S' para screenshot\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_tensor = preprocess_face(face_img)
            pred_idx, confidence = predict_emotion(model, face_tensor, device)
            
            emotion = EMOTIONS[pred_idx]
            emotion_es = EMOTIONS_ES[pred_idx]
            color = COLORS[pred_idx]
            
            # Rectángulo alrededor de la cara
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Etiqueta
            label = f"{emotion_es} ({confidence*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y-th-10), (x+tw+10, y), color, -1)
            cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # Pictograma SOLO al lado de la cara
            if emotion in pictograms:
                picto = pictograms[emotion]
                picto_x = x + w + 15  # A la derecha
                picto_y = y + (h - picto.shape[0]) // 2  # Centrado
                
                # Si no cabe a la derecha, ponerlo a la izquierda
                if picto_x + picto.shape[1] > frame.shape[1]:
                    picto_x = x - picto.shape[1] - 15
                
                if picto_x >= 0:
                    overlay_pictogram(frame, picto, picto_x, max(0, picto_y))
        
        # Título simple
        cv2.putText(frame, "MoodLens v2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        cv2.imshow('MoodLens v2 - Pictogramas TEA', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'):
            cv2.imwrite(f'screenshot_{frame_count}.png', frame)
            print(f"📸 Screenshot: screenshot_{frame_count}.png")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Cerrado")


if __name__ == "__main__":
    main()
