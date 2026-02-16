"""
MoodLens - Detección de Emociones en Tiempo Real con Pictogramas TEA
Captura video de la webcam, detecta caras, clasifica emociones y muestra pictogramas TEA.
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import os

print("=" * 60)
print("🎭 MoodLens - Traductor de Emociones a Pictogramas TEA")
print("=" * 60)

# Verificar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📍 Dispositivo: {device}")

# Rutas
MODEL_PATH = 'models/emotion_cnn.pth'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
PICTOGRAMS_DIR = 'pictograms'

# Emociones (orden debe coincidir con el entrenamiento)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_ES = ['Enojo', 'Asco', 'Miedo', 'Felicidad', 'Neutral', 'Tristeza', 'Sorpresa']
COLORS = [
    (0, 0, 255),      # Enojo - Rojo
    (0, 128, 0),      # Asco - Verde oscuro
    (128, 0, 128),    # Miedo - Púrpura
    (0, 255, 255),    # Felicidad - Amarillo
    (128, 128, 128),  # Neutral - Gris
    (255, 0, 0),      # Tristeza - Azul
    (0, 165, 255)     # Sorpresa - Naranja
]

# Arquitectura CNN (debe coincidir con el entrenamiento)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


def load_pictograms(pictograms_dir, size=(300, 300)):
    """Carga todos los pictogramas TEA."""
    pictograms = {}
    print(f"🖼️  Cargando pictogramas desde: {pictograms_dir}")
    
    for emotion in EMOTIONS:
        # Buscar archivo jpg o png
        for ext in ['.jpg', '.jpeg', '.png']:
            path = os.path.join(pictograms_dir, f"{emotion}{ext}")
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, size)
                    pictograms[emotion] = img
                    print(f"   ✓ {emotion}: {path}")
                break
        
        if emotion not in pictograms:
            print(f"   ⚠ {emotion}: No encontrado")
    
    return pictograms


def load_model(model_path, device):
    """Carga el modelo entrenado."""
    print(f"📂 Cargando modelo desde: {model_path}")
    
    model = EmotionCNN(num_classes=7).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    accuracy = checkpoint.get('accuracy', 'N/A')
    print(f"   ✓ Modelo cargado (Accuracy: {accuracy:.2f}%)")
    return model


def preprocess_face(face_img):
    """Preprocesa una imagen de cara para el modelo."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    
    return transform(face_pil).unsqueeze(0)


def predict_emotion(model, face_tensor, device):
    """Predice la emoción de una cara."""
    face_tensor = face_tensor.to(device)
    
    with torch.no_grad():
        output = model(face_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = output.argmax(1).item()
        confidence = probs[0][pred_idx].item()
    
    return pred_idx, confidence, probs[0].cpu().numpy()


def overlay_pictogram(frame, pictogram, x, y):
    """Superpone un pictograma en el frame."""
    h, w = pictogram.shape[:2]
    
    # Asegurar que no salga del frame
    if y + h > frame.shape[0]:
        h = frame.shape[0] - y
    if x + w > frame.shape[1]:
        w = frame.shape[1] - x
    if y < 0 or x < 0:
        return
    
    # Superponer el pictograma
    frame[y:y+h, x:x+w] = pictogram[:h, :w]


def draw_emotion_bar(frame, probs, x, y, width=180, height=12):
    """Dibuja una barra de probabilidades para cada emoción."""
    for i, (emotion, prob) in enumerate(zip(EMOTIONS_ES, probs)):
        bar_y = y + i * (height + 4)
        # Fondo
        cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + height), (30, 30, 30), -1)
        # Barra de progreso
        bar_width = int(prob * width)
        cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + height), COLORS[i], -1)
        # Texto
        text = f"{emotion}: {prob*100:.0f}%"
        cv2.putText(frame, text, (x + width + 5, bar_y + height - 1), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def main():
    # Cargar modelo
    model = load_model(MODEL_PATH, device)
    
    # Cargar pictogramas
    pictograms = load_pictograms(PICTOGRAMS_DIR, size=(120, 120))
    
    if len(pictograms) == 0:
        print("❌ Error: No se encontraron pictogramas")
        return
    
    # Cargar detector de caras
    print(f"📷 Cargando detector facial...")
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    # Iniciar webcam
    print(f"🎥 Iniciando webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la webcam")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "=" * 60)
    print("✅ Sistema listo!")
    print("   Presiona 'Q' para salir")
    print("   Presiona 'S' para capturar screenshot")
    print("=" * 60 + "\n")
    
    frame_count = 0
    current_emotion = None
    current_confidence = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Voltear horizontalmente (efecto espejo)
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Convertir a escala de grises para detección facial
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        
        # Procesar cada cara detectada
        for (x, y, w, h) in faces:
            # Extraer región de la cara
            face_img = frame[y:y+h, x:x+w]
            
            # Preprocesar y predecir
            face_tensor = preprocess_face(face_img)
            pred_idx, confidence, probs = predict_emotion(model, face_tensor, device)
            
            emotion = EMOTIONS[pred_idx]
            emotion_es = EMOTIONS_ES[pred_idx]
            color = COLORS[pred_idx]
            
            current_emotion = emotion
            current_confidence = confidence
            
            # Dibujar rectángulo alrededor de la cara
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Etiqueta de emoción sobre la cara
            label = f"{emotion_es} ({confidence*100:.0f}%)"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Mostrar pictograma al lado de la cara
            if emotion in pictograms:
                picto = pictograms[emotion]
                picto_x = x + w + 20  # A la derecha de la cara
                picto_y = y + (h - picto.shape[0]) // 2  # Centrado verticalmente
                
                # Si no cabe a la derecha, ponerlo a la izquierda
                if picto_x + picto.shape[1] > frame.shape[1]:
                    picto_x = x - picto.shape[1] - 20
                
                if picto_x >= 0 and picto_y >= 0:
                    overlay_pictogram(frame, picto, picto_x, max(0, picto_y))
            
            # Barra de probabilidades (solo para la primera cara)
            if len(faces) == 1:
                draw_emotion_bar(frame, probs, 10, 60)
        
        # Panel de pictograma grande cuando hay detección
        if current_emotion and current_emotion in pictograms:
            # Área de pictograma grande en la esquina superior derecha
            big_picto = cv2.resize(pictograms[current_emotion], (180, 180))
            panel_x = frame.shape[1] - 200
            panel_y = 10
            
            # Fondo del panel
            cv2.rectangle(frame, (panel_x - 10, panel_y - 5), 
                         (panel_x + 190, panel_y + 215), (40, 40, 40), -1)
            cv2.rectangle(frame, (panel_x - 10, panel_y - 5), 
                         (panel_x + 190, panel_y + 215), COLORS[EMOTIONS.index(current_emotion)], 2)
            
            # Pictograma
            overlay_pictogram(frame, big_picto, panel_x, panel_y)
            
            # Texto debajo del pictograma
            emotion_text = EMOTIONS_ES[EMOTIONS.index(current_emotion)]
            cv2.putText(frame, emotion_text, (panel_x + 10, panel_y + 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Título
        cv2.rectangle(frame, (0, 0), (350, 45), (40, 40, 40), -1)
        cv2.putText(frame, "MoodLens - Pictogramas TEA", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Info del sistema
        info = f"Caras: {len(faces)} | GPU: {'Si' if torch.cuda.is_available() else 'No'}"
        cv2.putText(frame, info, (10, frame.shape[0] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Mostrar frame
        cv2.imshow('MoodLens - Traductor de Emociones a Pictogramas TEA', frame)
        
        # Controles de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n👋 Cerrando aplicación...")
            break
        elif key == ord('s') or key == ord('S'):
            screenshot_path = f'screenshot_{frame_count}.png'
            cv2.imwrite(screenshot_path, frame)
            print(f"📸 Screenshot guardado: {screenshot_path}")
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Aplicación cerrada correctamente")


if __name__ == "__main__":
    main()
