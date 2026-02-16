"""
MoodLens API - Backend FastAPI para detección de emociones
"""

import asyncio
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import io

app = FastAPI(title="MoodLens API")

# CORS para React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_cnn.pth')
PICTOGRAMS_DIR = os.path.join(BASE_DIR, 'pictograms')
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Emociones
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_ES = ['Enojo', 'Asco', 'Miedo', 'Felicidad', 'Neutral', 'Tristeza', 'Sorpresa']

# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📍 Dispositivo: {device}")


# Arquitectura CNN
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


# Cargar modelo
print(f"📂 Cargando modelo...")
model = EmotionCNN(7).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"   ✓ Modelo cargado")

# Cargar detector facial
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Transformaciones
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def process_frame(frame_data: str):
    """Procesa un frame base64 y devuelve la emoción detectada."""
    try:
        # Decodificar base64
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
        
        # Detectar caras
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        if len(faces) == 0:
            return {"detected": False}
        
        # Procesar primera cara
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        
        # Predecir
        with torch.no_grad():
            output = model(face_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = output.argmax(1).item()
            confidence = probs[0][pred_idx].item()
        
        # Coordenadas normalizadas para el frontend
        h_frame, w_frame = frame.shape[:2]
        
        return {
            "detected": True,
            "emotion": EMOTIONS[pred_idx],
            "emotion_es": EMOTIONS_ES[pred_idx],
            "confidence": round(confidence * 100, 1),
            "probabilities": {EMOTIONS[i]: round(probs[0][i].item() * 100, 1) for i in range(7)},
            "face": {
                "x": round(x / w_frame, 3),
                "y": round(y / h_frame, 3),
                "width": round(w / w_frame, 3),
                "height": round(h / h_frame, 3)
            }
        }
    
    except Exception as e:
        print(f"Error procesando frame: {e}")
        return None


@app.get("/")
async def root():
    return {"message": "MoodLens API", "status": "running"}


@app.get("/pictograms/{emotion}")
async def get_pictogram(emotion: str):
    """Devuelve el pictograma de una emoción."""
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(PICTOGRAMS_DIR, f"{emotion}{ext}")
        if os.path.exists(path):
            return FileResponse(path)
    return {"error": "Pictogram not found"}


@app.websocket("/ws/emotion")
async def websocket_emotion(websocket: WebSocket):
    """WebSocket para detección de emociones en tiempo real."""
    await websocket.accept()
    print("🔌 Cliente conectado")
    
    try:
        while True:
            # Recibir frame base64
            data = await websocket.receive_text()
            
            # Procesar
            result = process_frame(data)
            
            if result:
                await websocket.send_json(result)
            
            # Pequeña pausa para no saturar
            await asyncio.sleep(0.05)
    
    except WebSocketDisconnect:
        print("🔌 Cliente desconectado")
    except Exception as e:
        print(f"Error WebSocket: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
