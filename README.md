# MoodLens

**Traductor de emociones a pictogramas TEA en tiempo real**

MoodLens es una aplicación que utiliza inteligencia artificial para detectar emociones faciales a través de la webcam y mostrar pictogramas TEA (Trastorno del Espectro Autista) correspondientes. Diseñada para facilitar la comunicación y comprensión emocional.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![React](https://img.shields.io/badge/React-18+-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)

---

## Demo

<!-- Enlace a video de Vimeo -->
[Ver demo en Vimeo](https://vimeo.com/TU_VIDEO_ID)

---

## Características

- **CNN propia** - Red neuronal convolucional entrenada con dataset FER2013
- **Detección en tiempo real** - Procesamiento de webcam con GPU
- **7 emociones** - Enojo, asco, miedo, felicidad, neutral, tristeza, sorpresa
- **Pictogramas TEA** - Comunicación visual accesible
- **App web moderna** - React + FastAPI con diseño oscuro
- **GPU accelerated** - Soporte CUDA para inferencia rápida

---

## Arquitectura

```
moodlens/
├── data/FER2013/              # Dataset (28,709 train + 7,178 test)
├── models/
│   ├── emotion_cnn.pth        # Modelo entrenado (69.7% accuracy)
│   ├── training_history.png   # Gráficas de entrenamiento
│   └── confusion_matrix.png   # Matriz de confusión
├── pictograms/                # Pictogramas TEA (7 emociones)
├── notebooks/
│   └── 01_emotion_cnn_training.ipynb
├── app/
│   ├── backend/
│   │   └── main.py            # FastAPI + WebSocket
│   ├── frontend/
│   │   └── src/App.jsx        # React + Vite
│   └── start.bat              # Script de inicio
├── webcam_emotion.py          # App standalone OpenCV
├── webcam_emotion_v2.py       # Versión simplificada
├── train_model.py             # Script de entrenamiento
└── requirements.txt           # Dependencias Python
```

---

## Instalación

### Requisitos
- Python 3.11+
- Node.js 18+
- GPU con CUDA (recomendado)

### 1. Clonar y crear entorno
```bash
cd moodlens
python -m venv .venv
.venv\Scripts\activate  # Windows
```

### 2. Instalar dependencias Python
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn[standard] websockets opencv-python numpy pillow
```

### 3. Instalar dependencias Frontend
```bash
cd app/frontend
npm install
```

---

## Uso

### Opción 1: App Web (Recomendada)

**Terminal 1 - Backend:**
```bash
cd moodlens
.venv\Scripts\activate
python app/backend/main.py
```

**Terminal 2 - Frontend:**
```bash
cd moodlens/app/frontend
npm run dev
```

Abrir: **http://localhost:5173**

### Opción 2: App Standalone (OpenCV)
```bash
cd moodlens
.venv\Scripts\activate
python webcam_emotion_v2.py
```

Controles: `Q` = Salir | `S` = Screenshot

---

## Modelo CNN

### Arquitectura
- 4 bloques convolucionales (64 → 128 → 256 → 512 filtros)
- BatchNorm + ReLU + Dropout (0.25)
- 2 capas fully connected (512 → 256 → 7)
- ~6 millones de parámetros

### Rendimiento
| Métrica | Valor |
|---------|-------|
| Accuracy | 69.7% |
| Dataset | FER2013 |
| Épocas | ~40 (Early Stopping) |

### Precisión por emoción
| Emoción | Precision |
|---------|-----------|
| Felicidad | 89% |
| Sorpresa | 80% |
| Asco | 66% |
| Enojo | 64% |
| Neutral | 61% |
| Tristeza | 56% |
| Miedo | 55% |

---

## Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| ML Framework | PyTorch 2.0 |
| Backend | FastAPI |
| Frontend | React + Vite |
| Streaming | WebSocket |
| Visión | OpenCV |
| GPU | CUDA 12.1 |

---

## Reentrenar modelo

```bash
cd moodlens
.venv\Scripts\activate
python train_model.py
```

El script:
1. Carga dataset FER2013 de `data/FER2013/`
2. Aplica data augmentation
3. Entrena con Early Stopping (patience=7)
4. Guarda modelo en `models/emotion_cnn.pth`

---

## Screenshots

<!-- Añadir capturas de pantalla aquí -->

| Pantalla de bienvenida | Detección de emociones |
|------------------------|------------------------|
| ![Welcome](screenshots/welcome.png) | ![Detection](screenshots/detection.png) |

---

## Licencia

Proyecto educativo. Pictogramas sujetos a sus respectivas licencias.

---

## Autor

Desarrollado por Borja Barber.
