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



## Metricas

añadir aqui

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

### 1. Clonar y crear entorno virtual
```bash
git clone https://github.com/TU_USUARIO/moodlens.git
cd moodlens
python -m venv .venv

# Activar entorno virtual
.venv\Scripts\activate        # Windows (cmd / PowerShell)
source .venv/bin/activate      # macOS / Linux
```

### 2. Instalar dependencias Python
```bash
# PyTorch con soporte CUDA (GPU) — elige la URL según tu versión de CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Resto de dependencias del proyecto
pip install -r requirements.txt

# Dependencias del backend (FastAPI, WebSocket, OpenCV…)
pip install -r app/backend/requirements.txt
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
# Activar entorno virtual (si no está activo)
.venv\Scripts\activate        # Windows
source .venv/bin/activate      # macOS / Linux

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
# Activar entorno virtual (si no está activo)
.venv\Scripts\activate        # Windows
source .venv/bin/activate      # macOS / Linux

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

![Métricas del modelo](screenshoots/metricas/Captura%20de%20pantalla%202026-01-27%20115137.png)

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
# Activar entorno virtual
.venv\Scripts\activate        # Windows
source .venv/bin/activate      # macOS / Linux

python train_model.py
```

El script:
1. Carga dataset FER2013 de `data/FER2013/`
2. Aplica data augmentation
3. Entrena con Early Stopping (patience=7)
4. Guarda modelo en `models/emotion_cnn.pth`

---

## Screenshots

![Pantalla de bienvenida](screenshoots/app_final/pantalla_welcome.png)

| | | |
|---|---|---|
| ![App 1](screenshoots/app_final/Captura%20de%20pantalla%202026-03-12%20201003.png) | ![App 2](screenshoots/app_final/Captura%20de%20pantalla%202026-03-12%20201018.png) | ![App 3](screenshoots/app_final/Captura%20de%20pantalla%202026-03-12%20201042.png) |
| ![App 4](screenshoots/app_final/Captura%20de%20pantalla%202026-03-12%20201106.png) | ![App 5](screenshoots/app_final/Captura%20de%20pantalla%202026-03-12%20201127.png) | ![App 6](screenshoots/app_final/Captura%20de%20pantalla%202026-03-12%20201147.png) |
| ![App 7](screenshoots/app_final/Captura%20de%20pantalla%202026-03-12%20201205.png) | ![App 8](screenshoots/app_final/Captura%20de%20pantalla%202026-03-12%20201224.png) | ![App 9](screenshoots/app_final/Captura%20de%20pantalla%202026-03-12%20201243.png) |
| ![App 10](screenshoots/app_final/Captura%20de%20pantalla%202026-03-12%20201323.png) | ![App 11](screenshoots/app_final/Captura%20de%20pantalla%202026-03-12%20201343.png) | |

---

## Licencia

Proyecto educativo. Pictogramas sujetos a sus respectivas licencias.

---

## Autor

Desarrollado por Borja Barber.
