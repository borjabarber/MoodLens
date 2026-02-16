"""
MoodLens - Script de Entrenamiento de CNN para Detección de Emociones
Ejecuta el entrenamiento completo con PyTorch y GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import os

print("=" * 60)
print("🎭 MoodLens - Entrenamiento de CNN para Emociones")
print("=" * 60)

# Verificar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n📍 Dispositivo: {device}')
if torch.cuda.is_available():
    print(f'🎮 GPU: {torch.cuda.get_device_name(0)}')

# Hiperparámetros
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
IMG_SIZE = 48
NUM_CLASSES = 7
PATIENCE = 7

# Rutas
DATA_DIR = 'data/FER2013'
MODEL_PATH = 'models/emotion_cnn.pth'

# Nombres de emociones
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_ES = ['Enojo', 'Asco', 'Miedo', 'Felicidad', 'Neutral', 'Tristeza', 'Sorpresa']

print(f'\n📊 Configuración:')
print(f'   Batch size: {BATCH_SIZE}')
print(f'   Learning rate: {LEARNING_RATE}')
print(f'   Épocas máximas: {EPOCHS}')
print(f'   Early stopping patience: {PATIENCE}')

# Transformaciones
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

print('\n📁 Cargando dataset...')
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f'   Imágenes de entrenamiento: {len(train_dataset)}')
print(f'   Imágenes de test: {len(test_dataset)}')

# Arquitectura CNN
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

print('\n🧠 Creando modelo CNN...')
model = EmotionCNN(NUM_CLASSES).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'   Parámetros totales: {total_params:,}')

# Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'   ⏳ EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

early_stopping = EarlyStopping(patience=PATIENCE)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='   Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# Entrenamiento
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print('\n' + '=' * 60)
print('🚀 INICIANDO ENTRENAMIENTO')
print('=' * 60)

for epoch in range(EPOCHS):
    print(f'\n📌 Época {epoch+1}/{EPOCHS}')
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f'   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    
    scheduler.step(val_loss)
    
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print('\n🛑 ¡Early stopping activado!')
        model.load_state_dict(early_stopping.best_model)
        break

print('\n' + '=' * 60)
print('✅ ENTRENAMIENTO COMPLETADO')
print('=' * 60)

# Guardar gráficas
print('\n📊 Generando gráficas...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
axes[0].set_title('Pérdida durante Entrenamiento', fontsize=14)
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_acc'], label='Train', linewidth=2)
axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
axes[1].set_title('Precisión durante Entrenamiento', fontsize=14)
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/training_history.png', dpi=150)
print('   ✓ Guardada: models/training_history.png')

# Matriz de confusión
print('\n📈 Generando matriz de confusión...')
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='   Evaluando'):
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTIONS_ES, yticklabels=EMOTIONS_ES)
plt.title('Matriz de Confusión', fontsize=14)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
print('   ✓ Guardada: models/confusion_matrix.png')

# Reporte
print('\n📋 Reporte de Clasificación:')
print('=' * 60)
print(classification_report(all_labels, all_preds, target_names=EMOTIONS_ES))

# Guardar modelo
print('\n💾 Guardando modelo...')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'emotions': EMOTIONS,
    'emotions_es': EMOTIONS_ES,
    'history': history,
    'accuracy': history['val_acc'][-1]
}, MODEL_PATH)

print(f'   ✓ Modelo guardado en: {MODEL_PATH}')
print(f'\n🎯 Precisión final: {history["val_acc"][-1]:.2f}%')
print('\n' + '=' * 60)
print('🎭 ¡Modelo listo para usar con webcam!')
print('=' * 60)
