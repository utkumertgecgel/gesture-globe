"""
Model Eğitim Script'i — MLP, CNN, LSTM Karşılaştırması
=====================================================
El landmark verisiyle 3 farklı derin öğrenme modelini eğitir ve karşılaştırır.

Kullanım:
  python train_models.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ============================================================
# AYARLAR
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "raw", "landmarks.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# DATASET
# ============================================================
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# MODELLER
# ============================================================

class MLPModel(nn.Module):
    """Çok Katmanlı Algılayıcı (Multi-Layer Perceptron)"""
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class CNNModel(nn.Module):
    """1D Evrişimsel Sinir Ağı — Landmark'ları (21,3) olarak yeniden şekillendirir"""
    def __init__(self, num_classes):
        super().__init__()
        # Input: (batch, 1, 21, 3+features)
        self.features = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 63) -> (batch, 3, 21) - her kanal x/y/z
        batch_size = x.size(0)
        # Sadece ilk 63 feature (landmark koordinatları) al
        landmarks = x[:, :63].reshape(batch_size, 21, 3).permute(0, 2, 1)
        features = self.features(landmarks).squeeze(-1)
        return self.classifier(features)


class LSTMModel(nn.Module):
    """LSTM — Landmark'ları sıralı olarak işler (parmak sırası)"""
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=3, hidden_size=64,
            num_layers=2, batch_first=True,
            dropout=0.3, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # bidirectional: 64*2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        landmarks = x[:, :63].reshape(batch_size, 21, 3)
        lstm_out, _ = self.lstm(landmarks)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)


# ============================================================
# EĞİTİM FONKSİYONLARI
# ============================================================

def load_data():
    """CSV'den veri yükler ve hazırlar"""
    print(f"Veri yükleniyor: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Toplam örnek: {len(df)}")
    print(f"Sınıf dağılımı:\n{df['gesture_name'].value_counts()}")
    
    # Feature ve label ayır
    feature_cols = [c for c in df.columns if c.startswith('lm') or c.startswith('dist_')]
    X = df[feature_cols].values.astype(np.float32)
    
    le = LabelEncoder()
    y = le.fit_transform(df['gesture_name'].values)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Eğitim: {len(X_train)}, Test: {len(X_test)}")
    print(f"Sınıflar: {list(le.classes_)}")
    
    return X_train, X_test, y_train, y_test, le


def train_model(model, train_loader, test_loader, model_name):
    """Tek bir modeli eğitir"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    best_acc = 0
    
    print(f"\n{'='*50}")
    print(f"  {model_name} Eğitimi Başlıyor")
    print(f"  Parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Cihaz: {DEVICE}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        # TEST
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += y_batch.size(0)
                test_correct += predicted.eq(y_batch).sum().item()
        
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        scheduler.step(avg_test_loss)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name.lower()}_best.pth"))
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f} Acc: {test_acc:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\n  {model_name} tamamlandı! Süre: {elapsed:.1f}s | En iyi test acc: {best_acc:.4f}")
    
    return history, best_acc, elapsed


def evaluate_model(model, test_loader, label_encoder, model_name):
    """Model performansını detaylı değerlendirir"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    # Classification report
    report = classification_report(all_labels, all_preds, 
                                    target_names=label_encoder.classes_, 
                                    output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return report, cm, all_preds, all_labels


def plot_results(histories, label_encoder):
    """Tüm modellerin sonuçlarını görselleştirir"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Karşılaştırması', fontsize=16, fontweight='bold')
    colors = {'MLP': '#6366f1', 'CNN': '#22c55e', 'LSTM': '#f59e0b'}
    
    # Loss grafikleri
    for name, hist in histories.items():
        axes[0][0].plot(hist['train_loss'], label=f'{name} (Train)', color=colors[name], alpha=0.7)
        axes[0][0].plot(hist['test_loss'], label=f'{name} (Test)', color=colors[name], linestyle='--')
    axes[0][0].set_title('Loss'); axes[0][0].set_xlabel('Epoch'); axes[0][0].legend(); axes[0][0].grid(True, alpha=0.3)
    
    # Accuracy grafikleri
    for name, hist in histories.items():
        axes[0][1].plot(hist['train_acc'], label=f'{name} (Train)', color=colors[name], alpha=0.7)
        axes[0][1].plot(hist['test_acc'], label=f'{name} (Test)', color=colors[name], linestyle='--')
    axes[0][1].set_title('Accuracy'); axes[0][1].set_xlabel('Epoch'); axes[0][1].legend(); axes[0][1].grid(True, alpha=0.3)
    
    # Bar chart - final accuracy
    names = list(histories.keys())
    final_accs = [histories[n]['test_acc'][-1] for n in names]
    bars = axes[1][0].bar(names, final_accs, color=[colors[n] for n in names])
    axes[1][0].set_title('Final Test Accuracy'); axes[1][0].set_ylim(0, 1)
    for bar, acc in zip(bars, final_accs):
        axes[1][0].text(bar.get_x() + bar.get_width()/2., acc + 0.01, f'{acc:.2%}', ha='center', fontweight='bold')
    
    # Boş (confusion matrix ayrı)
    axes[1][1].axis('off')
    axes[1][1].text(0.5, 0.5, 'Confusion Matrix\nayrı dosyalarda', ha='center', va='center', fontsize=12, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Grafik kaydedildi: results/model_comparison.png")


def plot_confusion_matrix(cm, labels, model_name):
    """Confusion matrix görselleştirir"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'{model_name} — Confusion Matrix')
    plt.xlabel('Tahmin'); plt.ylabel('Gerçek')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'cm_{model_name.lower()}.png'), dpi=150)
    plt.close()


# ============================================================
# ONNX EXPORT (TF.js için)
# ============================================================
def export_best_model(model, input_size, model_name):
    """En iyi modeli ONNX formatında export eder"""
    model.eval()
    dummy = torch.randn(1, input_size).to(DEVICE)
    onnx_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_best.onnx")
    torch.onnx.export(model, dummy, onnx_path, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
    print(f"  ONNX export: {onnx_path}")


# ============================================================
# ANA FONKSİYON
# ============================================================
def main():
    print("=" * 60)
    print("  GESTURE MODEL EĞİTİMİ")
    print(f"  Cihaz: {DEVICE}")
    print("=" * 60)
    
    # Veri yükle
    X_train, X_test, y_train, y_test, le = load_data()
    input_size = X_train.shape[1]
    num_classes = len(le.classes_)
    
    train_dataset = GestureDataset(X_train, y_train)
    test_dataset = GestureDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Modelleri tanımla
    models = {
        'MLP': MLPModel(input_size, num_classes),
        'CNN': CNNModel(num_classes),
        'LSTM': LSTMModel(num_classes),
    }
    
    histories = {}
    results_summary = {}
    
    for name, model in models.items():
        # Eğit
        history, best_acc, elapsed = train_model(model, train_loader, test_loader, name)
        histories[name] = history
        
        # En iyi modeli yükle ve değerlendir
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name.lower()}_best.pth"), weights_only=True))
        model.to(DEVICE)
        report, cm, _, _ = evaluate_model(model, test_loader, le, name)
        
        # Confusion matrix kaydet
        plot_confusion_matrix(cm, le.classes_, name)
        
        # Sonuçları kaydet
        param_count = sum(p.numel() for p in model.parameters())
        results_summary[name] = {
            'accuracy': best_acc,
            'params': param_count,
            'training_time': round(elapsed, 1),
            'report': report
        }
        
        print(f"\n  {name} Classification Report:")
        print(classification_report(
            [le.classes_[i] for i in range(num_classes)],
            [le.classes_[i] for i in range(num_classes)],
            target_names=le.classes_
        ))
    
    # Grafikleri oluştur
    plot_results(histories, le)
    
    # En iyi modeli export et
    best_model_name = max(results_summary, key=lambda k: results_summary[k]['accuracy'])
    best_model = models[best_model_name]
    best_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{best_model_name.lower()}_best.pth"), weights_only=True))
    best_model.to(DEVICE)
    export_best_model(best_model, input_size, best_model_name)
    
    # Sonuç özeti
    print("\n" + "=" * 60)
    print("  SONUÇ ÖZETİ")
    print("=" * 60)
    for name, res in results_summary.items():
        star = " * EN IYI" if name == best_model_name else ""
        print(f"  {name:6s} | Acc: {res['accuracy']:.4f} | Params: {res['params']:,} | Sure: {res['training_time']}s{star}")
    
    # JSON kaydet
    summary_path = os.path.join(RESULTS_DIR, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'report'} for k, v in results_summary.items()}, f, indent=2)
    
    # Label mapping kaydet (web app için)
    label_map = {int(i): name for i, name in enumerate(le.classes_)}
    with open(os.path.join(MODELS_DIR, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\n  En iyi model: {best_model_name} ({results_summary[best_model_name]['accuracy']:.2%})")
    print(f"  Sonuçlar: {RESULTS_DIR}")
    print(f"  Modeller: {MODELS_DIR}")


if __name__ == "__main__":
    main()
