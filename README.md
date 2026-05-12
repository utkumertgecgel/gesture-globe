# 🌍 GestureGlobe — El Hareketleriyle 3D Dünya Küresini Kontrol Etme

**İleri Derin Öğrenme Final Projesi** | Bilgisayar Mühendisliği 4. Sınıf

Kameradan el hareketlerini tespit edip, bu hareketlerle tarayıcıda 3D dünya küresini döndürme, yakınlaştırma ve etkileşimli kontrol etme projesi.

## 🎯 Proje Özeti

Bu proje iki ana bileşenden oluşur:

1. **Derin Öğrenme Pipeline (Python/PyTorch)** — MediaPipe el landmark verisinden gesture sınıflandırma modeli eğitimi (MLP, CNN, LSTM karşılaştırması)
2. **Web Uygulaması (JavaScript)** — Three.js ile 3D dünya küresinin el hareketleriyle gerçek zamanlı kontrolü

## ✋ Desteklenen El Hareketleri

| Hareket | Açıklama | 3D Aksiyon |
|---------|----------|------------|
| ✋ Açık Avuç | Avucu aç, sürükle | Küreyi döndür |
| 🤏 Kıstırma | Başparmak + işaret yaklaştır/uzaklaştır | Zoom in/out |
| ✊ Yumruk | Yumruk yap | Durdur |
| ☝️ İşaret | Parmakla göster | Ülke seç |

## 🛠️ Teknoloji Yığını

### Model Eğitimi (Python)
- **PyTorch** — Derin öğrenme framework
- **MediaPipe** — El landmark tespiti (21 nokta)
- **scikit-learn** — Metrik hesaplama
- **OpenCV** — Görüntü işleme

### Web Uygulaması (JavaScript)
- **Three.js** — WebGL 3D küre rendering
- **MediaPipe Hands JS** — Tarayıcıda el takibi
- **TensorFlow.js** — Eğitilmiş modelin tarayıcıda çalıştırılması

## 📁 Proje Yapısı

```
hand-gesture-3d-globe/
├── training/                    # Python - Model Eğitimi
│   ├── data_collector.py        # Veri toplama script'i
│   ├── data/raw/                # Ham landmark verisi
│   └── models/                  # Eğitilmiş modeller
├── web-app/                     # JavaScript - Web Uygulaması
│   ├── index.html               # Ana sayfa
│   ├── css/styles.css           # Dark mode uzay teması
│   └── js/
│       ├── globe-controller.js  # Three.js 3D küre
│       ├── hand-tracker.js      # MediaPipe el takibi
│       ├── gesture-classifier.js # Gesture sınıflandırma
│       └── ui-overlay.js        # UI yönetimi
└── docs/                        # Rapor ve dokümanlar
```

## 🚀 Çalıştırma

### Web Uygulaması
```bash
cd web-app
python -m http.server 8080
```
Tarayıcıda `http://localhost:8080` aç.

### Veri Toplama
```bash
cd training
python data_collector.py
```

## 📊 Model Karşılaştırması

| Model | Accuracy | Parametre | Eğitim Süresi |
|-------|----------|-----------|---------------|
| MLP | — | — | — |
| 1D-CNN | — | — | — |
| LSTM | — | — | — |

> Sonuçlar model eğitimi tamamlandıktan sonra güncellenecektir.

## 👤 Geliştirici

- **Utku Mert Geçgel**

## 📝 Lisans

Bu proje eğitim amaçlıdır.
