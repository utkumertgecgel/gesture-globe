# GestureGlobe — El Hareketleriyle 3D Dünya Küresi Kontrolü

Tekrar geliştirilecek.!

**İleri Derin Öğrenme Final Projesi** | İstanbul Gedik Üniversitesi — Bilgisayar Mühendisliği

Kameradan gerçek zamanlı el hareketlerini tespit edip, eğitilmiş bir CNN modeli ile sınıflandırarak tarayıcıda 3D dünya küresini kontrol etmeyi sağlayan uçtan uca bir derin öğrenme projesidir.

---

## Proje Özeti

Bu proje iki ana bileşenden oluşur:

1. **Derin Öğrenme Pipeline (Python/PyTorch)** — MediaPipe ile elde edilen el landmark verisinden gesture sınıflandırma modellerinin eğitimi (MLP, 1D-CNN, LSTM karşılaştırması)
2. **Web Uygulaması (JavaScript)** — Three.js ile renderlanan 3D dünya küresinin, eğitilmiş CNN modelinin tarayıcıda ONNX Runtime ile çalıştırılmasıyla gerçek zamanlı kontrolü

---

## Desteklenen El Hareketleri

| Hareket | Açıklama | 3D Aksiyon |
|---------|----------|------------|
| Açık Avuç | Avucu aç, eli hareket ettir | Küreyi döndür |
| Kıstırma | Başparmak + işaret yaklaştır/uzaklaştır | Yakınlaştır / Uzaklaştır |
| Yumruk | Yumruk yap | Durdur (kilitle) |

---

## Sistem Mimarisi

```
Kamera Görüntüsü
      |
      v
MediaPipe Hands (21 landmark noktası)
      |
      v
Ön İşleme (Normalizasyon + Mesafe Hesaplama)
      |
      v
CNN Modeli (ONNX Runtime Web)  <--  Eğitilmiş PyTorch modeli
      |
      v
Gesture Sınıflandırma (open_palm / pinch / fist / none)
      |
      v
Globe Controller (Three.js)
      |
      v
3D Dünya Küresi Etkileşimi
```

### Veri Akışı

1. **MediaPipe Hands** tarayıcıda her karede elin 21 landmark noktasını (x, y, z) tespit eder
2. **Ön İşleme** modülü bu 21 noktayı bilek orijinli normalize eder ve 10 parmak ucu mesafesi hesaplar (toplam 73 özellik)
3. **CNN Modeli** bu 73 boyutlu vektörü ONNX Runtime ile sınıflandırır
4. **Yumuşatma Katmanı** geçmiş karelerdeki tahminleri çoğunluk oylamasıyla filtreler (titreşimi önler)
5. **Globe Controller** sonuca göre küreyi döndürür, yakınlaştırır veya kilitler

---

## Teknoloji Yığını

### Model Eğitimi (Python)

| Teknoloji | Kullanım |
|-----------|----------|
| PyTorch | Derin öğrenme çerçevesi (MLP, CNN, LSTM) |
| MediaPipe | El landmark tespiti (21 nokta x 3 koordinat) |
| scikit-learn | Eğitim/test bölme, LabelEncoder, metrik hesaplama |
| pandas | Veri yükleme ve işlem |
| matplotlib / seaborn | Karmaşıklık matrisi ve grafik oluşturma |
| ONNX | Model dışa aktarma (PyTorch → ONNX) |

### Web Uygulaması (JavaScript)

| Teknoloji | Kullanım |
|-----------|----------|
| Three.js (r128) | WebGL 3D küre renderlama ve uzay sahne yönetimi |
| MediaPipe Hands JS | Tarayıcıda gerçek zamanlı el takibi |
| ONNX Runtime Web | Eğitilmiş CNN modelini tarayıcıda çalıştırma |
| Vanilla JS / CSS | Arayüz, animasyonlar ve panel yönetimi |
| Google Fonts (Inter, Outfit) | Modern tipografi |

---

## Proje Yapısı

```
hand-gesture-3d-globe/
|
|-- training/                        # Python - Model Eğitimi
|   |-- train_models.py              # Ana eğitim betiği (MLP, CNN, LSTM)
|   |-- data/
|   |   |-- raw/
|   |       |-- landmarks.csv        # Toplanmış el landmark verisi (386 örnek)
|   |-- models/
|   |   |-- cnn_best.onnx            # En iyi model (CNN, ONNX formatı)
|   |   |-- cnn_best.pth             # CNN PyTorch kontrol noktası
|   |   |-- mlp_best.pth             # MLP PyTorch kontrol noktası
|   |   |-- lstm_best.pth            # LSTM PyTorch kontrol noktası
|   |   |-- label_map.json           # Sınıf indeks → isim eşleşmesi
|   |-- results/
|       |-- model_comparison.png     # Eğitim/test grafikleri
|       |-- cm_cnn.png               # CNN Karmaşıklık Matrisi
|       |-- cm_mlp.png               # MLP Karmaşıklık Matrisi
|       |-- cm_lstm.png              # LSTM Karmaşıklık Matrisi
|       |-- results_summary.json     # Sayısal sonuç özeti
|
|-- web-app/                         # JavaScript - Web Uygulaması
|   |-- index.html                   # Ana sayfa
|   |-- collect.html                 # Veri toplama arayüzü
|   |-- css/
|   |   |-- styles.css               # Karanlık mod uzay teması
|   |-- js/
|   |   |-- globe-controller.js      # Three.js 3D küre ve sahne yönetimi
|   |   |-- hand-tracker.js          # MediaPipe el takibi ve kamera yönetimi
|   |   |-- gesture-classifier.js    # ONNX model çıkarımı + kural tabanlı yedek
|   |   |-- ui-overlay.js            # Arayüz panelleri, güven çubukları
|   |-- models/
|       |-- cnn_best.onnx            # Web için ONNX modeli
|       |-- label_map.json           # Sınıf haritası
|
|-- docs/                            # Rapor ve belgeler
|-- README.md
```

---

## Model Eğitimi

### Veri Toplama

El landmark verileri `web-app/collect.html` arayüzü üzerinden toplanmıştır. Her bir örnek şu şekilde oluşturulur:

1. MediaPipe Hands 21 landmark noktasının (x, y, z) koordinatlarını saptar
2. Bilek noktası (landmark 0) orijin yapılarak tüm koordinatlar normalize edilir
3. Maksimum mutlak değere bölünür ([-1, 1] aralığına ölçekleme)
4. 5 parmak ucunun birbirine olan 10 mesafesi hesaplanır
5. Sonuç: 63 normalize koordinat + 10 mesafe = **73 boyutlu özellik vektörü**

**Toplanan veri:** 386 örnek (yumruk: 118, açık avuç: 130, kıstırma: 138)

### Eğitilen Modeller

Üç farklı derin öğrenme mimarisi eğitilmiş ve karşılaştırılmıştır:

#### MLP (Çok Katmanlı Algılayıcı)
- **Mimari:** Linear(73→128) → BatchNorm → ReLU → Dropout(0.3) → Linear(128→64) → BatchNorm → ReLU → Dropout(0.3) → Linear(64→32) → ReLU → Linear(32→3)
- **Parametre sayısı:** 20.291

#### 1D-CNN (1 Boyutlu Evrişimsel Sinir Ağı)
- **Mimari:** Landmark koordinatları (21, 3) olarak yeniden şekillendirilerek 1D evrişim uygulanır
- Conv1d(3→64, k=3) → BN → ReLU → Conv1d(64→128, k=3) → BN → ReLU → AdaptiveAvgPool1d → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→3)
- **Parametre sayısı:** 34.179

#### LSTM (Uzun Kısa Süreli Bellek)
- **Mimari:** 21 landmark sırasıyla işlenir (parmak sırası)
- Çift Yönlü LSTM(giriş=3, gizli=64, katman=2) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→3)
- **Parametre sayısı:** 143.107

### Eğitim Sonuçları

| Model | Test Doğruluğu | Parametre Sayısı | Eğitim Süresi |
|-------|----------------|------------------|---------------|
| MLP | %98,72 | 20.291 | 8,9 sn |
| **1D-CNN** | **%100,00** | **34.179** | **7,1 sn** |
| LSTM | %96,15 | 143.107 | 7,8 sn |

**En iyi model:** 1D-CNN (%100 test doğruluğuyla)

CNN modeli %100 test doğruluğuna ulaşarak en başarılı model olmuş ve ONNX formatında web uygulamasına aktarılmıştır. CNN'in başarısı, el landmark'larının mekânsal yapısını (x, y, z kanalları) etkin bir şekilde yakalayabilmesinden kaynaklanmaktadır.

### Eğitim Yapılandırması

- **Eniyileyici:** Adam (öğrenme hızı=0,001)
- **Zamanlayıcı:** ReduceLROnPlateau (sabır=10, çarpan=0,5)
- **Toplu iş boyutu:** 32
- **Dönem sayısı:** 100
- **Eğitim/Test bölme:** %80/%20 (tabakalı)
- **Donanım:** CUDA (GPU)

---

## Kurulum ve Çalıştırma

### Gereksinimler

**Model Eğitimi için (isteğe bağlı):**
- Python 3.10+
- PyTorch (CUDA destekli)
- pandas, scikit-learn, matplotlib, seaborn

```bash
pip install -r requirements.txt
```

**Web Uygulaması için:**
- Modern bir web tarayıcısı (Chrome, Edge, Firefox)
- Web kamerası
- Python (yalnızca yerel sunucu için)

### Web Uygulamasını Çalıştırma

```bash
cd web-app
python -m http.server 8080
```

Tarayıcıda `http://localhost:8080` adresini açın. Uygulama sırasıyla:
1. Three.js sahnesini ve 3D küreyi oluşturur
2. Kamera erişim izni ister
3. MediaPipe Hands modelini yükler
4. CNN ONNX modelini yükler
5. Gerçek zamanlı el takibi ve gesture sınıflandırma başlar

### Veri Toplama (İsteğe Bağlı)

Kendi el verinizi toplamak isterseniz:

```bash
cd web-app
python -m http.server 8080
```

Tarayıcıda `http://localhost:8080/collect.html` adresini açın. Her gesture için en az 100 örnek toplamanız önerilir.

### Modeli Yeniden Eğitme (İsteğe Bağlı)

```bash
cd training
python train_models.py
```

Eğitim tamamlandığında en iyi model otomatik olarak `models/cnn_best.onnx` dosyasına aktarılır. Bu dosyayı `web-app/models/` klasörüne kopyalayın.

---

## Web Uygulaması Ayrıntıları

### 3D Sahne (globe-controller.js)
- Three.js ile WebGL tabanlı 3D dünya küresi
- Yüksek çözünürlüklü dünya dokusu ve atmosfer efekti
- Yıldızlı uzay arka planı
- Otomatik döndürme ve kullanıcı kontrollü döndürme/yakınlaştırma

### El Takibi (hand-tracker.js)
- MediaPipe Hands ile 21 landmark noktası tespiti
- Akıllı kamera seçimi (sanal kameraları atlar, PC web kamerasını tercih eder)
- Tuval üzerine el iskeleti çizimi ve hata ayıklama katmanı

### Gesture Sınıflandırma (gesture-classifier.js)
- **Birincil:** ONNX Runtime Web ile CNN model çıkarımı
- **Yedek:** Kural tabanlı karar ağacı (model yüklenemezse devreye girer)
- Çift katmanlı yumuşatma: geçmiş tabanlı çoğunluk oyu + kararlılık filtresi
- Güven eşiği: Düşük güvenli tahminlerde "yok" döndürme

### Arayüz (ui-overlay.js)
- Animasyonlu yükleme ekranı
- Gerçek zamanlı gesture gösterge paneli
- Güven çubuğu grafikleri
- Hareket rehberi paneli
- FPS sayacı

---

## Geliştirici

- **Utku Mert Geçgel** — İstanbul Gedik Üniversitesi, Bilgisayar Mühendisliği

---

## Lisans

Bu proje eğitim amacıyla geliştirilmiştir.
