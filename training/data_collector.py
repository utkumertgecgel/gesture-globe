"""
El Hareketi Veri Toplama Script'i
=================================
Bu script, kameradan el landmark'larını yakalayıp CSV'ye kaydeder.
Her gesture için farklı tuşlara basarak veri toplarsın.

Kullanım:
  python data_collector.py

Tuşlar:
  0 - "open_palm" (açık avuç - döndürme için)
  1 - "pinch" (kıstırma - zoom için)  
  2 - "fist" (yumruk - durdurma için)
  3 - "point" (işaret - ülke seçimi için)
  4 - "none" (el yok / nötr pozisyon)
  
  s - Kaydetmeyi başlat/durdur (toggle)
  q - Çıkış
  r - Son kaydedilen örneği sil (geri al)
"""

import csv
import os
import sys
import time
import copy
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime


# ============================================================
# AYARLAR
# ============================================================

# Gesture sınıfları
GESTURES = {
    0: "open_palm",     # Açık avuç → küreyi döndür
    1: "pinch",         # Başparmak + işaret parmağı → zoom
    2: "fist",          # Yumruk → durdur
    3: "point",         # İşaret parmağı → ülke seç
    4: "none",          # Nötr / el yok
}

# Kamera ayarları
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# MediaPipe ayarları
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 1

# Veri kayıt yolu
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUTPUT_FILE = os.path.join(DATA_DIR, "landmarks.csv")


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def normalize_landmarks(landmarks):
    """
    El landmark'larını normalize eder.
    - Bilek noktasını (0) orijin olarak alır
    - El boyutuna göre ölçekler (scale invariant)
    
    Bu sayede elin kameraya uzaklığı veya büyüklüğü fark etmez.
    """
    # 21 landmark × 3 koordinat = 63 değer
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    
    # Bilek noktasını (landmark 0) orijin yap
    base_x, base_y, base_z = coords[0], coords[1], coords[2]
    
    normalized = []
    for i in range(0, len(coords), 3):
        normalized.append(coords[i] - base_x)
        normalized.append(coords[i + 1] - base_y)
        normalized.append(coords[i + 2] - base_z)
    
    # Maksimum mutlak değere göre ölçekle (0-1 arası)
    max_val = max(abs(v) for v in normalized)
    if max_val > 0:
        normalized = [v / max_val for v in normalized]
    
    return normalized


def calc_finger_distances(landmarks):
    """
    Parmak uçları arası mesafeleri hesaplar.
    Pinch tespiti için özellikle başparmak-işaret parmağı mesafesi önemli.
    """
    # Landmark indeksleri
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    
    distances = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            lm1 = landmarks[tips[i]]
            lm2 = landmarks[tips[j]]
            dist = np.sqrt(
                (lm1.x - lm2.x) ** 2 + 
                (lm1.y - lm2.y) ** 2 + 
                (lm1.z - lm2.z) ** 2
            )
            distances.append(round(dist, 6))
    
    return distances  # 10 mesafe değeri


def create_csv_header():
    """CSV dosyası için başlık satırını oluşturur."""
    header = ["gesture_id", "gesture_name", "timestamp"]
    
    # 21 landmark × 3 koordinat (normalize edilmiş)
    for i in range(21):
        header.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])
    
    # 10 parmak ucu mesafesi
    tips = ["thumb", "index", "middle", "ring", "pinky"]
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            header.append(f"dist_{tips[i]}_{tips[j]}")
    
    return header


def draw_info_panel(image, gesture_counts, is_recording, current_gesture, fps):
    """Ekranda bilgi paneli çizer."""
    h, w = image.shape[:2]
    
    # Yarı saydam siyah panel (sol taraf)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (280, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Başlık
    cv2.putText(image, "EL HAREKETI VERI TOPLAMA", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(image, f"FPS: {fps:.0f}", (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Kayıt durumu
    if is_recording:
        # Yanıp sönen kırmızı nokta
        if int(time.time() * 2) % 2:
            cv2.circle(image, (20, 55), 8, (0, 0, 255), -1)
        cv2.putText(image, "KAYIT AKTIF", (35, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(image, "KAYIT DURDURULDU", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Mevcut gesture
    cv2.putText(image, f"Gesture: {current_gesture}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Ayırıcı çizgi
    cv2.line(image, (10, 105), (270, 105), (100, 100, 100), 1)
    
    # Her gesture'ın kayıt sayısı
    y = 130
    cv2.putText(image, "Toplanan Ornek Sayisi:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 25
    
    for gid, gname in GESTURES.items():
        count = gesture_counts.get(gid, 0)
        # Renk: yeterli örnek yeşil, az olan kırmızı
        color = (0, 255, 0) if count >= 100 else (0, 165, 255) if count >= 50 else (0, 0, 255)
        bar_width = min(int(count / 2), 120)  # Max 120px bar
        
        cv2.putText(image, f"[{gid}] {gname}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.rectangle(image, (150, y - 12), (150 + bar_width, y + 2), color, -1)
        cv2.putText(image, str(count), (155 + bar_width, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 22
    
    # Toplam
    total = sum(gesture_counts.values())
    cv2.line(image, (10, y), (270, y), (100, 100, 100), 1)
    y += 20
    cv2.putText(image, f"TOPLAM: {total} ornek", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Kontroller
    y += 35
    cv2.line(image, (10, y - 15), (270, y - 15), (100, 100, 100), 1)
    cv2.putText(image, "KONTROLLER:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y += 22
    controls = [
        "0-4: Gesture sec",
        "s: Kaydi baslat/durdur",
        "r: Son ornegi sil",
        "q: Cikis",
    ]
    for ctrl in controls:
        cv2.putText(image, ctrl, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        y += 18
    
    return image


def draw_landmarks_custom(image, hand_landmarks, mp_hands):
    """El landmark'larını özel stilde çizer."""
    # Bağlantı çizgileri
    connections = mp_hands.HAND_CONNECTIONS
    h, w = image.shape[:2]
    
    # Çizgileri çiz
    for connection in connections:
        start = hand_landmarks.landmark[connection[0]]
        end = hand_landmarks.landmark[connection[1]]
        
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        
        cv2.line(image, start_point, end_point, (0, 200, 0), 2)
    
    # Noktaları çiz
    for i, lm in enumerate(hand_landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        
        # Parmak uçları daha büyük
        if i in [4, 8, 12, 16, 20]:
            cv2.circle(image, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(image, (cx, cy), 8, (255, 255, 255), 2)
        # Bilek
        elif i == 0:
            cv2.circle(image, (cx, cy), 7, (255, 0, 0), -1)
            cv2.circle(image, (cx, cy), 7, (255, 255, 255), 2)
        else:
            cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)
    
    return image


# ============================================================
# ANA FONKSİYON
# ============================================================

def main():
    print("=" * 60)
    print("  EL HAREKETİ VERİ TOPLAMA SİSTEMİ")
    print("  İleri Derin Öğrenme - Final Projesi")
    print("=" * 60)
    print()
    
    # Veri klasörünü oluştur
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Mevcut veri var mı kontrol et
    existing_data = []
    gesture_counts = {gid: 0 for gid in GESTURES}
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                existing_data.append(row)
                gid = int(row[0])
                gesture_counts[gid] = gesture_counts.get(gid, 0) + 1
        total = sum(gesture_counts.values())
        print(f"  Mevcut veri bulundu: {total} örnek")
        for gid, gname in GESTURES.items():
            print(f"    [{gid}] {gname}: {gesture_counts[gid]} örnek")
    else:
        print("  Yeni veri dosyası oluşturulacak.")
    
    print()
    print("  Kamera başlatılıyor...")
    
    # MediaPipe Hands başlat
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    
    # Kamera başlat
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
        print("  HATA: Kamera açılamadı!")
        print("  Kameranın bağlı olduğundan emin ol.")
        sys.exit(1)
    
    print("  Kamera başarıyla açıldı!")
    print()
    print("  Kontroller:")
    print("    0-4 : Gesture seç")
    print("    s   : Kaydetmeyi başlat/durdur")
    print("    r   : Son kaydı sil")
    print("    q   : Çıkış")
    print()
    
    # Durum değişkenleri
    is_recording = False
    current_gesture_id = 0
    current_gesture_name = GESTURES[0]
    all_data = existing_data.copy()
    fps = 0
    prev_time = time.time()
    frame_count = 0
    
    # CSV header'ı hazırla
    csv_header = create_csv_header()
    
    while True:
        # Frame oku
        ret, frame = cap.read()
        if not ret:
            print("  HATA: Frame okunamadı!")
            break
        
        # Ayna efekti (daha doğal hissettirmek için)
        frame = cv2.flip(frame, 1)
        
        # FPS hesapla
        frame_count += 1
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time
        
        # BGR → RGB (MediaPipe RGB bekler)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # El tespiti
        results = hands.process(rgb_frame)
        
        # Landmark'ları çiz ve kaydet
        hand_detected = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_detected = True
                
                # Landmark'ları çiz
                frame = draw_landmarks_custom(frame, hand_landmarks, mp_hands)
                
                # Kayıt modunda ise veriyi kaydet
                if is_recording:
                    # Normalize edilmiş landmark'lar
                    normalized = normalize_landmarks(hand_landmarks.landmark)
                    
                    # Parmak mesafeleri
                    distances = calc_finger_distances(hand_landmarks.landmark)
                    
                    # Satır oluştur
                    row = [
                        current_gesture_id,
                        current_gesture_name,
                        datetime.now().isoformat()
                    ] + [round(v, 6) for v in normalized] + distances
                    
                    all_data.append(row)
                    gesture_counts[current_gesture_id] = gesture_counts.get(current_gesture_id, 0) + 1
        
        # El tespit durumu göster
        if not hand_detected:
            cv2.putText(frame, "EL TESPIT EDILEMEDI", 
                       (frame.shape[1] // 2 - 120, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Bilgi panelini çiz
        frame = draw_info_panel(frame, gesture_counts, is_recording, 
                               f"[{current_gesture_id}] {current_gesture_name}", fps)
        
        # Ekranda göster
        cv2.imshow("El Hareketi Veri Toplama", frame)
        
        # Tuş girişi
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Çıkış
            break
        
        elif key == ord('s'):
            # Kaydetmeyi başlat/durdur
            is_recording = not is_recording
            status = "BASLADI" if is_recording else "DURDU"
            print(f"  Kayıt {status} - Gesture: [{current_gesture_id}] {current_gesture_name}")
        
        elif key == ord('r'):
            # Son kaydı sil
            if all_data:
                removed = all_data.pop()
                gid = int(removed[0])
                gesture_counts[gid] = max(0, gesture_counts.get(gid, 0) - 1)
                print(f"  Son kayıt silindi (gesture: {removed[1]})")
        
        elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
            # Gesture seç
            gid = key - ord('0')
            if gid in GESTURES:
                current_gesture_id = gid
                current_gesture_name = GESTURES[gid]
                print(f"  Gesture değiştirildi: [{gid}] {current_gesture_name}")
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # Verileri kaydet
    if all_data:
        print()
        print(f"  Veriler kaydediliyor: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(all_data)
        
        total = sum(gesture_counts.values())
        print(f"  Toplam {total} örnek kaydedildi!")
        print()
        for gid, gname in GESTURES.items():
            count = gesture_counts.get(gid, 0)
            status = "✓" if count >= 100 else "△" if count >= 50 else "✗"
            print(f"    {status} [{gid}] {gname}: {count} örnek")
        
        print()
        if all(gesture_counts.get(gid, 0) >= 100 for gid in GESTURES):
            print("  ✅ Tüm gesture'lar için yeterli veri toplandı!")
        else:
            print("  ⚠ Bazı gesture'lar için daha fazla veri gerekebilir (hedef: 100+)")
    else:
        print("  Hiç veri kaydedilmedi.")
    
    print()
    print("  Program sonlandırıldı.")


if __name__ == "__main__":
    main()
