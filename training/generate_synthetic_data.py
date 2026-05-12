"""
Sentetik Veri Oluşturucu
========================
Gerçek kamera verisi toplamadan önce modeli test etmek için
sentetik (yapay) el landmark verisi oluşturur.

Her gesture için tipik parmak pozisyonlarını simüle eder.
"""

import csv
import os
import numpy as np
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUTPUT_FILE = os.path.join(DATA_DIR, "landmarks.csv")
os.makedirs(DATA_DIR, exist_ok=True)

SAMPLES_PER_GESTURE = 300
NOISE_LEVEL = 0.08

# 21 landmark varsayılan pozisyonları (normalize edilmiş)
# Her landmark: (x, y, z) — bilek orijinli
BASE_HAND = np.array([
    [0.0, 0.0, 0.0],       # 0: Bilek
    [-0.05, -0.1, -0.01],  # 1: Başparmak CMC
    [-0.1, -0.18, -0.02],  # 2: Başparmak MCP
    [-0.14, -0.25, -0.02], # 3: Başparmak IP
    [-0.16, -0.32, -0.02], # 4: Başparmak Tip
    [-0.02, -0.35, 0.0],   # 5: İşaret MCP
    [-0.02, -0.50, 0.0],   # 6: İşaret PIP
    [-0.02, -0.58, 0.0],   # 7: İşaret DIP
    [-0.02, -0.65, 0.0],   # 8: İşaret Tip
    [0.03, -0.36, 0.0],    # 9: Orta MCP
    [0.03, -0.52, 0.0],    # 10: Orta PIP
    [0.03, -0.60, 0.0],    # 11: Orta DIP
    [0.03, -0.67, 0.0],    # 12: Orta Tip
    [0.08, -0.34, 0.01],   # 13: Yüzük MCP
    [0.08, -0.48, 0.01],   # 14: Yüzük PIP
    [0.08, -0.55, 0.01],   # 15: Yüzük DIP
    [0.08, -0.61, 0.01],   # 16: Yüzük Tip
    [0.13, -0.30, 0.02],   # 17: Serçe MCP
    [0.13, -0.40, 0.02],   # 18: Serçe PIP
    [0.13, -0.46, 0.02],   # 19: Serçe DIP
    [0.13, -0.51, 0.02],   # 20: Serçe Tip
])


def generate_open_palm(base, noise):
    """Açık avuç — tüm parmaklar açık"""
    hand = base.copy()
    hand += np.random.randn(*hand.shape) * noise
    return hand


def generate_pinch(base, noise):
    """Kıstırma — başparmak ve işaret parmağı birbirine yakın"""
    hand = base.copy()
    # Başparmak ve işaret parmağı uçlarını yakınlaştır
    mid = (hand[4] + hand[8]) / 2
    hand[4] = mid + np.random.randn(3) * 0.01
    hand[8] = mid + np.random.randn(3) * 0.01
    hand[3] = (hand[2] + hand[4]) / 2
    hand[7] = (hand[6] + hand[8]) / 2
    # Diğer parmakları biraz kapat
    for i in [12, 16, 20]:
        hand[i] = hand[i-3] + (hand[i] - hand[i-3]) * 0.5
    hand += np.random.randn(*hand.shape) * noise
    return hand


def generate_fist(base, noise):
    """Yumruk — tüm parmaklar kapalı"""
    hand = base.copy()
    # Parmak uçlarını MCP'ye yaklaştır
    for tip, pip, dip, mcp in [(4,3,2,1), (8,7,6,5), (12,11,10,9), (16,15,14,13), (20,19,18,17)]:
        hand[tip] = hand[mcp] + np.array([0, 0.05, 0.03])
        hand[dip] = (hand[mcp] + hand[tip]) / 2
        hand[pip] = hand[mcp] + (hand[tip] - hand[mcp]) * 0.3
    hand += np.random.randn(*hand.shape) * noise
    return hand


def generate_point(base, noise):
    """İşaret — sadece işaret parmağı açık"""
    hand = generate_fist(base, noise * 0.3)  # Önce yumruk yap
    # İşaret parmağını aç
    hand[5:9] = base[5:9].copy()
    hand += np.random.randn(*hand.shape) * noise * 0.5
    return hand


def generate_none(noise):
    """Nötr — rastgele gürültü (el yok simülasyonu)"""
    return np.random.randn(21, 3) * 0.1


def normalize(hand):
    """Bilek orijinli normalize et"""
    hand = hand - hand[0]
    max_val = np.max(np.abs(hand))
    if max_val > 0:
        hand = hand / max_val
    return hand.flatten()


def calc_distances(hand_flat):
    """Parmak uçları arası mesafeler"""
    hand = hand_flat.reshape(21, 3)
    tips = [4, 8, 12, 16, 20]
    distances = []
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            d = np.linalg.norm(hand[tips[i]] - hand[tips[j]])
            distances.append(round(float(d), 6))
    return distances


def create_header():
    header = ["gesture_id", "gesture_name", "timestamp"]
    for i in range(21):
        header.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])
    tips = ["thumb", "index", "middle", "ring", "pinky"]
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            header.append(f"dist_{tips[i]}_{tips[j]}")
    return header


def main():
    print("Sentetik veri oluşturuluyor...")
    
    generators = {
        0: ("open_palm", generate_open_palm),
        1: ("pinch", generate_pinch),
        2: ("fist", generate_fist),
        3: ("point", generate_point),
    }
    
    header = create_header()
    rows = []
    
    for gid, (gname, gen_func) in generators.items():
        for i in range(SAMPLES_PER_GESTURE):
            if gname == "none":
                hand = generate_none(NOISE_LEVEL)
            else:
                hand = gen_func(BASE_HAND, NOISE_LEVEL)
            
            normalized = normalize(hand)
            distances = calc_distances(normalized)
            
            row = [gid, gname, datetime.now().isoformat()]
            row += [round(float(v), 6) for v in normalized]
            row += distances
            rows.append(row)
    
    # "none" sınıfı
    for i in range(SAMPLES_PER_GESTURE):
        hand = generate_none(NOISE_LEVEL)
        normalized = normalize(hand)
        distances = calc_distances(normalized)
        row = [4, "none", datetime.now().isoformat()]
        row += [round(float(v), 6) for v in normalized]
        row += distances
        rows.append(row)
    
    # Karıştır
    np.random.shuffle(rows)
    
    # Kaydet
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    total = len(rows)
    print(f"Toplam {total} örnek oluşturuldu: {OUTPUT_FILE}")
    for gid in range(5):
        count = sum(1 for r in rows if r[0] == gid)
        gname = ["open_palm", "pinch", "fist", "point", "none"][gid]
        print(f"  [{gid}] {gname}: {count}")


if __name__ == "__main__":
    main()
