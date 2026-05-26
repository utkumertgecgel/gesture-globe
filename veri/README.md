# Veri Seti Bilgileri

## Veri Seti Özeti

Bu projede kullanılan veri seti, `web-app/collect.html` arayüzü kullanılarak elle toplanmıştır.
Harici bir veri seti kullanılmamıştır; tüm veriler projeyle birlikte gelen CSV dosyasında mevcuttur.

## Veri Dosyası

```
training/data/raw/landmarks.csv
```

## Format

Her satır bir el hareketi örneğini temsil eder:
- **73 sayısal özellik:** 63 normalize landmark koordinatı (21 nokta × 3 eksen) + 10 parmak ucu mesafesi
- **1 etiket sütunu:** `label` (fist / open_palm / pinch)

## Sınıf Dağılımı

| Sınıf | Örnek Sayısı |
|-------|-------------|
| pinch | 138 |
| open_palm | 130 |
| fist | 118 |
| **Toplam** | **386** |

## Nasıl Toplanır?

1. `web-app/` klasöründe `python -m http.server 8080` komutunu çalıştırın
2. Tarayıcıda `http://localhost:8080/collect.html` adresine gidin
3. Her gesture için butonlara tıklayarak veri toplayın
4. "CSV İndir" butonu ile veriyi indirip `training/data/raw/landmarks.csv` olarak kaydedin

## Lisans

Veri seti özgün olup proje kapsamında toplanmıştır. Eğitim amaçlı kullanım serbesttir.
