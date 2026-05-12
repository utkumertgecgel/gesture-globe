/**
 * GestureClassifier — El Hareketlerini Sınıflandırır
 * 
 * İki mod destekler:
 * 1. Rule-based (kural tabanlı) — TF.js modeli yüklenmeden önce
 * 2. Model-based — Eğitilmiş TF.js modeli ile (PyTorch'tan export)
 * 
 * Gesture'lar:
 *   open_palm  → Küreyi döndür
 *   pinch      → Zoom in/out
 *   fist       → Durdur
 *   point      → Ülke seç
 *   none       → Hiçbir şey yapma
 */

class GestureClassifier {
    constructor() {
        this.model = null;
        this.useModel = false; // TF.js modeli yüklenince true olur
        this.gestureNames = ['open_palm', 'pinch', 'fist', 'point', 'none'];
        this.gestureIcons = {
            'open_palm': '✋',
            'pinch': '🤏',
            'fist': '✊',
            'point': '☝️',
            'none': '—'
        };
        this.gestureTR = {
            'open_palm': 'Açık Avuç',
            'pinch': 'Kıstırma',
            'fist': 'Yumruk',
            'point': 'İşaret',
            'none': 'Bekleniyor'
        };

        // Smoothing: Son N frame'in sonuçlarını tut
        this.historySize = 5;
        this.history = [];

        // Önceki frame'in landmark pozisyonları (hareket yönü için)
        this.prevPalmCenter = null;
        this.prevPinchDist = null;
        
        // Hareket verileri
        this.palmDeltaX = 0;
        this.palmDeltaY = 0;
        this.pinchDelta = 0;
    }

    async loadModel(modelPath) {
        try {
            // TensorFlow.js modelini yükle
            if (typeof tf !== 'undefined') {
                this.model = await tf.loadLayersModel(modelPath);
                this.useModel = true;
                console.log('TF.js modeli yüklendi:', modelPath);
            }
        } catch (e) {
            console.warn('TF.js modeli yüklenemedi, rule-based mod kullanılacak:', e.message);
            this.useModel = false;
        }
    }

    /**
     * Ana sınıflandırma fonksiyonu
     * @param {Array} landmarks - MediaPipe 21 landmark
     * @returns {Object} { gesture, confidence, allConfidences, palmDelta, pinchDelta }
     */
    classify(landmarks) {
        if (!landmarks) {
            this.prevPalmCenter = null;
            this.prevPinchDist = null;
            this.palmDeltaX = 0;
            this.palmDeltaY = 0;
            this.pinchDelta = 0;
            return {
                gesture: 'none',
                confidence: 1.0,
                allConfidences: { open_palm: 0, pinch: 0, fist: 0, point: 0, none: 1 },
                palmDeltaX: 0,
                palmDeltaY: 0,
                pinchDelta: 0
            };
        }

        let result;
        if (this.useModel && this.model) {
            result = this.classifyWithModel(landmarks);
        } else {
            result = this.classifyRuleBased(landmarks);
        }

        // Hareket verilerini hesapla
        this.calculateMotion(landmarks);

        result.palmDeltaX = this.palmDeltaX;
        result.palmDeltaY = this.palmDeltaY;
        result.pinchDelta = this.pinchDelta;

        // Smoothing uygula
        this.history.push(result.gesture);
        if (this.history.length > this.historySize) {
            this.history.shift();
        }
        result.gesture = this.getMostFrequent(this.history);

        return result;
    }

    /**
     * Kural tabanlı gesture sınıflandırma
     * MediaPipe landmark'larından geometrik kurallarla gesture belirler
     */
    classifyRuleBased(landmarks) {
        const features = this.extractFeatures(landmarks);
        const confidences = { open_palm: 0, pinch: 0, fist: 0, point: 0, none: 0 };

        // ---- PINCH tespiti ----
        // Başparmak ucu (4) ile işaret parmağı ucu (8) arası mesafe
        if (features.thumbIndexDist < 0.055) {
            confidences.pinch = 0.9 - features.thumbIndexDist * 5;
        }

        // ---- FIST tespiti ----
        // Tüm parmak uçları avuç merkezine yakınsa = yumruk
        if (features.allFingersClosed && features.avgFingerDist < 0.15) {
            confidences.fist = 0.85;
        }

        // ---- POINT tespiti ----
        // Sadece işaret parmağı açık, diğerleri kapalı
        if (features.indexExtended && !features.middleExtended && 
            !features.ringExtended && !features.pinkyExtended) {
            confidences.point = 0.85;
        }

        // ---- OPEN PALM tespiti ----
        // Çoğu parmak açık
        const openCount = [features.indexExtended, features.middleExtended, 
                          features.ringExtended, features.pinkyExtended]
                          .filter(Boolean).length;
        if (openCount >= 3 && features.thumbIndexDist > 0.08) {
            confidences.open_palm = 0.7 + openCount * 0.05;
        }

        // En yüksek confidence'ı bul
        let maxGesture = 'none';
        let maxConf = 0.3; // Minimum eşik
        for (const [gesture, conf] of Object.entries(confidences)) {
            if (conf > maxConf) {
                maxConf = conf;
                maxGesture = gesture;
            }
        }

        // None confidence
        if (maxGesture === 'none') {
            confidences.none = 0.5;
        }

        // Normalize
        const total = Object.values(confidences).reduce((a, b) => a + b, 0) || 1;
        for (const key of Object.keys(confidences)) {
            confidences[key] /= total;
        }

        return {
            gesture: maxGesture,
            confidence: maxConf,
            allConfidences: confidences
        };
    }

    /**
     * TF.js model ile sınıflandırma
     */
    classifyWithModel(landmarks) {
        // Landmark'ları normalize et
        const normalized = this.normalizeLandmarks(landmarks);
        
        // Parmak mesafelerini ekle
        const distances = this.calcFingerDistances(landmarks);
        const input = [...normalized, ...distances];

        // Model inference
        const tensor = tf.tensor2d([input]);
        const prediction = this.model.predict(tensor);
        const probs = prediction.dataSync();
        tensor.dispose();
        prediction.dispose();

        // En yüksek olasılıklı gesture
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        const confidences = {};
        this.gestureNames.forEach((name, i) => {
            confidences[name] = i < probs.length ? probs[i] : 0;
        });

        return {
            gesture: this.gestureNames[maxIdx],
            confidence: probs[maxIdx],
            allConfidences: confidences
        };
    }

    /**
     * Landmark'lardan geometrik özellikler çıkarır
     */
    extractFeatures(landmarks) {
        // Parmak ucu indeksleri
        const THUMB_TIP = 4, THUMB_IP = 3, THUMB_MCP = 2;
        const INDEX_TIP = 8, INDEX_PIP = 6, INDEX_MCP = 5;
        const MIDDLE_TIP = 12, MIDDLE_PIP = 10, MIDDLE_MCP = 9;
        const RING_TIP = 16, RING_PIP = 14, RING_MCP = 13;
        const PINKY_TIP = 20, PINKY_PIP = 18, PINKY_MCP = 17;
        const WRIST = 0;

        // Mesafe hesaplama yardımcı fonksiyonu
        const dist = (a, b) => Math.sqrt(
            Math.pow(landmarks[a].x - landmarks[b].x, 2) +
            Math.pow(landmarks[a].y - landmarks[b].y, 2)
        );

        // Başparmak-işaret parmağı mesafesi
        const thumbIndexDist = dist(THUMB_TIP, INDEX_TIP);

        // Parmak açık mı kontrolü (parmak ucu, PIP'ten daha uzakta mı)
        const wrist = landmarks[WRIST];
        const isExtended = (tip, pip) => {
            const tipDist = Math.sqrt(
                Math.pow(landmarks[tip].x - wrist.x, 2) +
                Math.pow(landmarks[tip].y - wrist.y, 2)
            );
            const pipDist = Math.sqrt(
                Math.pow(landmarks[pip].x - wrist.x, 2) +
                Math.pow(landmarks[pip].y - wrist.y, 2)
            );
            return tipDist > pipDist;
        };

        const indexExtended = isExtended(INDEX_TIP, INDEX_PIP);
        const middleExtended = isExtended(MIDDLE_TIP, MIDDLE_PIP);
        const ringExtended = isExtended(RING_TIP, RING_PIP);
        const pinkyExtended = isExtended(PINKY_TIP, PINKY_PIP);

        // Avuç merkezi
        const palmCenter = {
            x: (landmarks[0].x + landmarks[5].x + landmarks[17].x) / 3,
            y: (landmarks[0].y + landmarks[5].y + landmarks[17].y) / 3
        };

        // Parmak uçlarının avuç merkezine ortalama mesafesi
        const fingerTips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP];
        const fingerDists = fingerTips.map(tip => 
            Math.sqrt(
                Math.pow(landmarks[tip].x - palmCenter.x, 2) +
                Math.pow(landmarks[tip].y - palmCenter.y, 2)
            )
        );
        const avgFingerDist = fingerDists.reduce((a, b) => a + b, 0) / fingerDists.length;

        // Tüm parmaklar kapalı mı
        const allFingersClosed = !indexExtended && !middleExtended && !ringExtended && !pinkyExtended;

        return {
            thumbIndexDist,
            indexExtended,
            middleExtended,
            ringExtended,
            pinkyExtended,
            allFingersClosed,
            avgFingerDist,
            palmCenter
        };
    }

    /**
     * El hareketinin yönünü ve hızını hesaplar
     */
    calculateMotion(landmarks) {
        // Avuç merkezi
        const palmCenter = {
            x: (landmarks[0].x + landmarks[5].x + landmarks[17].x) / 3,
            y: (landmarks[0].y + landmarks[5].y + landmarks[17].y) / 3
        };

        // Pinch mesafesi
        const pinchDist = Math.sqrt(
            Math.pow(landmarks[4].x - landmarks[8].x, 2) +
            Math.pow(landmarks[4].y - landmarks[8].y, 2)
        );

        if (this.prevPalmCenter) {
            this.palmDeltaX = (palmCenter.x - this.prevPalmCenter.x) * 100;
            this.palmDeltaY = (palmCenter.y - this.prevPalmCenter.y) * 100;
        } else {
            this.palmDeltaX = 0;
            this.palmDeltaY = 0;
        }

        if (this.prevPinchDist !== null) {
            this.pinchDelta = (pinchDist - this.prevPinchDist) * 100;
        } else {
            this.pinchDelta = 0;
        }

        this.prevPalmCenter = { ...palmCenter };
        this.prevPinchDist = pinchDist;
    }

    /**
     * Landmark'ları normalize eder (Python ile uyumlu)
     */
    normalizeLandmarks(landmarks) {
        const coords = [];
        for (const lm of landmarks) {
            coords.push(lm.x, lm.y, lm.z);
        }
        const baseX = coords[0], baseY = coords[1], baseZ = coords[2];
        const normalized = [];
        for (let i = 0; i < coords.length; i += 3) {
            normalized.push(coords[i] - baseX);
            normalized.push(coords[i + 1] - baseY);
            normalized.push(coords[i + 2] - baseZ);
        }
        const maxVal = Math.max(...normalized.map(Math.abs));
        return maxVal > 0 ? normalized.map(v => v / maxVal) : normalized;
    }

    /**
     * Parmak uçları arası mesafeleri hesaplar
     */
    calcFingerDistances(landmarks) {
        const tips = [4, 8, 12, 16, 20];
        const distances = [];
        for (let i = 0; i < tips.length; i++) {
            for (let j = i + 1; j < tips.length; j++) {
                const a = landmarks[tips[i]];
                const b = landmarks[tips[j]];
                distances.push(Math.sqrt(
                    Math.pow(a.x - b.x, 2) +
                    Math.pow(a.y - b.y, 2) +
                    Math.pow(a.z - b.z, 2)
                ));
            }
        }
        return distances;
    }

    /**
     * Dizideki en sık eleman
     */
    getMostFrequent(arr) {
        const counts = {};
        let maxItem = arr[arr.length - 1];
        let maxCount = 0;
        for (const item of arr) {
            counts[item] = (counts[item] || 0) + 1;
            if (counts[item] > maxCount) {
                maxCount = counts[item];
                maxItem = item;
            }
        }
        return maxItem;
    }
}

// Global instance
let gestureClassifier;

function initGestureClassifier() {
    gestureClassifier = new GestureClassifier();
    return gestureClassifier;
}
