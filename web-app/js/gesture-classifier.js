/**
 * GestureClassifier v3 — ONNX CNN Model + Rule-Based Fallback
 * 
 * Gesture'lar:
 *   open_palm  → Küreyi döndür
 *   pinch      → Zoom in/out
 *   fist       → Durdur
 *   none       → Hiçbir şey yapma
 */

class GestureClassifier {
    constructor() {
        this.session = null;
        this.useModel = false;
        // Model 4 sınıf döndürür (label_map'ten gelir)
        this.modelClassNames = ['fist', 'open_palm', 'pinch'];
        // UI için tüm gesture isimleri (none dahil)
        this.gestureNames = ['open_palm', 'pinch', 'fist', 'none'];
        this.gestureIcons = {
            'open_palm': '✋',
            'pinch': '🤏',
            'fist': '✊',
            'none': '—'
        };
        this.gestureTR = {
            'open_palm': 'Açık Avuç',
            'pinch': 'Kıstırma',
            'fist': 'Yumruk',
            'none': 'Bekleniyor'
        };

        // Smoothing: Son N frame — titreşimi önler
        this.historySize = 3;  // Daha hizli tepki
        this.history = [];

        // Hareket takibi
        this.prevPalmCenter = null;
        this.prevPinchDist = null;
        this.palmDeltaX = 0;
        this.palmDeltaY = 0;
        this.pinchDelta = 0;

        // Stabilite: Aynı gesture N frame boyunca tutarlıysa değiştir
        this.currentStableGesture = 'none';
        this.stableCount = 0;
        this.stableThreshold = 2;  // Daha hizli gecis

        // Model confidence esigi — altindaysa "none" kabul et
        this.confidenceThreshold = 0.55;  // Biraz daha toleransli

        // Zoom accumulator — kucuk hareketleri biriktirir
        this.zoomAccum = 0;
    }

    async loadModel(modelPath) {
        try {
            // ONNX modelini yükle
            this.session = await ort.InferenceSession.create(modelPath);
            
            const labelResponse = await fetch('models/label_map.json');
            if (labelResponse.ok) {
                const labelMap = await labelResponse.json();
                // Model sınıf isimleri (index sıralı)
                this.modelClassNames = [];
                const keys = Object.keys(labelMap).sort((a,b) => parseInt(a) - parseInt(b));
                for (const k of keys) {
                    this.modelClassNames.push(labelMap[k]);
                }
            }

            this.useModel = true;
            console.log('✅ ONNX CNN Modeli yüklendi! Sınıflar:', this.modelClassNames);
            console.log('   Input names:', this.session.inputNames);
            console.log('   Output names:', this.session.outputNames);
        } catch (e) {
            console.warn('⚠️ Model yüklenemedi, rule-based mod aktif:', e.message);
            this.useModel = false;
        }
    }

    /**
     * Ana sınıflandırma (async — ONNX inference)
     */
    async classify(landmarks) {
        if (!landmarks) {
            this.prevPalmCenter = null;
            this.prevPinchDist = null;
            this.palmDeltaX = 0;
            this.palmDeltaY = 0;
            this.pinchDelta = 0;
            return {
                gesture: 'none', confidence: 1.0,
                allConfidences: { open_palm: 0, pinch: 0, fist: 0, none: 1 },
                palmDeltaX: 0, palmDeltaY: 0, pinchDelta: 0
            };
        }

        // Parmak durumlarını hesapla (debug overlay için)
        this.getFingerStates(landmarks);

        // Sınıflandır
        let result;
        if (this.useModel && this.session) {
            result = await this.classifyWithONNX(landmarks);
        } else {
            result = this.classifyRuleBased(landmarks);
        }

        // Hareket hesapla
        this.calculateMotion(landmarks);
        result.palmDeltaX = this.palmDeltaX;
        result.palmDeltaY = this.palmDeltaY;
        result.pinchDelta = this.pinchDelta;

        // Çift katmanlı smoothing
        // 1. History-based majority vote
        this.history.push(result.gesture);
        if (this.history.length > this.historySize) this.history.shift();
        const voted = this.getMostFrequent(this.history);

        // 2. Stability filter — titreşimi önler
        if (voted === this.currentStableGesture) {
            this.stableCount = Math.min(this.stableCount + 1, 10);
        } else {
            this.stableCount++;
            if (this.stableCount >= this.stableThreshold) {
                this.currentStableGesture = voted;
                this.stableCount = 0;
            }
        }

        result.gesture = this.currentStableGesture;
        return result;
    }

    // ================================================================
    // ONNX MODEL SINIFLANDIRMA
    // ================================================================

    async classifyWithONNX(landmarks) {
        // Eğitim verisiyle BİREBİR aynı preprocessing
        const normalized = this.normalizeLandmarks(landmarks);
        const distances = this.calcFingerDistancesFromNormalized(normalized);
        const inputArray = [...normalized, ...distances];

        try {
            const inputTensor = new ort.Tensor('float32', Float32Array.from(inputArray), [1, 73]);
            const feeds = {};
            feeds[this.session.inputNames[0]] = inputTensor;
            
            const outputData = await this.session.run(feeds);
            const output = outputData[this.session.outputNames[0]].data;

            // Softmax
            const maxVal = Math.max(...output);
            const exps = Array.from(output).map(v => Math.exp(v - maxVal));
            const sumExps = exps.reduce((a, b) => a + b, 0);
            const probs = exps.map(v => v / sumExps);

            // En yüksek olasılıklı sınıfı bul
            let maxIdx = 0;
            for (let i = 1; i < probs.length; i++) {
                if (probs[i] > probs[maxIdx]) maxIdx = i;
            }

            const bestGesture = this.modelClassNames[maxIdx];
            const bestConf = probs[maxIdx];

            // Confidence haritası oluştur
            const confidences = { open_palm: 0, pinch: 0, fist: 0, none: 0 };
            this.modelClassNames.forEach((name, i) => {
                confidences[name] = Math.round(probs[i] * 1000) / 1000;
            });

            // Eğer en yüksek confidence eşiğin altındaysa → none
            let gesture = bestGesture;
            if (bestConf < this.confidenceThreshold) {
                gesture = 'none';
                confidences.none = 1.0 - bestConf;
            }

            return { gesture, confidence: bestConf, allConfidences: confidences };
        } catch(e) {
            console.error("ONNX inference hatası:", e);
            return this.classifyRuleBased(landmarks);
        }
    }

    // ================================================================
    // RULE-BASED SINIFLANDIRMA (Yedek)
    // ================================================================

    classifyRuleBased(landmarks) {
        const f = this.getFingerStates(landmarks);
        const confidences = { open_palm: 0, pinch: 0, fist: 0, none: 0.1 };
        let gesture = 'none';
        let confidence = 0.1;

        const closedCount = [!f.indexOpen, !f.middleOpen, !f.ringOpen, !f.pinkyOpen].filter(Boolean).length;
        const openCount = 4 - closedCount;

        // Pinch mesafesini el buyuklugune gore normalize et
        // El buyuklugu = bilek (0) ile orta parmak dibi (9) arasi mesafe
        const handSize = Math.sqrt(
            Math.pow(lm[0].x - lm[9].x, 2) + Math.pow(lm[0].y - lm[9].y, 2)
        ) || 0.1;
        const normalizedPinch = f.pinchDistance / handSize;

        // ONCELIK SIRASI: FIST > PINCH > OPEN_PALM
        if (closedCount >= 3) {
            gesture = 'fist';
            confidence = closedCount >= 4 ? 0.95 : 0.80;
            confidences.fist = confidence;
        }
        // Normalize edilmis pinch esigi — el kameraya ne kadar uzak olursa olsun tutarli
        else if (normalizedPinch < 0.35 && closedCount < 3) {
            gesture = 'pinch';
            confidence = normalizedPinch < 0.20 ? 0.95 : 0.80;
            confidences.pinch = confidence;
        }
        else if (openCount >= 3) {
            gesture = 'open_palm';
            confidence = openCount >= 4 ? 0.95 : 0.80;
            confidences.open_palm = confidence;
        }

        confidences[gesture] = confidence;
        const total = Object.values(confidences).reduce((a, b) => a + b, 0) || 1;
        for (const key of Object.keys(confidences)) {
            confidences[key] = Math.round((confidences[key] / total) * 100) / 100;
        }

        return { gesture, confidence, allConfidences: confidences };
    }

    /**
     * Parmak açık/kapalı durumlarını belirler
     */
    getFingerStates(lm) {
        const wrist = lm[0];
        const dist = (a, b) => Math.sqrt(
            Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2)
        );

        const thumbTipDist = dist(lm[4], lm[2]);
        const thumbIpDist = dist(lm[3], lm[2]);
        const thumbOpen = thumbTipDist > thumbIpDist * 1.1;

        const indexOpen  = dist(lm[8], wrist) > dist(lm[6], wrist);
        const middleOpen = dist(lm[12], wrist) > dist(lm[10], wrist);
        const ringOpen   = dist(lm[16], wrist) > dist(lm[14], wrist);
        const pinkyOpen  = dist(lm[20], wrist) > dist(lm[18], wrist);

        const pinchDistance = dist(lm[4], lm[8]);

        const palmCenter = {
            x: (lm[0].x + lm[5].x + lm[17].x) / 3,
            y: (lm[0].y + lm[5].y + lm[17].y) / 3
        };
        const tipIndices = [8, 12, 16, 20];
        const tipDists = tipIndices.map(i => dist(lm[i], palmCenter));
        const avgTipToPalm = tipDists.reduce((a, b) => a + b, 0) / tipDists.length;

        const modeStr = this.useModel ? '🧠Model' : '📏Kural';

        this.debugInfo = {
            thumbOpen, indexOpen, middleOpen, ringOpen, pinkyOpen,
            pinchDist: pinchDistance.toFixed(3),
            tipPalm: avgTipToPalm.toFixed(3),
            fingers: `${modeStr} ${thumbOpen?'T':'_'}${indexOpen?'I':'_'}${middleOpen?'M':'_'}${ringOpen?'R':'_'}${pinkyOpen?'P':'_'}`
        };

        return {
            thumbOpen, indexOpen, middleOpen, ringOpen, pinkyOpen,
            pinchDistance, avgTipToPalm, palmCenter
        };
    }

    // ================================================================
    // HAREKET TAKİBİ
    // ================================================================

    calculateMotion(landmarks) {
        const palmCenter = {
            x: (landmarks[0].x + landmarks[5].x + landmarks[17].x) / 3,
            y: (landmarks[0].y + landmarks[5].y + landmarks[17].y) / 3
        };
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

    // ================================================================
    // YARDIMCI FONKSİYONLAR
    // Eğitim verisiyle BİREBİR aynı preprocessing!
    // ================================================================

    /**
     * Landmark'ları normalize et:
     * 1. Bilek (index 0) orijin yapılır
     * 2. Max mutlak değere bölünerek [-1, 1] aralığına ölçeklenir
     * Çıktı: 63 elemanlı düz dizi [x0,y0,z0, x1,y1,z1, ...]
     */
    normalizeLandmarks(landmarks) {
        const coords = [];
        for (const lm of landmarks) {
            coords.push(lm.x, lm.y, lm.z || 0);
        }
        const baseX = coords[0], baseY = coords[1], baseZ = coords[2];
        const normalized = [];
        for (let i = 0; i < coords.length; i += 3) {
            normalized.push(coords[i] - baseX, coords[i+1] - baseY, coords[i+2] - baseZ);
        }
        const maxVal = Math.max(...normalized.map(Math.abs)) || 1;
        return normalized.map(v => v / maxVal);
    }

    /**
     * Parmak uçları arası 10 mesafe — NORMALİZE EDİLMİŞ koordinatlardan hesaplanır
     * (Eğitim verisindeki collect.html ile birebir aynı!)
     * Tips: 4(thumb), 8(index), 12(middle), 16(ring), 20(pinky)
     * Her tip'in normalize dizideki başlangıç indexi: tip_landmark * 3
     */
    calcFingerDistancesFromNormalized(normalizedCoords) {
        const tips = [4, 8, 12, 16, 20];
        const distances = [];
        for (let i = 0; i < tips.length; i++) {
            for (let j = i + 1; j < tips.length; j++) {
                const ai = tips[i] * 3, bi = tips[j] * 3;
                const dx = normalizedCoords[ai] - normalizedCoords[bi];
                const dy = normalizedCoords[ai+1] - normalizedCoords[bi+1];
                const dz = normalizedCoords[ai+2] - normalizedCoords[bi+2];
                distances.push(Math.sqrt(dx*dx + dy*dy + dz*dz));
            }
        }
        return distances;
    }

    getMostFrequent(arr) {
        const counts = {};
        let maxItem = arr[arr.length - 1], maxCount = 0;
        for (const item of arr) {
            counts[item] = (counts[item] || 0) + 1;
            if (counts[item] > maxCount) { maxCount = counts[item]; maxItem = item; }
        }
        return maxItem;
    }
}

// Global
let gestureClassifier;
function initGestureClassifier() {
    gestureClassifier = new GestureClassifier();
    return gestureClassifier;
}
