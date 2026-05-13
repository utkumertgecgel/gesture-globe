/**
 * GestureClassifier v2 — Geliştirilmiş El Hareketi Sınıflandırma
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
        this.modelWeights = null;
        this.useModel = false;
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

        // Smoothing: Son N frame — titreşimi önler
        this.historySize = 7;
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
        this.stableThreshold = 3; // 3 frame aynı olmalı
    }

    async loadModel(modelPath) {
        try {
            const response = await fetch(modelPath);
            if (!response.ok) throw new Error('Model dosyasi bulunamadi');
            const modelData = await response.json();
            this.model = modelData;
            this.modelWeights = modelData.weights;

            const labelResponse = await fetch('models/label_map.json');
            if (labelResponse.ok) {
                const labelMap = await labelResponse.json();
                this.gestureNames = Object.values(labelMap);
            }

            // NOT: Gerçek veri ile eğitilene kadar model devre dışı
            this.useModel = false;
            console.log('Model yuklendi ama rule-based mod aktif (gercek veri gerekli)');
        } catch (e) {
            console.warn('Model yuklenemedi, rule-based mod aktif:', e.message);
            this.useModel = false;
        }
    }

    /**
     * Ana sınıflandırma
     */
    classify(landmarks) {
        if (!landmarks) {
            this.prevPalmCenter = null;
            this.prevPinchDist = null;
            this.palmDeltaX = 0;
            this.palmDeltaY = 0;
            this.pinchDelta = 0;
            return {
                gesture: 'none', confidence: 1.0,
                allConfidences: { open_palm: 0, pinch: 0, fist: 0, point: 0, none: 1 },
                palmDeltaX: 0, palmDeltaY: 0, pinchDelta: 0
            };
        }

        // Sınıflandır
        let result;
        if (this.useModel && this.modelWeights) {
            result = this.classifyWithModel(landmarks);
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
    // RULE-BASED SINIFLANDIRMA (Geliştirilmiş v2)
    // ================================================================

    classifyRuleBased(landmarks) {
        const f = this.getFingerStates(landmarks);
        const confidences = { open_palm: 0, pinch: 0, fist: 0, point: 0, none: 0.1 };
        let gesture = 'none';
        let confidence = 0.1;

        const closedCount = [!f.indexOpen, !f.middleOpen, !f.ringOpen, !f.pinkyOpen].filter(Boolean).length;
        const openCount = 4 - closedCount;

        // ============================================================
        // ÖNCELİK SIRASI: FIST > POINT > PINCH > OPEN_PALM
        // Her adımda kesin karar verilir, çakışma olmaz
        // ============================================================

        // 1) FIST — en az 3 parmak kapalı
        if (closedCount >= 3) {
            // İşaret parmağı da kapalıysa → kesin yumruk
            if (closedCount >= 4) {
                gesture = 'fist';
                confidence = 0.95;
                confidences.fist = 0.95;
            }
            // 3 kapalı ama işaret açık → POINT olabilir, kontrol et
            else if (f.indexOpen && !f.middleOpen && !f.ringOpen && !f.pinkyOpen) {
                gesture = 'point';
                confidence = 0.90;
                confidences.point = 0.90;
            }
            // 3 kapalı, başka bir parmak açık → muhtemelen yumruk
            else {
                gesture = 'fist';
                confidence = 0.80;
                confidences.fist = 0.80;
            }
        }
        // 2) POINT — işaret açık, diğerleri kapalı (closedCount = 2 veya 3)
        else if (f.indexOpen && !f.ringOpen && !f.pinkyOpen && closedCount >= 2) {
            gesture = 'point';
            confidence = 0.85;
            confidences.point = 0.85;
        }
        // 3) PINCH — başparmak + işaret yakın AMA diğer parmaklar tam kapalı DEĞİL
        //    (fist'te de pinch mesafesi düşüktür, o yüzden closedCount<3 koşulu önemli)
        else if (f.pinchDistance < 0.10 && closedCount < 3) {
            gesture = 'pinch';
            confidence = f.pinchDistance < 0.06 ? 0.95 : 0.80;
            confidences.pinch = confidence;
        }
        // 4) OPEN PALM — en az 3 parmak açık
        else if (openCount >= 3) {
            gesture = 'open_palm';
            confidence = openCount >= 4 ? 0.95 : 0.80;
            confidences.open_palm = confidence;
        }

        // Normalize confidences for display
        confidences[gesture] = confidence;
        const total = Object.values(confidences).reduce((a, b) => a + b, 0) || 1;
        for (const key of Object.keys(confidences)) {
            confidences[key] = Math.round((confidences[key] / total) * 100) / 100;
        }

        return { gesture, confidence, allConfidences: confidences };
    }

    /**
     * Parmak açık/kapalı durumlarını belirler
     * MediaPipe'ın Y ekseni aşağı doğru artar!
     * Parmak ucu (tip) eklem noktasından (PIP) daha YUKARIDA ise → parmak açık
     */
    getFingerStates(lm) {
        const wrist = lm[0];

        // Mesafe hesaplama
        const dist = (a, b) => Math.sqrt(
            Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2)
        );

        // Başparmak: tip(4) vs ip(3) — MCP(2) referans
        const thumbTipDist = dist(lm[4], lm[2]);
        const thumbIpDist = dist(lm[3], lm[2]);
        const thumbOpen = thumbTipDist > thumbIpDist * 1.1;

        // Diğer parmaklar: tip bilege uzaksa açık
        const indexOpen  = dist(lm[8], wrist) > dist(lm[6], wrist);
        const middleOpen = dist(lm[12], wrist) > dist(lm[10], wrist);
        const ringOpen   = dist(lm[16], wrist) > dist(lm[14], wrist);
        const pinkyOpen  = dist(lm[20], wrist) > dist(lm[18], wrist);

        // Pinch mesafesi
        const pinchDistance = dist(lm[4], lm[8]);

        // Avuç merkezi ve parmak uçlarının ortalama mesafesi (fist için)
        const palmCenter = {
            x: (lm[0].x + lm[5].x + lm[17].x) / 3,
            y: (lm[0].y + lm[5].y + lm[17].y) / 3
        };
        const tipIndices = [8, 12, 16, 20];
        const tipDists = tipIndices.map(i => dist(lm[i], palmCenter));
        const avgTipToPalm = tipDists.reduce((a, b) => a + b, 0) / tipDists.length;

        // Debug
        this.debugInfo = {
            thumbOpen, indexOpen, middleOpen, ringOpen, pinkyOpen,
            pinchDist: pinchDistance.toFixed(3),
            tipPalm: avgTipToPalm.toFixed(3),
            fingers: `${thumbOpen?'T':'_'}${indexOpen?'I':'_'}${middleOpen?'M':'_'}${ringOpen?'R':'_'}${pinkyOpen?'P':'_'}`
        };

        return {
            thumbOpen, indexOpen, middleOpen, ringOpen, pinkyOpen,
            pinchDistance, avgTipToPalm,
            palmCenter
        };
    }

    // ================================================================
    // MODEL-BASED SINIFLANDIRMA (Saf JS)
    // ================================================================

    classifyWithModel(landmarks) {
        const normalized = this.normalizeLandmarks(landmarks);
        const distances = this.calcFingerDistances(landmarks);
        let x = [...normalized, ...distances];

        const w = this.modelWeights;

        // Linear(73→128) + BatchNorm + ReLU
        x = this.linearLayer(x, w['network.0.weight'], w['network.0.bias']);
        x = this.batchNormLayer(x, w['network.1.weight'], w['network.1.bias'],
            w['network.1.running_mean'], w['network.1.running_var']);
        x = x.map(v => Math.max(0, v));

        // Linear(128→64) + BatchNorm + ReLU
        x = this.linearLayer(x, w['network.4.weight'], w['network.4.bias']);
        x = this.batchNormLayer(x, w['network.5.weight'], w['network.5.bias'],
            w['network.5.running_mean'], w['network.5.running_var']);
        x = x.map(v => Math.max(0, v));

        // Linear(64→32) + ReLU
        x = this.linearLayer(x, w['network.8.weight'], w['network.8.bias']);
        x = x.map(v => Math.max(0, v));

        // Linear(32→5)
        x = this.linearLayer(x, w['network.10.weight'], w['network.10.bias']);

        // Softmax
        const maxVal = Math.max(...x);
        const exps = x.map(v => Math.exp(v - maxVal));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(v => v / sumExps);

        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        const confidences = {};
        this.gestureNames.forEach((name, i) => {
            confidences[name] = i < probs.length ? probs[i] : 0;
        });

        return { gesture: this.gestureNames[maxIdx], confidence: probs[maxIdx], allConfidences: confidences };
    }

    linearLayer(input, weight, bias) {
        const outSize = weight.length;
        const result = new Array(outSize);
        for (let i = 0; i < outSize; i++) {
            let sum = bias[i];
            for (let j = 0; j < input.length; j++) {
                sum += weight[i][j] * input[j];
            }
            result[i] = sum;
        }
        return result;
    }

    batchNormLayer(input, gamma, beta, runningMean, runningVar) {
        const eps = 1e-5;
        return input.map((v, i) => {
            const normalized = (v - runningMean[i]) / Math.sqrt(runningVar[i] + eps);
            return gamma[i] * normalized + beta[i];
        });
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
    // ================================================================

    normalizeLandmarks(landmarks) {
        const coords = [];
        for (const lm of landmarks) coords.push(lm.x, lm.y, lm.z);
        const baseX = coords[0], baseY = coords[1], baseZ = coords[2];
        const normalized = [];
        for (let i = 0; i < coords.length; i += 3) {
            normalized.push(coords[i] - baseX, coords[i+1] - baseY, coords[i+2] - baseZ);
        }
        const maxVal = Math.max(...normalized.map(Math.abs));
        return maxVal > 0 ? normalized.map(v => v / maxVal) : normalized;
    }

    calcFingerDistances(landmarks) {
        const tips = [4, 8, 12, 16, 20];
        const distances = [];
        for (let i = 0; i < tips.length; i++) {
            for (let j = i + 1; j < tips.length; j++) {
                const a = landmarks[tips[i]], b = landmarks[tips[j]];
                distances.push(Math.sqrt(
                    Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2) + Math.pow(a.z - b.z, 2)
                ));
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
