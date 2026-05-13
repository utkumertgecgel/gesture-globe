/**
 * HandTracker — MediaPipe Hands ile El Takibi
 * 
 * Kameradan el landmark'larını tespit eder ve
 * gesture classifier'a iletir.
 */

class HandTracker {
    constructor() {
        this.hands = null;
        this.camera = null;
        this.videoElement = null;
        this.canvasElement = null;
        this.canvasCtx = null;
        this.isRunning = false;
        this.onResults = null; // callback
        this.lastLandmarks = null;
    }

    async init(videoElementId, canvasElementId) {
        this.videoElement = document.getElementById(videoElementId);
        this.canvasElement = document.getElementById(canvasElementId);
        this.canvasCtx = this.canvasElement.getContext('2d');

        // MediaPipe Hands başlat
        this.hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });

        this.hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.4,
        });

        this.hands.onResults((results) => this.processResults(results));

        // Kamerayı başlat — laptop webcam'ini tercih et
        try {
            // Önce mevcut kameraları listele
            const devices = await navigator.mediaDevices.enumerateDevices();
            const cameras = devices.filter(d => d.kind === 'videoinput');
            console.log('Bulunan kameralar:', cameras.map(c => c.label || c.deviceId));

            // Laptop kamerasını bul (DroidCam, phone, virtual gibi isimleri atla)
            const skipKeywords = ['droid', 'phone', 'virtual', 'obs', 'snap'];
            let preferredDeviceId = null;
            
            for (const cam of cameras) {
                const label = (cam.label || '').toLowerCase();
                const isVirtual = skipKeywords.some(kw => label.includes(kw));
                if (!isVirtual && cam.deviceId) {
                    preferredDeviceId = cam.deviceId;
                    console.log('Seçilen kamera:', cam.label || cam.deviceId);
                    break;
                }
            }

            // Kamera stream'ini al
            const constraints = {
                video: preferredDeviceId 
                    ? { deviceId: { exact: preferredDeviceId }, width: 640, height: 480 }
                    : { facingMode: 'user', width: 640, height: 480 }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = stream;
            await this.videoElement.play();

            // Frame döngüsü başlat
            this.isRunning = true;
            const processFrame = async () => {
                if (!this.isRunning) return;
                try {
                    await this.hands.send({ image: this.videoElement });
                } catch (e) {
                    // Frame atlama hatası, devam et
                }
                requestAnimationFrame(processFrame);
            };
            requestAnimationFrame(processFrame);

        } catch (err) {
            console.error('Kamera erişim hatası:', err);
            // Kullanıcıya bilgi göster
            const wrapper = document.querySelector('.camera-wrapper');
            if (wrapper) {
                wrapper.innerHTML = `
                    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:12px;text-align:center;">
                        <span style="font-size:2rem;margin-bottom:8px;">📷</span>
                        <span style="color:#ef4444;font-size:0.75rem;">Kamera erişilemedi</span>
                        <span style="color:#9ca3af;font-size:0.65rem;margin-top:4px;">Tarayıcı izni kontrol et</span>
                    </div>`;
            }
        }
    }

    processResults(results) {
        // Canvas boyutunu video ile eşitle
        this.canvasElement.width = this.videoElement.videoWidth || 320;
        this.canvasElement.height = this.videoElement.videoHeight || 240;
        
        const ctx = this.canvasCtx;
        ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];
            this.lastLandmarks = landmarks;

            // El landmark'larını çiz
            this.drawHand(ctx, landmarks);

            // Callback'e bildir
            if (this.onResults) {
                this.onResults(landmarks, results.multiHandedness);
            }
        } else {
            this.lastLandmarks = null;
            if (this.onResults) {
                this.onResults(null, null);
            }
        }
    }

    drawHand(ctx, landmarks) {
        const w = this.canvasElement.width;
        const h = this.canvasElement.height;

        // Bağlantı çizgileri
        const connections = [
            [0,1],[1,2],[2,3],[3,4],        // Başparmak
            [0,5],[5,6],[6,7],[7,8],        // İşaret
            [0,9],[9,10],[10,11],[11,12],   // Orta
            [0,13],[13,14],[14,15],[15,16], // Yüzük
            [0,17],[17,18],[18,19],[19,20], // Serçe
            [5,9],[9,13],[13,17],           // Avuç içi
        ];

        // Çizgiler
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.6)';
        ctx.lineWidth = 2;
        for (const [a, b] of connections) {
            ctx.beginPath();
            ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
            ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
            ctx.stroke();
        }

        // Noktalar
        for (let i = 0; i < landmarks.length; i++) {
            const x = landmarks[i].x * w;
            const y = landmarks[i].y * h;

            // Parmak uçları büyük ve renkli
            if ([4, 8, 12, 16, 20].includes(i)) {
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fillStyle = '#ef4444';
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
            // Bilek
            else if (i === 0) {
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fillStyle = '#6366f1';
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
            // Diğer noktalar
            else {
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(99, 102, 241, 0.7)';
                ctx.fill();
            }
        }

        // Başparmak-işaret parmağı arası mesafe çizgisi (pinch göstergesi)
        const thumb = landmarks[4];
        const index = landmarks[8];
        const dist = Math.sqrt(
            Math.pow(thumb.x - index.x, 2) + 
            Math.pow(thumb.y - index.y, 2)
        );
        
        const pinchColor = dist < 0.06 ? '#f59e0b' : 'rgba(99, 102, 241, 0.3)';
        ctx.strokeStyle = pinchColor;
        ctx.lineWidth = dist < 0.06 ? 3 : 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(thumb.x * w, thumb.y * h);
        ctx.lineTo(index.x * w, index.y * h);
        ctx.stroke();
        ctx.setLineDash([]);

        // ---- DEBUG OVERLAY: Canvas üzerine parmak durumu yaz ----
        if (typeof gestureClassifier !== 'undefined' && gestureClassifier.debugInfo) {
            const d = gestureClassifier.debugInfo;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
            ctx.fillRect(0, h - 28, w, 28);
            ctx.font = 'bold 13px monospace';
            ctx.fillStyle = '#00ff00';
            ctx.fillText(`${d.fingers} pnch:${d.pinchDist} palm:${d.tipPalm}`, 4, h - 9);
        }
    }

    // Landmark'ları normalize et (Python'daki ile aynı)
    normalizeLandmarks(landmarks) {
        if (!landmarks) return null;

        const coords = [];
        for (const lm of landmarks) {
            coords.push(lm.x, lm.y, lm.z);
        }

        // Bilek noktasını orijin yap
        const baseX = coords[0], baseY = coords[1], baseZ = coords[2];
        const normalized = [];
        for (let i = 0; i < coords.length; i += 3) {
            normalized.push(coords[i] - baseX);
            normalized.push(coords[i + 1] - baseY);
            normalized.push(coords[i + 2] - baseZ);
        }

        // Max değere göre ölçekle
        const maxVal = Math.max(...normalized.map(Math.abs));
        if (maxVal > 0) {
            return normalized.map(v => v / maxVal);
        }
        return normalized;
    }
}

// Global instance
let handTracker;

async function initHandTracker() {
    handTracker = new HandTracker();
    await handTracker.init('camera-feed', 'hand-overlay');
    return handTracker;
}
