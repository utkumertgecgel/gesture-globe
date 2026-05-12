/**
 * UIOverlay — Kullanıcı Arayüzü Yöneticisi & Uygulama Başlatıcı
 * 
 * Tüm modülleri birleştirir:
 * - GlobeController (3D küre)
 * - HandTracker (el takibi) 
 * - GestureClassifier (gesture tanıma)
 * 
 * Gesture → Globe kontrolünü bağlar.
 */

class UIOverlay {
    constructor() {
        this.globe = null;
        this.tracker = null;
        this.classifier = null;
        this.isInitialized = false;
    }

    async init() {
        const statusEl = document.getElementById('loading-status');
        const progressBar = document.getElementById('progress-bar');

        try {
            // Aşama 1: Gesture Classifier
            this.updateLoading(statusEl, progressBar, 'Gesture sınıflandırıcı hazırlanıyor...', 20);
            this.classifier = initGestureClassifier();
            await this.sleep(200);

            // Aşama 2: Confidence bar'ları oluştur
            this.updateLoading(statusEl, progressBar, 'Arayüz hazırlanıyor...', 40);
            this.createConfidenceBars();
            await this.sleep(200);

            // Aşama 3: Hand Tracker (kamera erişimi)
            this.updateLoading(statusEl, progressBar, 'Kamera erişimi isteniyor...', 60);
            this.tracker = new HandTracker();
            
            // Tracker callback'ini bağla
            this.tracker.onResults = (landmarks, handedness) => {
                this.onHandResults(landmarks, handedness);
            };
            
            await this.tracker.init('camera-feed', 'hand-overlay');
            await this.sleep(300);

            // Aşama 4: TF.js model yükleme (varsa)
            this.updateLoading(statusEl, progressBar, 'Model kontrol ediliyor...', 80);
            try {
                await this.classifier.loadModel('models/model.json');
            } catch (e) {
                console.log('Model bulunamadı, rule-based mod aktif.');
            }
            await this.sleep(200);

            // Tamamlandı
            this.updateLoading(statusEl, progressBar, 'Hazır!', 95);
            await this.sleep(400);

            // ÖNCELİKLE App'i göster (böylece container boyutu hesaplanabilir)
            const app = document.getElementById('app');
            app.classList.remove('hidden');

            // Loading ekranını kaldır
            const loadingScreen = document.getElementById('loading-screen');
            loadingScreen.classList.add('fade-out');

            // Globe'u ŞIMDI oluştur (container artık görünür ve boyutu doğru)
            await this.sleep(100);
            this.globe = initGlobe();

            // Loading ekranını DOM'dan kaldır
            setTimeout(() => loadingScreen.remove(), 800);

            // Guide toggle
            this.setupGuideToggle();

            this.isInitialized = true;
            console.log('GestureGlobe başarıyla başlatıldı!');

        } catch (error) {
            console.error('Başlatma hatası:', error);
            statusEl.textContent = 'Hata: ' + error.message;
            statusEl.style.color = '#ef4444';
        }
    }

    /**
     * El tespiti sonuçları geldiğinde çağrılır (her frame)
     */
    onHandResults(landmarks, handedness) {
        // Gesture'ı sınıflandır
        const result = this.classifier.classify(landmarks);

        // UI'ı güncelle
        this.updateGestureUI(result);

        // Confidence bar'ları güncelle
        this.updateConfidenceBars(result.allConfidences);

        // Globe'u kontrol et
        this.controlGlobe(result);
    }

    /**
     * Gesture sonucuna göre küreyi kontrol eder
     */
    controlGlobe(result) {
        if (!this.globe) return;

        switch (result.gesture) {
            case 'open_palm':
                // Avuç hareketiyle küreyi döndür
                if (Math.abs(result.palmDeltaX) > 0.15 || Math.abs(result.palmDeltaY) > 0.15) {
                    // Kamera ayna olduğu için X'i ters çeviriyoruz
                    this.globe.rotate(-result.palmDeltaX, result.palmDeltaY);
                }
                this.globe.unlock();
                break;

            case 'pinch':
                // Pinch mesafesi değişimine göre zoom
                if (Math.abs(result.pinchDelta) > 0.05) {
                    // Pinch kapanıyor → zoom in (negatif delta = yakınlaş)
                    // Pinch açılıyor → zoom out (pozitif delta = uzaklaş)
                    this.globe.zoom(-result.pinchDelta * 2);
                }
                break;

            case 'fist':
                // Yumruk → küreyi kilitle
                this.globe.lock();
                break;

            case 'point':
                // İşaret parmağı → şimdilik auto-rotate durdur
                this.globe.autoRotate = false;
                this.globe.unlock();
                break;

            case 'none':
                // El yok → auto-rotate devam
                if (!this.globe.isLocked) {
                    this.globe.autoRotate = true;
                }
                break;
        }
    }

    /**
     * Gesture bilgi panelini günceller
     */
    updateGestureUI(result) {
        const iconEl = document.getElementById('gesture-icon');
        const nameEl = document.getElementById('gesture-name');
        const confEl = document.getElementById('gesture-confidence');

        if (iconEl) iconEl.textContent = this.classifier.gestureIcons[result.gesture] || '—';
        if (nameEl) nameEl.textContent = this.classifier.gestureTR[result.gesture] || result.gesture;
        if (confEl) confEl.textContent = `Güven: ${(result.confidence * 100).toFixed(0)}%`;
    }

    /**
     * Confidence bar HTML'ini oluşturur
     */
    createConfidenceBars() {
        const container = document.getElementById('confidence-bars');
        if (!container) return;

        container.innerHTML = '';
        const gestures = ['open_palm', 'pinch', 'fist', 'point', 'none'];
        const labels = ['Açık Avuç', 'Kıstırma', 'Yumruk', 'İşaret', 'Yok'];

        gestures.forEach((gesture, i) => {
            const item = document.createElement('div');
            item.className = 'confidence-bar-item';
            item.innerHTML = `
                <span class="confidence-bar-label">${labels[i]}</span>
                <div class="confidence-bar-track">
                    <div class="confidence-bar-fill" id="conf-bar-${gesture}"></div>
                </div>
                <span class="confidence-bar-value" id="conf-val-${gesture}">0%</span>
            `;
            container.appendChild(item);
        });
    }

    /**
     * Confidence bar'ları günceller
     */
    updateConfidenceBars(confidences) {
        for (const [gesture, conf] of Object.entries(confidences)) {
            const barEl = document.getElementById(`conf-bar-${gesture}`);
            const valEl = document.getElementById(`conf-val-${gesture}`);
            if (barEl) barEl.style.width = `${(conf * 100).toFixed(0)}%`;
            if (valEl) valEl.textContent = `${(conf * 100).toFixed(0)}%`;
        }
    }

    /**
     * Guide panel toggle
     */
    setupGuideToggle() {
        const toggleBtn = document.getElementById('guide-toggle');
        const content = document.getElementById('guide-content');
        if (toggleBtn && content) {
            toggleBtn.addEventListener('click', () => {
                const isHidden = content.style.display === 'none';
                content.style.display = isHidden ? 'block' : 'none';
                toggleBtn.textContent = isHidden ? '▼' : '▲';
            });
        }
    }

    // Yardımcı fonksiyonlar
    updateLoading(statusEl, progressBar, message, percent) {
        if (statusEl) statusEl.textContent = message;
        if (progressBar) progressBar.style.width = `${percent}%`;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// ============================================================
// UYGULAMA BAŞLATICI
// ============================================================

document.addEventListener('DOMContentLoaded', async () => {
    const app = new UIOverlay();
    await app.init();
});
