/**
 * GlobeController — Three.js ile 3D Dünya Küresini Yönetir
 * 
 * Özellikler:
 * - NASA Blue Marble texture'ı ile gerçekçi dünya
 * - Atmosfer glow efekti (Fresnel shader)
 * - Yıldız arka planı (particle system)
 * - Smooth rotation/zoom animasyonları
 * - Gesture komutlarıyla kontrol API'si
 */

class GlobeController {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.globe = null;
        this.atmosphere = null;
        this.stars = null;
        this.clouds = null;

        // Kontrol durumu
        this.targetRotationX = 0.3;  // Başlangıç açısı
        this.targetRotationY = 0;
        this.currentRotationX = 0.3;
        this.currentRotationY = 0;
        this.targetZoom = 3.5;
        this.currentZoom = 3.5;
        this.autoRotate = true;
        this.autoRotateSpeed = 0.001;
        this.isLocked = false;

        // Smooth hareket parametreleri
        this.rotationSmoothing = 0.08;
        this.zoomSmoothing = 0.06;
        this.minZoom = 1.5;
        this.maxZoom = 6.0;

        // FPS
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
        this.fps = 0;

        this.init();
    }

    init() {
        this.createScene();
        this.createGlobe();
        this.createAtmosphere();
        this.createStars();
        this.addLighting();
        this.animate();

        // Pencere boyutu değiştiğinde
        window.addEventListener('resize', () => this.onResize());
    }

    createScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000008);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            45,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.z = this.currentZoom;

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false,
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);
    }

    createGlobe() {
        const geometry = new THREE.SphereGeometry(1, 64, 64);

        // Prosedürel dünya texture'ı oluştur (harici dosya gerekmez)
        const canvas = document.createElement('canvas');
        canvas.width = 2048;
        canvas.height = 1024;
        const ctx = canvas.getContext('2d');

        // Okyanus arka planı
        const oceanGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
        oceanGradient.addColorStop(0, '#1a3a5c');
        oceanGradient.addColorStop(0.3, '#1e4d7a');
        oceanGradient.addColorStop(0.5, '#1a3a5c');
        oceanGradient.addColorStop(0.7, '#1e4d7a');
        oceanGradient.addColorStop(1, '#1a3a5c');
        ctx.fillStyle = oceanGradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Kıtaları çiz (basitleştirilmiş)
        this.drawContinents(ctx, canvas.width, canvas.height);

        const texture = new THREE.CanvasTexture(canvas);
        texture.wrapS = THREE.RepeatWrapping;

        const material = new THREE.MeshPhongMaterial({
            map: texture,
            specular: new THREE.Color(0x333333),
            shininess: 15,
            bumpScale: 0.02,
        });

        this.globe = new THREE.Mesh(geometry, material);
        this.scene.add(this.globe);

        // Bulut katmanı
        this.createClouds();
    }

    drawContinents(ctx, w, h) {
        // Equirectangular projeksiyonda kıta koordinatları (yaklaşık)
        const landColor = '#2d5a27';
        const landHighColor = '#3a7a33';
        const desertColor = '#8a7d5a';
        const iceColor = '#d4dce8';

        ctx.fillStyle = iceColor;
        // Kuzey kutbu
        ctx.fillRect(0, 0, w, h * 0.06);
        // Güney kutbu (Antarktika)
        ctx.fillRect(0, h * 0.92, w, h * 0.08);

        // Kuzey Amerika
        ctx.fillStyle = landColor;
        ctx.beginPath();
        ctx.moveTo(w * 0.12, h * 0.15);
        ctx.quadraticCurveTo(w * 0.18, h * 0.12, w * 0.25, h * 0.15);
        ctx.quadraticCurveTo(w * 0.30, h * 0.20, w * 0.28, h * 0.30);
        ctx.quadraticCurveTo(w * 0.22, h * 0.35, w * 0.20, h * 0.42);
        ctx.quadraticCurveTo(w * 0.17, h * 0.38, w * 0.14, h * 0.35);
        ctx.quadraticCurveTo(w * 0.10, h * 0.25, w * 0.12, h * 0.15);
        ctx.fill();

        // Güney Amerika
        ctx.fillStyle = landHighColor;
        ctx.beginPath();
        ctx.moveTo(w * 0.22, h * 0.48);
        ctx.quadraticCurveTo(w * 0.27, h * 0.50, w * 0.28, h * 0.55);
        ctx.quadraticCurveTo(w * 0.29, h * 0.65, w * 0.26, h * 0.75);
        ctx.quadraticCurveTo(w * 0.24, h * 0.80, w * 0.22, h * 0.82);
        ctx.quadraticCurveTo(w * 0.20, h * 0.75, w * 0.21, h * 0.65);
        ctx.quadraticCurveTo(w * 0.20, h * 0.55, w * 0.22, h * 0.48);
        ctx.fill();

        // Avrupa
        ctx.fillStyle = landColor;
        ctx.beginPath();
        ctx.moveTo(w * 0.46, h * 0.15);
        ctx.quadraticCurveTo(w * 0.50, h * 0.14, w * 0.54, h * 0.16);
        ctx.quadraticCurveTo(w * 0.52, h * 0.22, w * 0.50, h * 0.28);
        ctx.quadraticCurveTo(w * 0.47, h * 0.30, w * 0.45, h * 0.26);
        ctx.quadraticCurveTo(w * 0.44, h * 0.20, w * 0.46, h * 0.15);
        ctx.fill();

        // Afrika
        ctx.fillStyle = desertColor;
        ctx.beginPath();
        ctx.moveTo(w * 0.45, h * 0.32);
        ctx.quadraticCurveTo(w * 0.50, h * 0.30, w * 0.55, h * 0.33);
        ctx.quadraticCurveTo(w * 0.57, h * 0.45, w * 0.55, h * 0.55);
        ctx.quadraticCurveTo(w * 0.53, h * 0.65, w * 0.50, h * 0.70);
        ctx.quadraticCurveTo(w * 0.47, h * 0.65, w * 0.46, h * 0.55);
        ctx.quadraticCurveTo(w * 0.44, h * 0.42, w * 0.45, h * 0.32);
        ctx.fill();

        // Asya (büyük alan)
        ctx.fillStyle = landColor;
        ctx.beginPath();
        ctx.moveTo(w * 0.54, h * 0.12);
        ctx.quadraticCurveTo(w * 0.65, h * 0.10, w * 0.80, h * 0.14);
        ctx.quadraticCurveTo(w * 0.85, h * 0.18, w * 0.82, h * 0.25);
        ctx.quadraticCurveTo(w * 0.78, h * 0.30, w * 0.72, h * 0.35);
        ctx.quadraticCurveTo(w * 0.65, h * 0.38, w * 0.58, h * 0.36);
        ctx.quadraticCurveTo(w * 0.55, h * 0.30, w * 0.54, h * 0.22);
        ctx.quadraticCurveTo(w * 0.53, h * 0.16, w * 0.54, h * 0.12);
        ctx.fill();

        // Hindistan
        ctx.fillStyle = landHighColor;
        ctx.beginPath();
        ctx.moveTo(w * 0.65, h * 0.35);
        ctx.quadraticCurveTo(w * 0.68, h * 0.33, w * 0.70, h * 0.35);
        ctx.quadraticCurveTo(w * 0.69, h * 0.42, w * 0.67, h * 0.47);
        ctx.quadraticCurveTo(w * 0.65, h * 0.42, w * 0.65, h * 0.35);
        ctx.fill();

        // Avustralya
        ctx.fillStyle = desertColor;
        ctx.beginPath();
        ctx.moveTo(w * 0.78, h * 0.58);
        ctx.quadraticCurveTo(w * 0.84, h * 0.56, w * 0.88, h * 0.60);
        ctx.quadraticCurveTo(w * 0.88, h * 0.67, w * 0.84, h * 0.70);
        ctx.quadraticCurveTo(w * 0.80, h * 0.70, w * 0.78, h * 0.66);
        ctx.quadraticCurveTo(w * 0.76, h * 0.62, w * 0.78, h * 0.58);
        ctx.fill();

        // Japonya + adalar
        ctx.fillStyle = landColor;
        ctx.beginPath();
        ctx.arc(w * 0.84, h * 0.28, w * 0.015, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(w * 0.85, h * 0.26, w * 0.008, 0, Math.PI * 2);
        ctx.fill();

        // İngiltere
        ctx.beginPath();
        ctx.arc(w * 0.47, h * 0.20, w * 0.008, 0, Math.PI * 2);
        ctx.fill();

        // Endonezya / Güney Asya adaları
        for (let i = 0; i < 5; i++) {
            ctx.beginPath();
            ctx.arc(w * (0.74 + i * 0.02), h * 0.50, w * 0.006, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    createClouds() {
        const cloudCanvas = document.createElement('canvas');
        cloudCanvas.width = 1024;
        cloudCanvas.height = 512;
        const ctx = cloudCanvas.getContext('2d');

        // Rastgele bulut desenleri
        ctx.clearRect(0, 0, cloudCanvas.width, cloudCanvas.height);
        for (let i = 0; i < 200; i++) {
            const x = Math.random() * cloudCanvas.width;
            const y = Math.random() * cloudCanvas.height;
            const size = Math.random() * 60 + 10;
            const opacity = Math.random() * 0.15 + 0.02;

            const gradient = ctx.createRadialGradient(x, y, 0, x, y, size);
            gradient.addColorStop(0, `rgba(255, 255, 255, ${opacity})`);
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
            ctx.fillStyle = gradient;
            ctx.fillRect(x - size, y - size, size * 2, size * 2);
        }

        const cloudTexture = new THREE.CanvasTexture(cloudCanvas);
        cloudTexture.wrapS = THREE.RepeatWrapping;

        const cloudGeometry = new THREE.SphereGeometry(1.01, 48, 48);
        const cloudMaterial = new THREE.MeshPhongMaterial({
            map: cloudTexture,
            transparent: true,
            opacity: 0.4,
            depthWrite: false,
        });

        this.clouds = new THREE.Mesh(cloudGeometry, cloudMaterial);
        this.scene.add(this.clouds);
    }

    createAtmosphere() {
        // Atmosfer glow efekti (basit yaklaşım - ikinci küre)
        const atmosphereGeometry = new THREE.SphereGeometry(1.12, 48, 48);

        // Custom shader ile Fresnel efekti
        const atmosphereMaterial = new THREE.ShaderMaterial({
            vertexShader: `
                varying vec3 vNormal;
                varying vec3 vPosition;
                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                varying vec3 vNormal;
                varying vec3 vPosition;
                void main() {
                    float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
                    vec3 atmosphereColor = vec3(0.3, 0.6, 1.0);
                    gl_FragColor = vec4(atmosphereColor, intensity * 0.6);
                }
            `,
            blending: THREE.AdditiveBlending,
            side: THREE.BackSide,
            transparent: true,
            depthWrite: false,
        });

        this.atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
        this.scene.add(this.atmosphere);
    }

    createStars() {
        const starGeometry = new THREE.BufferGeometry();
        const starCount = 3000;
        const positions = new Float32Array(starCount * 3);
        const sizes = new Float32Array(starCount);

        for (let i = 0; i < starCount; i++) {
            // Rastgele pozisyon (kürenin uzağında)
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 50 + Math.random() * 100;

            positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = r * Math.cos(phi);

            sizes[i] = Math.random() * 2 + 0.5;
        }

        starGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        starGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        const starMaterial = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.5,
            transparent: true,
            opacity: 0.9,
            sizeAttenuation: true,
        });

        this.stars = new THREE.Points(starGeometry, starMaterial);
        this.scene.add(this.stars);
    }

    addLighting() {
        // Ana ışık (güneş)
        const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
        sunLight.position.set(5, 3, 5);
        this.scene.add(sunLight);

        // Ortam ışığı
        const ambientLight = new THREE.AmbientLight(0x333344, 0.5);
        this.scene.add(ambientLight);

        // Arka ışık (gece tarafını hafif aydınlat)
        const backLight = new THREE.DirectionalLight(0x334466, 0.15);
        backLight.position.set(-5, -2, -5);
        this.scene.add(backLight);
    }

    // ---- GESTURE KONTROL API'SI ----

    rotate(deltaX, deltaY) {
        if (this.isLocked) return;
        this.autoRotate = false;
        this.targetRotationY += deltaX * 0.08;
        this.targetRotationX += deltaY * 0.08;

        // X rotasyonunu sınırla
        this.targetRotationX = Math.max(-1.2, Math.min(1.2, this.targetRotationX));
    }

    zoom(delta) {
        if (this.isLocked) return;
        this.targetZoom += delta * 0.15;
        this.targetZoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.targetZoom));
    }

    lock() {
        this.isLocked = true;
        this.autoRotate = false;
    }

    unlock() {
        this.isLocked = false;
    }

    reset() {
        this.targetRotationX = 0.3;
        this.targetRotationY = 0;
        this.targetZoom = 3.5;
        this.autoRotate = true;
        this.isLocked = false;
    }

    // ---- ANİMASYON DÖNGÜSÜ ----

    animate() {
        requestAnimationFrame(() => this.animate());

        // FPS hesapla
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastFpsTime = now;

            const fpsEl = document.getElementById('fps-value');
            if (fpsEl) fpsEl.textContent = this.fps;
        }

        // Auto rotate
        if (this.autoRotate && !this.isLocked) {
            this.targetRotationY += this.autoRotateSpeed;
        }

        // Smooth interpolation (lerp)
        this.currentRotationX += (this.targetRotationX - this.currentRotationX) * this.rotationSmoothing;
        this.currentRotationY += (this.targetRotationY - this.currentRotationY) * this.rotationSmoothing;
        this.currentZoom += (this.targetZoom - this.currentZoom) * this.zoomSmoothing;

        // Globe rotasyonu
        if (this.globe) {
            this.globe.rotation.x = this.currentRotationX;
            this.globe.rotation.y = this.currentRotationY;
        }

        // Bulutlar biraz daha yavaş döner
        if (this.clouds) {
            this.clouds.rotation.x = this.currentRotationX;
            this.clouds.rotation.y = this.currentRotationY + 0.0003 * performance.now() * 0.001;
        }

        // Kamera zoom
        this.camera.position.z = this.currentZoom;

        // Yıldızlar hafifçe döner
        if (this.stars) {
            this.stars.rotation.y += 0.00005;
        }

        this.renderer.render(this.scene, this.camera);
    }

    onResize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }
}

// Global instance
let globeController;

function initGlobe() {
    globeController = new GlobeController('globe-container');
    return globeController;
}
