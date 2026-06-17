/**
 * 🌹 Rose Garden — HD Canvas 2D Interactive Hand Roses
 * ═══════════════════════════════════════════════════════════════
 * Pixel-perfect fingertip alignment · Gradient-shaded bezier petals
 * Soft bloom glow · Floating pollen · Gesture interactions
 *
 * Canvas 2D gives us: direct pixel coordinates from MediaPipe,
 * gradient fills for realistic petals, shadowBlur for natural glow,
 * and compositing modes for bloom — no coordinate mapping issues.
 */

/* ═══════════════════════════════════════════════════════════════
   CONFIGURATION
   ═══════════════════════════════════════════════════════════════ */

const CFG = {
    roseSize:     28,        // base rose radius (px) — small & elegant
    petalLayers:  4,         // concentric petal rings
    stemLength:   40,        // stem length from fingertip
    bloomSpeed:   0.025,     // left-hand bloom rate per frame
    glowAlpha:    0.25,      // glow intensity (subtle, not tacky)
    glowBlur:     12,        // shadowBlur for rose halo
    bgDarken:     0.42,      // webcam darkening (0=black, 1=full)
    smoothing:    0.35,      // landmark EMA alpha
    pollenMax:    150,       // max floating pollen particles
    pollenRate:   0.06,      // spawn probability per rose per frame
    trailMax:     40,        // max trail dots
    trailFade:    0.025,     // trail opacity decay per frame
    burstCount:   16,        // petals in pinch burst
    bloomPassAlpha: 0.25,    // screen-space bloom composite alpha
};

/* Five red-rose varieties — rich, natural tones (not garish) */
const ROSE_COLORS = [
    { outer: [190, 32, 46],  mid: [155, 20, 35],  inner: [115, 12, 25],  name: 'Classic'  },
    { outer: [205, 38, 42],  mid: [170, 25, 32],  inner: [125, 15, 22],  name: 'Scarlet'  },
    { outer: [145, 22, 38],  mid: [110, 14, 28],  inner: [78,   8, 18],  name: 'Burgundy' },
    { outer: [210, 42, 58],  mid: [175, 28, 45],  inner: [130, 18, 32],  name: 'Crimson'  },
    { outer: [180, 28, 55],  mid: [145, 18, 42],  inner: [105, 10, 30],  name: 'Ruby'     },
];

const FINGERTIP  = [4, 8, 12, 16, 20];
const FINGER_PIP = [3, 6, 10, 14, 18];

/* Layer configs: each ring of petals */
const LAYER_CFG = [
    { n: 8, rFrac: 1.00, angleOff: 0,    dark: 0,    alpha: 0.88 },  // outermost
    { n: 7, rFrac: 0.76, angleOff: 0.38, dark: 0.12, alpha: 0.90 },
    { n: 6, rFrac: 0.54, angleOff: 0.18, dark: 0.24, alpha: 0.92 },
    { n: 4, rFrac: 0.34, angleOff: 0.50, dark: 0.34, alpha: 0.94 },  // innermost
];


/* ═══════════════════════════════════════════════════════════════
   UTILITIES
   ═══════════════════════════════════════════════════════════════ */

const lerp = (a, b, t) => a + (b - a) * t;
const lmDist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);


/* ═══════════════════════════════════════════════════════════════
   POLLEN PARTICLE SYSTEM
   ═══════════════════════════════════════════════════════════════ */

class PollenSystem {
    constructor() {
        this.particles = [];
    }

    emit(x, y, color, count = 1) {
        for (let i = 0; i < count && this.particles.length < CFG.pollenMax; i++) {
            this.particles.push({
                x: x + (Math.random() - 0.5) * 20,
                y: y + (Math.random() - 0.5) * 20,
                vx: (Math.random() - 0.5) * 0.8,
                vy: -Math.random() * 1.2 - 0.3,
                life: 1.0,
                decay: 0.004 + Math.random() * 0.004,
                size: 1.5 + Math.random() * 2.5,
                color,
            });
        }
    }

    update(time) {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.life -= p.decay;
            if (p.life <= 0) { this.particles.splice(i, 1); continue; }
            p.x += p.vx + Math.sin(time * 1.5 + p.x * 0.02) * 0.3;
            p.y += p.vy;
            p.vy *= 0.998;
        }
    }

    draw(ctx) {
        for (const p of this.particles) {
            const a = p.life * 0.6;
            ctx.globalAlpha = a;
            ctx.fillStyle = `rgba(${p.color[0]},${p.color[1]},${p.color[2]},1)`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = 1;
    }
}


/* ═══════════════════════════════════════════════════════════════
   TRAIL SYSTEM
   ═══════════════════════════════════════════════════════════════ */

class TrailSystem {
    constructor() { this.dots = []; }

    add(x, y) {
        if (this.dots.length >= CFG.trailMax) return;
        this.dots.push({ x, y, alpha: 0.3, size: 6 + Math.random() * 4 });
    }

    update() {
        for (let i = this.dots.length - 1; i >= 0; i--) {
            this.dots[i].alpha -= CFG.trailFade;
            if (this.dots[i].alpha <= 0) this.dots.splice(i, 1);
        }
    }

    draw(ctx) {
        for (const d of this.dots) {
            const g = ctx.createRadialGradient(d.x, d.y, 0, d.x, d.y, d.size);
            g.addColorStop(0, `rgba(200,60,70,${d.alpha})`);
            g.addColorStop(1, `rgba(150,30,40,0)`);
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.arc(d.x, d.y, d.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
   BURST EFFECT (pinch gesture)
   ═══════════════════════════════════════════════════════════════ */

class BurstEffect {
    constructor() { this.petals = []; }

    trigger(x, y, color) {
        for (let i = 0; i < CFG.burstCount; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = 2.5 + Math.random() * 3;
            this.petals.push({
                x, y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                rot: Math.random() * Math.PI * 2,
                vr: (Math.random() - 0.5) * 0.3,
                life: 1.0,
                size: 5 + Math.random() * 6,
                color,
            });
        }
    }

    update() {
        for (let i = this.petals.length - 1; i >= 0; i--) {
            const p = this.petals[i];
            p.life -= 0.02;
            if (p.life <= 0) { this.petals.splice(i, 1); continue; }
            p.x += p.vx;
            p.y += p.vy;
            p.vy += 0.12; // gravity
            p.rot += p.vr;
            p.vx *= 0.97;
            p.vy *= 0.97;
        }
    }

    draw(ctx) {
        for (const p of this.petals) {
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rot);
            ctx.globalAlpha = p.life * 0.85;
            const c = p.color;
            const grad = ctx.createLinearGradient(0, -p.size, 0, p.size);
            grad.addColorStop(0, `rgba(${c[0]+40},${c[1]+20},${c[2]+15},0.9)`);
            grad.addColorStop(1, `rgba(${c[0]},${c[1]},${c[2]},0.5)`);
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.moveTo(0, -p.size);
            ctx.bezierCurveTo(p.size*0.6, -p.size*0.5, p.size*0.5, p.size*0.5, 0, p.size);
            ctx.bezierCurveTo(-p.size*0.5, p.size*0.5, -p.size*0.6, -p.size*0.5, 0, -p.size);
            ctx.fill();
            ctx.restore();
        }
        ctx.globalAlpha = 1;
    }
}


/* ═══════════════════════════════════════════════════════════════
   ROSE RENDERER  (gradient bezier petals on Canvas 2D)
   ═══════════════════════════════════════════════════════════════ */

function drawRose(ctx, x, y, size, bloom, fingerIdx, time) {
    if (bloom < 0.05) return;
    const pal = ROSE_COLORS[fingerIdx];

    ctx.save();
    ctx.translate(x, y);

    // ── Subtle glow halo ───────────────────────────────────────
    ctx.shadowColor = `rgba(${pal.outer[0]}, ${pal.outer[1]+30}, ${pal.outer[2]+20}, ${CFG.glowAlpha * bloom})`;
    ctx.shadowBlur = CFG.glowBlur * bloom;

    // ── Stem (curved, thin, organic) ───────────────────────────
    const stemLen = CFG.stemLength * bloom;
    ctx.strokeStyle = `rgba(50, 105, 50, ${0.45 * bloom})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.quadraticCurveTo(4, stemLen * 0.6, -2, stemLen);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // ── Small leaf on stem ─────────────────────────────────────
    if (bloom > 0.3) {
        const leafY = stemLen * 0.55;
        ctx.save();
        ctx.translate(2, leafY);
        ctx.rotate(0.5);
        ctx.fillStyle = `rgba(55, 115, 55, ${0.45 * bloom})`;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.quadraticCurveTo(7, -3, 13, 0);
        ctx.quadraticCurveTo(7, 3, 0, 0);
        ctx.fill();
        ctx.restore();
    }

    // ── Multi-layered petals ───────────────────────────────────
    ctx.shadowColor = `rgba(${pal.outer[0]}, ${pal.outer[1]+20}, ${pal.outer[2]+10}, ${CFG.glowAlpha * bloom * 0.6})`;
    ctx.shadowBlur = CFG.glowBlur * bloom * 0.5;

    const baseRot = time * 0.15 + fingerIdx * 1.1;

    for (let li = 0; li < CFG.petalLayers; li++) {
        const L = LAYER_CFG[li];
        const layerR = size * L.rFrac * bloom;
        if (layerR < 2) continue;

        // Pick colour for this layer (outer→mid→inner blend)
        const d = 1 - L.dark;
        let rgb;
        if (li < 2) rgb = [lerp(pal.outer[0], pal.mid[0], L.dark*2)|0, lerp(pal.outer[1], pal.mid[1], L.dark*2)|0, lerp(pal.outer[2], pal.mid[2], L.dark*2)|0];
        else         rgb = [lerp(pal.mid[0], pal.inner[0], (L.dark-0.2)*3)|0, lerp(pal.mid[1], pal.inner[1], (L.dark-0.2)*3)|0, lerp(pal.mid[2], pal.inner[2], (L.dark-0.2)*3)|0];

        for (let i = 0; i < L.n; i++) {
            const angle = (i / L.n) * Math.PI * 2 + L.angleOff + baseRot;
            const petalH = layerR * 0.52;
            const petalW = layerR * 0.30;

            ctx.save();
            ctx.rotate(angle);

            // Gradient: deep at base → lighter at tip (like real rose petal)
            const grad = ctx.createLinearGradient(0, 0, 0, -petalH);
            grad.addColorStop(0,   `rgba(${rgb[0]-20}, ${rgb[1]}, ${rgb[2]}, ${L.alpha})`);
            grad.addColorStop(0.5, `rgba(${rgb[0]+15}, ${rgb[1]+8}, ${rgb[2]+5}, ${L.alpha - 0.05})`);
            grad.addColorStop(1,   `rgba(${rgb[0]+40}, ${rgb[1]+25}, ${rgb[2]+20}, ${L.alpha - 0.2})`);
            ctx.fillStyle = grad;

            // Petal bezier shape (rounded, natural rose petal)
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.bezierCurveTo( petalW,  -petalH * 0.28,  petalW * 0.85, -petalH * 0.72,  0, -petalH);
            ctx.bezierCurveTo(-petalW * 0.85, -petalH * 0.72, -petalW, -petalH * 0.28,  0, 0);
            ctx.fill();

            ctx.restore();
        }
    }

    ctx.shadowBlur = 0;

    // ── Golden center (stamens) ────────────────────────────────
    if (bloom > 0.25) {
        const cr = size * 0.09 * bloom;
        const cg = ctx.createRadialGradient(0, 0, 0, 0, 0, cr);
        cg.addColorStop(0, `rgba(255, 235, 130, ${bloom * 0.9})`);
        cg.addColorStop(0.6, `rgba(220, 185, 70, ${bloom * 0.6})`);
        cg.addColorStop(1, `rgba(180, 140, 40, 0)`);
        ctx.fillStyle = cg;
        ctx.beginPath();
        ctx.arc(0, 0, cr, 0, Math.PI * 2);
        ctx.fill();
    }

    ctx.restore();
}


/* ── Big rose (hands together) ────────────────────────────────── */

function drawBigRose(ctx, x, y, size, bloom, time) {
    if (bloom < 0.02) return;
    const pal = ROSE_COLORS[0]; // classic red

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(time * 0.08);

    // Larger glow
    ctx.shadowColor = `rgba(200, 50, 60, ${0.3 * bloom})`;
    ctx.shadowBlur = 20 * bloom;

    // Extra-large rose with 5 layers
    const layers = [
        { n: 10, rFrac: 1.0,  angleOff: 0,    dark: 0    },
        { n: 8,  rFrac: 0.80, angleOff: 0.3,  dark: 0.10 },
        { n: 7,  rFrac: 0.62, angleOff: 0.15, dark: 0.20 },
        { n: 6,  rFrac: 0.44, angleOff: 0.45, dark: 0.30 },
        { n: 4,  rFrac: 0.28, angleOff: 0.25, dark: 0.40 },
    ];

    for (const L of layers) {
        const r = size * L.rFrac * bloom;
        if (r < 3) continue;
        const d = 1 - L.dark;
        const rgb = [pal.outer[0]*d|0, pal.outer[1]*d|0, pal.outer[2]*d|0];

        for (let i = 0; i < L.n; i++) {
            const angle = (i / L.n) * Math.PI * 2 + L.angleOff + time * 0.1;
            const pH = r * 0.5, pW = r * 0.28;
            ctx.save();
            ctx.rotate(angle);
            const grad = ctx.createLinearGradient(0, 0, 0, -pH);
            grad.addColorStop(0, `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.9)`);
            grad.addColorStop(1, `rgba(${rgb[0]+35}, ${rgb[1]+20}, ${rgb[2]+15}, 0.55)`);
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.bezierCurveTo(pW, -pH*0.28, pW*0.85, -pH*0.72, 0, -pH);
            ctx.bezierCurveTo(-pW*0.85, -pH*0.72, -pW, -pH*0.28, 0, 0);
            ctx.fill();
            ctx.restore();
        }
    }

    ctx.shadowBlur = 0;
    // Center
    const cr = size * 0.07 * bloom;
    const cg = ctx.createRadialGradient(0, 0, 0, 0, 0, cr);
    cg.addColorStop(0, `rgba(255,235,130,${bloom*0.9})`);
    cg.addColorStop(1, `rgba(200,160,50,0)`);
    ctx.fillStyle = cg;
    ctx.beginPath();
    ctx.arc(0, 0, cr, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
}


/* ═══════════════════════════════════════════════════════════════
   GESTURE DETECTOR
   ═══════════════════════════════════════════════════════════════ */

function isOpenPalm(lm) {
    const wrist = lm[0];
    let ext = 0;
    for (let i = 0; i < 5; i++)
        if (lmDist(lm[FINGERTIP[i]], wrist) > lmDist(lm[FINGER_PIP[i]], wrist) * 1.05) ext++;
    return ext >= 4;
}

function isPinch(lm) { return lmDist(lm[4], lm[8]) < 0.06; }

function palmCenter(lm) {
    return {
        x: (lm[0].x + lm[5].x + lm[17].x) / 3,
        y: (lm[0].y + lm[5].y + lm[17].y) / 3,
    };
}


/* ═══════════════════════════════════════════════════════════════
   MAIN APPLICATION
   ═══════════════════════════════════════════════════════════════ */

class RoseGardenApp {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx    = this.canvas.getContext('2d');
        this.video  = document.getElementById('video');
        this.loadEl = document.getElementById('loading');

        // Off-screen canvas for bloom composite
        this.bloomCanvas = document.createElement('canvas');
        this.bloomCtx    = this.bloomCanvas.getContext('2d');

        // Systems
        this.pollen = new PollenSystem();
        this.trails = new TrailSystem();
        this.burst  = new BurstEffect();

        // Hand state
        this.handResults = null;
        this.smoothed = [null, null]; // per-hand smoothed pixel coords [21][{x,y}]
        this.bloom = [[0,0,0,0,0], [0,0,0,0,0]]; // per-finger bloom progress
        this.pinchCD = [0, 0];
        this.bigBloomProgress = 0;
        this.trailN = 0;

        // Timing
        this.time = 0;
        this.lastTime = 0;
        this.frameCount = 0;
        this.fpsTime = 0;
        this.fpsText = '';
    }

    async init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());

        await this.setupCamera();
        this.animate(performance.now());
    }

    resize() {
        const dpr = Math.min(window.devicePixelRatio, 2);
        const w = window.innerWidth, h = window.innerHeight;
        this.canvas.width  = w * dpr;
        this.canvas.height = h * dpr;
        this.canvas.style.width  = w + 'px';
        this.canvas.style.height = h + 'px';
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.W = w;
        this.H = h;
        // Bloom canvas at 1/4 res
        this.bloomCanvas.width  = (w * dpr / 4) | 0;
        this.bloomCanvas.height = (h * dpr / 4) | 0;
    }

    async setupCamera() {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
            audio: false,
        });
        this.video.srcObject = stream;
        await this.video.play();

        const hands = new window.Hands({
            locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
        });
        hands.setOptions({
            maxNumHands: 2, modelComplexity: 1,
            minDetectionConfidence: 0.65, minTrackingConfidence: 0.50,
        });
        hands.onResults(r => { this.handResults = r; });

        const cam = new window.Camera(this.video, {
            onFrame: async () => { await hands.send({ image: this.video }); },
            width: 1280, height: 720,
        });
        await cam.start();
        this.loadEl.classList.add('hidden');
    }

    /* ── Convert landmark to mirrored pixel coords ─────────── */
    lmToPx(lm) {
        return { x: (1 - lm.x) * this.W, y: lm.y * this.H };
    }

    /* ── Smooth landmarks with EMA ─────────────────────────── */
    smoothLandmarks(slot, lms) {
        const a = CFG.smoothing;
        if (!this.smoothed[slot]) {
            this.smoothed[slot] = lms.map(l => this.lmToPx(l));
        } else {
            for (let i = 0; i < 21; i++) {
                const target = this.lmToPx(lms[i]);
                this.smoothed[slot][i].x = lerp(this.smoothed[slot][i].x, target.x, a);
                this.smoothed[slot][i].y = lerp(this.smoothed[slot][i].y, target.y, a);
            }
        }
        return this.smoothed[slot];
    }

    /* ── Main render ───────────────────────────────────────── */

    animate(now) {
        requestAnimationFrame(t => this.animate(t));
        const dt = Math.min((now - this.lastTime) / 1000, 0.05);
        this.lastTime = now;
        this.time += dt;

        const ctx = this.ctx;
        const W = this.W, H = this.H;

        // ── 1. Webcam background (mirrored, darkened) ──────────
        ctx.save();
        ctx.translate(W, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(this.video, 0, 0, W, H);
        ctx.restore();

        // Darken overlay
        ctx.fillStyle = `rgba(5, 3, 8, ${1 - CFG.bgDarken})`;
        ctx.fillRect(0, 0, W, H);

        // ── 2. Process hands ───────────────────────────────────
        const res = this.handResults;
        const hands = res?.multiHandLandmarks;
        const hd    = res?.multiHandedness;
        const activeSlots = new Set();

        if (hands) {
            for (let hi = 0; hi < hands.length && hi < 2; hi++) {
                const lm = hands[hi];
                const label = hd?.[hi]?.label || 'Right';
                const isLeft = (label === 'Left');
                const si = isLeft ? 0 : 1;
                activeSlots.add(si);

                const pts = this.smoothLandmarks(si, lm);

                // Gestures
                const palmOpen = isOpenPalm(lm);
                const pinch = isPinch(lm);

                // Pinch burst
                if (pinch && this.pinchCD[si] <= 0) {
                    const mid = {
                        x: (pts[4].x + pts[8].x) / 2,
                        y: (pts[4].y + pts[8].y) / 2,
                    };
                    this.burst.trigger(mid.x, mid.y, ROSE_COLORS[1].outer);
                    this.pollen.emit(mid.x, mid.y, [255, 180, 180], 6);
                    this.pinchCD[si] = 25; // frames cooldown
                }
                if (this.pinchCD[si] > 0) this.pinchCD[si]--;

                // ── Fingertip roses ────────────────────────────
                for (let fi = 0; fi < 5; fi++) {
                    const tip = pts[FINGERTIP[fi]];

                    // Bloom progress
                    if (isLeft) {
                        const target = palmOpen ? 0.85 : 0.12;
                        const speed = palmOpen ? CFG.bloomSpeed : -CFG.bloomSpeed * 0.6;
                        this.bloom[si][fi] = Math.max(0, Math.min(0.85,
                            this.bloom[si][fi] + speed));
                    } else {
                        this.bloom[si][fi] = Math.min(1, this.bloom[si][fi] + 0.04);
                    }

                    const b = this.bloom[si][fi];
                    drawRose(ctx, tip.x, tip.y, CFG.roseSize, b, fi, this.time);

                    // Emit pollen from bloomed roses
                    if (b > 0.4 && Math.random() < CFG.pollenRate)
                        this.pollen.emit(tip.x, tip.y, ROSE_COLORS[fi].outer, 1);
                }

                // ── Trail from palm movement ───────────────────
                this.trailN++;
                if (this.trailN % 5 === 0) {
                    const pc = palmCenter(lm);
                    this.trails.add((1 - pc.x) * W, pc.y * H);
                }
            }
        }

        // Deactivate absent hands
        for (let s = 0; s < 2; s++) {
            if (!activeSlots.has(s)) {
                this.smoothed[s] = null;
                for (let fi = 0; fi < 5; fi++)
                    this.bloom[s][fi] = Math.max(0, this.bloom[s][fi] - 0.03);
            }
        }

        // ── Big rose (hands together) ──────────────────────────
        if (hands && hands.length >= 2 && this.smoothed[0] && this.smoothed[1]) {
            const c0 = this.smoothed[0][0], c1 = this.smoothed[1][0];
            const dist = Math.hypot(c0.x - c1.x, c0.y - c1.y);
            if (dist < W * 0.2) {
                this.bigBloomProgress = Math.min(1, this.bigBloomProgress + 0.03);
                const mx = (c0.x + c1.x) / 2, my = (c0.y + c1.y) / 2;
                drawBigRose(ctx, mx, my, 55, this.bigBloomProgress, this.time);
                if (Math.random() < 0.1)
                    this.pollen.emit(mx, my, [255, 160, 170], 2);
            } else {
                this.bigBloomProgress = Math.max(0, this.bigBloomProgress - 0.025);
            }
        } else {
            this.bigBloomProgress = Math.max(0, this.bigBloomProgress - 0.025);
        }

        // ── 3. Draw effects ────────────────────────────────────
        this.trails.update();
        this.trails.draw(ctx);
        this.burst.update();
        this.burst.draw(ctx);
        this.pollen.update(this.time);
        this.pollen.draw(ctx);

        // ── 4. Screen-space bloom pass ─────────────────────────
        this.applyBloom(ctx, W, H);

        // ── 5. FPS ─────────────────────────────────────────────
        this.frameCount++;
        this.fpsTime += dt;
        if (this.frameCount % 30 === 0) {
            this.fpsText = `${Math.round(30 / this.fpsTime)} FPS`;
            this.fpsTime = 0;
        }
        if (this.fpsText) {
            ctx.font = '11px Inter, system-ui, sans-serif';
            ctx.fillStyle = 'rgba(120,220,120,0.5)';
            ctx.fillText(this.fpsText, 14, 24);
        }
    }

    /* ── Screen-space bloom (downsample → blur → composite) ── */

    applyBloom(ctx, W, H) {
        const bc = this.bloomCtx;
        const bw = this.bloomCanvas.width, bh = this.bloomCanvas.height;

        // Downsample main canvas
        bc.drawImage(this.canvas, 0, 0, bw, bh);

        // Blur (CSS filter on Canvas 2D context)
        bc.filter = 'blur(6px)';
        bc.drawImage(this.bloomCanvas, 0, 0, bw, bh);
        bc.filter = 'none';

        // Composite back with 'screen' blending
        ctx.save();
        ctx.globalCompositeOperation = 'screen';
        ctx.globalAlpha = CFG.bloomPassAlpha;
        ctx.drawImage(this.bloomCanvas, 0, 0, W, H);
        ctx.restore();
    }
}


/* ═══════════════════════════════════════════════════════════════
   LAUNCH
   ═══════════════════════════════════════════════════════════════ */

const app = new RoseGardenApp();
app.init().catch(err => {
    console.error('Rose Garden error:', err);
    const el = document.getElementById('loading');
    if (el) {
        el.querySelector('.loader-text').textContent = 'Could not start';
        el.querySelector('.loader-sub').textContent = err.message || 'Check camera permissions';
    }
});
