#!/usr/bin/env python3
"""
🌺 Generative Flower Garden — TouchDesigner-style Interactive Flower Effect
════════════════════════════════════════════════════════════════════════════
Real-time webcam app: glowing animated flowers and curved stems shoot
outward from each fingertip in a fan/spray pattern.

Requirements: pip install opencv-python mediapipe numpy
Run:          python flower_garden.py
Quit:         press 'q'
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import random as rng
import threading
import queue

# ─── MediaPipe Tasks API (v0.10+) ─────────────────────────────────────────
BaseOptions       = mp.tasks.BaseOptions
HandLandmarker    = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
RunMode           = mp.tasks.vision.RunningMode


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CAM_W, CAM_H       = 1280, 720
MAX_HANDS           = 2
SMOOTH_ALPHA        = 0.45          # EMA smoothing for landmark jitter
GROW_FRAMES         = 22            # Frames for stem growth
HOLD_FRAMES         = 6             # Frames to hold full bloom
CYCLE               = GROW_FRAMES + HOLD_FRAMES
STEMS_MIN, STEMS_MAX = 3, 5        # Stems per finger per cycle
STEM_MAX_LEN        = 130           # Max stem length px
FAN_SPREAD          = math.pi * 0.50  # ~90° fan
PETAL_N             = 5             # Petals per flower head
PETAL_LEN           = 18            # Petal length
PETAL_W             = 6             # Petal width at shoulder
BG_ALPHA            = 0.50          # Background dim
PALM_LINES          = 5             # Crossing lines on palm
PALM_RESHUFFLE      = 50            # Re-randomise palm pairs every N frames
CONF_THRESH         = 0.65          # Min hand confidence to render

FINGERTIP  = [4, 8, 12, 16, 20]
KNUCKLE    = [5, 6, 9, 10, 13, 14, 17, 18]   # MCP + PIP


# ═══════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTES  (BGR)
# ═══════════════════════════════════════════════════════════════════════════
#  Matching the TouchDesigner reference: warm stems/petals, cool palm lines.

COLORS = [
    {   # Thumb — deep orange / flame
        "stem": (0, 90, 235),  "pet1": (0, 55, 255),
        "pet2": (0, 120, 240), "glow": (0, 70, 200),
        "core": (190, 215, 255),
    },
    {   # Index — golden yellow / amber
        "stem": (0, 195, 255), "pet1": (0, 170, 250),
        "pet2": (10, 215, 255),"glow": (0, 175, 230),
        "core": (175, 240, 255),
    },
    {   # Middle — hot magenta / violet
        "stem": (190, 0, 200), "pet1": (255, 20, 200),
        "pet2": (200, 40, 180),"glow": (170, 0, 160),
        "core": (255, 190, 245),
    },
    {   # Ring — teal / cyan
        "stem": (180, 180, 0), "pet1": (230, 230, 0),
        "pet2": (195, 205, 0), "glow": (155, 155, 0),
        "core": (225, 255, 200),
    },
    {   # Pinky — coral / pink
        "stem": (95, 125, 240),"pet1": (140, 145, 255),
        "pet2": (115, 135, 248),"glow":(80, 105, 215),
        "core": (205, 195, 255),
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  THREADED CAMERA
# ═══════════════════════════════════════════════════════════════════════════

class CamThread:
    """Background thread grabs frames so the render loop never waits on I/O."""

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.q = queue.Queue(maxsize=2)
        self.stopped = False

    @property
    def opened(self):
        return self.cap.isOpened()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                self.stopped = True; break
            if self.q.full():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(f)

    def read(self):
        try: return self.q.get(timeout=1)
        except queue.Empty: return None

    def release(self):
        self.stopped = True; self.cap.release()


# ═══════════════════════════════════════════════════════════════════════════
#  LANDMARK SMOOTHER  (EMA)
# ═══════════════════════════════════════════════════════════════════════════

class Smoother:
    def __init__(self, a=SMOOTH_ALPHA):
        self.a = a; self.p = None

    def __call__(self, lm, w, h):
        raw = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float64)
        if self.p is None: self.p = raw.copy()
        else:              self.p += (raw - self.p) * self.a
        return self.p.copy()

    def reset(self): self.p = None


# ═══════════════════════════════════════════════════════════════════════════
#  FINGER BLOOM  (growth cycle state machine)
# ═══════════════════════════════════════════════════════════════════════════

class Bloom:
    """Per-finger animation: grow → hold → randomise → repeat."""

    def __init__(self):
        self.t = 0
        self.ns = 0
        self.ang = []          # angle offsets per stem
        self.mlen = []         # max-length per stem
        self.curv = []         # curvature per stem
        self._rand()

    def _rand(self):
        self.ns = rng.randint(STEMS_MIN, STEMS_MAX)
        half = FAN_SPREAD / 2
        base = np.linspace(-half, half, self.ns)
        self.ang  = [float(a + rng.uniform(-0.10, 0.10)) for a in base]
        self.mlen = [STEM_MAX_LEN * rng.uniform(0.72, 1.0) for _ in range(self.ns)]
        self.curv = [rng.choice([-1, 1]) * rng.uniform(0.12, 0.32) for _ in range(self.ns)]

    def tick(self):
        self.t += 1
        if self.t >= CYCLE:
            self.t = 0; self._rand()

    @property
    def g(self):
        """Growth 0→1 with cubic ease-out."""
        if self.t < GROW_FRAMES:
            r = self.t / GROW_FRAMES
            return 1.0 - (1.0 - r) ** 3
        return 1.0

    def reset(self):
        self.t = 0; self._rand()


# ═══════════════════════════════════════════════════════════════════════════
#  FAST DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _sc(c, f):
    """Scale BGR colour by factor, clamped."""
    return (max(0, min(255, int(c[0]*f))),
            max(0, min(255, int(c[1]*f))),
            max(0, min(255, int(c[2]*f))))


# ── Bezier curve sampler ──────────────────────────────────────────────────

def _bezier(sx, sy, cx, cy, ex, ey, n=14):
    """Return (n+1, 1, 2) int32 array for cv2.polylines."""
    ts = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    inv = 1.0 - ts
    xs = (inv*inv*sx + 2*inv*ts*cx + ts*ts*ex).astype(np.int32)
    ys = (inv*inv*sy + 2*inv*ts*cy + ts*ts*ey).astype(np.int32)
    return np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)


# ── Glow stem (3-pass blur stack — NO heavy GaussianBlur) ─────────────────

def draw_stem(img, sx, sy, ex, ey, curv, color):
    """Curved stem with 3-pass glow: thick-faint → medium → thin-bright."""
    mx, my = (sx + ex) / 2, (sy + ey) / 2
    d = max(1.0, math.hypot(ex - sx, ey - sy))
    px, py = -(ey - sy) / d, (ex - sx) / d   # perpendicular
    off = d * curv
    cxp, cyp = mx + px * off, my + py * off
    pts = _bezier(sx, sy, cxp, cyp, ex, ey)
    cv2.polylines(img, [pts], False, _sc(color, 0.22), 7, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, _sc(color, 0.50), 3, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, color,             1, cv2.LINE_AA)
    return int(pts[-1][0][0]), int(pts[-1][0][1])


# ── Pointed petal (tulip / lily shape, polygon fill) ──────────────────────

def draw_petal(img, cx, cy, angle, length, width, color):
    """Elongated pointed petal: narrow base → shoulder → sharp tip."""
    ca, sa = math.cos(angle), math.sin(angle)
    cp, sp = math.cos(angle + 1.5708), math.sin(angle + 1.5708)  # +π/2
    hw = width * 0.5
    sf = 0.35  # shoulder fraction along length
    # shoulder centre
    smx, smy = cx + ca * length * sf, cy + sa * length * sf
    # tip
    tx, ty = cx + ca * length, cy + sa * length
    pts = np.array([
        [int(cx),                int(cy)],
        [int(smx + cp * hw),     int(smy + sp * hw)],
        [int(tx),                int(ty)],
        [int(smx - cp * hw),     int(smy - sp * hw)],
    ], dtype=np.int32)
    # glow pass (slightly larger)
    g_hw = hw + 2
    gpts = np.array([
        [int(cx),                         int(cy)],
        [int(smx + cp * g_hw),            int(smy + sp * g_hw)],
        [int(cx + ca * (length + 3)),     int(cy + sa * (length + 3))],
        [int(smx - cp * g_hw),            int(smy - sp * g_hw)],
    ], dtype=np.int32)
    cv2.fillPoly(img, [gpts], _sc(color, 0.28), cv2.LINE_AA)
    cv2.fillPoly(img, [pts],  color,             cv2.LINE_AA)


# ── Complete flower head ──────────────────────────────────────────────────

def draw_flower(img, cx, cy, t, fi, sc=1.0):
    """Render a flower head with pointed petals + white-hot glowing core."""
    pal = COLORS[fi]
    rot = t * 0.45 + fi * 1.25
    pl, pw = PETAL_LEN * sc, PETAL_W * sc
    if pl < 3:
        return
    for i in range(PETAL_N):
        a = rot + 2 * math.pi * i / PETAL_N
        c = pal["pet1"] if i % 2 == 0 else pal["pet2"]
        draw_petal(img, cx, cy, a, pl, pw, c)
    # Glowing centre: halo → core → white
    cr = max(2, int(4 * sc))
    cv2.circle(img, (cx, cy), cr * 3, _sc(pal["glow"], 0.30), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), cr * 2, _sc(pal["core"], 0.65), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), cr,     (255, 255, 255),         -1, cv2.LINE_AA)


# ── Palm crossing lines (blue / purple shimmer) ──────────────────────────

def draw_palm_lines(img, pts, pairs, t):
    """Thin iridescent lines crossing the palm between knuckle landmarks."""
    for (a, b) in pairs:
        if a >= len(pts) or b >= len(pts):
            continue
        p1 = (int(pts[a][0]), int(pts[a][1]))
        p2 = (int(pts[b][0]), int(pts[b][1]))
        sh = 0.40 + 0.40 * math.sin(t * 5.5 + a * 1.1)
        hue = int((t * 28 + a * 35) % 180)
        hsv = np.uint8([[[hue, 160, int(185 * sh)]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        col = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        # 3-pass glow
        cv2.line(img, p1, p2, _sc(col, 0.22), 5, cv2.LINE_AA)
        cv2.line(img, p1, p2, _sc(col, 0.50), 2, cv2.LINE_AA)
        cv2.line(img, p1, p2, col,             1, cv2.LINE_AA)


# ── Lightweight hand skeleton (thin coloured lines on fingers) ────────────

HAND_CONNS = [
    # Thumb
    (0,1),(1,2),(2,3),(3,4),
    # Index
    (0,5),(5,6),(6,7),(7,8),
    # Middle
    (0,9),(9,10),(10,11),(11,12),
    # Ring
    (0,13),(13,14),(14,15),(15,16),
    # Pinky
    (0,17),(17,18),(18,19),(19,20),
    # Palm
    (5,9),(9,13),(13,17),
]

HAND_LINE_COL = (55, 180, 55)   # subtle green like the reference

def draw_skeleton(img, pts):
    """Thin hand skeleton overlay (visible in the reference video)."""
    for (a, b) in HAND_CONNS:
        p1 = (int(pts[a][0]), int(pts[a][1]))
        p2 = (int(pts[b][0]), int(pts[b][1]))
        cv2.line(img, p1, p2, HAND_LINE_COL, 1, cv2.LINE_AA)
    # Small dots on landmarks
    for i in range(21):
        cv2.circle(img, (int(pts[i][0]), int(pts[i][1])), 2,
                   (80, 220, 80), -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _model():
    for p in [os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "hand_landmarker.task"),
              os.path.join(os.getcwd(), "hand_landmarker.task")]:
        if os.path.isfile(p):
            return p
    dl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "hand_landmarker.task")
    print("📥  Downloading hand_landmarker.task …")
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task", dl)
    print("✅  Done.")
    return dl


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cam = CamThread().start()
    if not cam.opened:
        print("❌  Cannot open webcam."); return

    first = cam.read()
    if first is None:
        print("❌  No frame."); cam.release(); return
    H, W = first.shape[:2]
    print(f"📷  {W}×{H}")

    lm_det = HandLandmarker.create_from_options(HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=_model()),
        running_mode=RunMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.45,
    ))

    # ── Pre-allocate (never allocate inside the loop) ─────────────────
    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    smoothers = [Smoother() for _ in range(MAX_HANDS)]
    blooms    = [[Bloom() for _ in range(5)] for _ in range(MAX_HANDS)]
    palm_p    = [[] for _ in range(MAX_HANDS)]
    p_shuf    = PALM_RESHUFFLE        # trigger on first frame

    print("🌺  Running — press 'q' to quit")
    t0 = time.time()
    fn = 0; fps_t = ""

    # ══════════════════════════════════════════════════════════════════
    while True:
        raw = cam.read()
        if raw is None: break

        frame = cv2.flip(raw, 1)
        t = time.time() - t0
        h, w = frame.shape[:2]

        # ── Detect ────────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = lm_det.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(t * 1000))

        hands  = res.hand_landmarks or []
        hconf  = res.handedness or []

        # ── Dim background ────────────────────────────────────────────
        dark = cv2.convertScaleAbs(frame, alpha=BG_ALPHA, beta=0)

        # ── Clear overlay ─────────────────────────────────────────────
        overlay[:] = 0

        # ── Reshuffle palm crossing pairs ─────────────────────────────
        p_shuf += 1
        if p_shuf >= PALM_RESHUFFLE:
            p_shuf = 0
            for hi in range(MAX_HANDS):
                idx = list(range(len(KNUCKLE)))
                rng.shuffle(idx)
                palm_p[hi] = [(KNUCKLE[idx[i*2]], KNUCKLE[idx[i*2+1]])
                              for i in range(min(PALM_LINES, len(idx)//2))]

        active = set()

        for hi in range(min(len(hands), MAX_HANDS)):
            # confidence gate
            if hconf and hconf[hi]:
                if hconf[hi][0].score < CONF_THRESH:
                    continue
            active.add(hi)

            pts = smoothers[hi](hands[hi], w, h)       # (21, 2)

            # Palm centre
            pcx = (pts[0][0] + pts[5][0] + pts[17][0]) / 3
            pcy = (pts[0][1] + pts[5][1] + pts[17][1]) / 3

            # ── Hand skeleton ─────────────────────────────────────────
            draw_skeleton(overlay, pts)

            # ── Palm crossing lines ───────────────────────────────────
            if palm_p[hi]:
                draw_palm_lines(overlay, pts, palm_p[hi], t)

            # ── Per-finger: fan of stems + flowers ────────────────────
            for fi, tid in enumerate(FINGERTIP):
                bl = blooms[hi][fi]
                bl.tick()
                fx, fy = pts[tid]

                # direction away from palm → base angle of fan
                dx, dy = fx - pcx, fy - pcy
                base_a = math.atan2(dy, dx)
                gr = bl.g                           # 0→1 eased growth

                for si in range(bl.ns):
                    a   = base_a + bl.ang[si]
                    sln = bl.mlen[si] * gr
                    if sln < 4: continue
                    ex = fx + math.cos(a) * sln
                    ey = fy + math.sin(a) * sln

                    tip = draw_stem(overlay, fx, fy, ex, ey,
                                    bl.curv[si], COLORS[fi]["stem"])

                    # flower only after 40 % growth
                    if gr > 0.40:
                        fsc = min(1.0, (gr - 0.40) / 0.60)
                        draw_flower(overlay, tip[0], tip[1], t, fi, fsc)

        # Reset inactive hands
        for hi in range(MAX_HANDS):
            if hi not in active:
                smoothers[hi].reset()
                for fi in range(5): blooms[hi][fi].reset()

        # ── Composite ─────────────────────────────────────────────────
        result = cv2.addWeighted(dark, 1.0, overlay, 1.0, 0)

        # ── FPS counter ───────────────────────────────────────────────
        fn += 1
        if fn % 25 == 0:
            el = time.time() - t0
            fps_t = f"FPS: {fn/el:.0f}" if el > 0 else ""
        if fps_t:
            cv2.putText(result, fps_t, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (80, 230, 80), 2, cv2.LINE_AA)

        cv2.imshow("Flower Garden", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    lm_det.close()
    print("🌺  Done.")


if __name__ == "__main__":
    main()
