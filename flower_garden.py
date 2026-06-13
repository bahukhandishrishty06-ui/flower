#!/usr/bin/env python3
"""
🌸 Generative Flower Garden — Real-time Hand-Tracked Flower Effects
════════════════════════════════════════════════════════════════════
A real-time webcam application using OpenCV and MediaPipe that detects
hand landmarks and renders stunning generative flower effects growing
from each fingertip. Each finger sprouts a unique flower species with
layered petals, luminous particles, and ethereal glow effects.

Requirements:
    pip install opencv-python mediapipe numpy

Controls:
    q — Quit the application

Author: Generative Art System
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import random as rng
from collections import deque

# MediaPipe Tasks API (0.10.x+)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CAM_WIDTH, CAM_HEIGHT = 1280, 720     # Requested camera resolution
MAX_HANDS = 2                          # Maximum hands to track
TRAIL_LENGTH = 15                      # Frames to remember for motion trail
STEM_MAX_LEN = 60                      # Maximum stem length in pixels
PETAL_LAYERS = 4                       # Concentric petal rings per flower
PETALS_PER_LAYER_BASE = 6             # Petals in the innermost ring
PARTICLE_SPAWN_RATE = 2                # Particles spawned per flower per frame
MAX_PARTICLES = 400                    # Global particle cap
PARTICLE_LIFE_BASE = 45               # Base particle lifetime in frames
SMOOTH_FACTOR = 0.35                   # EMA smoothing (lower = smoother)
BG_DARKEN = 0.50                       # Background darkening multiplier
BLOOM_SIGMA = 7                        # Gaussian sigma for glow bloom pass
GROWTH_SPEED = 0.07                    # How fast flowers bloom open (per frame)
FADE_SPEED = 0.04                      # How fast flowers fade when hand leaves

# MediaPipe fingertip landmark indices (thumb → pinky)
FINGERTIP_IDS = [4, 8, 12, 16, 20]

# Palm landmarks used for connection lines
PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]
PALM_CONNECTIONS = [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17)]


# ═══════════════════════════════════════════════════════════════════════════
#  FLOWER COLOR PALETTES — One unique species per finger (BGR format)
# ═══════════════════════════════════════════════════════════════════════════

FINGER_SPECIES = [
    {   # Thumb — Crimson Rose
        "name": "Crimson Rose",
        "petals": [
            (50, 30, 210),   # Deep crimson (inner)
            (40, 50, 240),   # Bright red
            (60, 80, 200),   # Rose
            (80, 100, 180),  # Soft pink (outer)
        ],
        "stem":     (50, 130, 55),
        "glow":     (40, 25, 190),
        "particle": (70, 90, 255),
        "center":   (50, 190, 255),
    },
    {   # Index — Golden Dahlia
        "name": "Golden Dahlia",
        "petals": [
            (15, 170, 250),  # Rich gold (inner)
            (25, 195, 240),  # Amber
            (10, 155, 220),  # Honey
            (35, 210, 255),  # Bright yellow (outer)
        ],
        "stem":     (45, 155, 50),
        "glow":     (15, 150, 230),
        "particle": (30, 210, 255),
        "center":   (70, 235, 255),
    },
    {   # Middle — Violet Orchid
        "name": "Violet Orchid",
        "petals": [
            (210, 45, 175),  # Deep violet (inner)
            (240, 65, 195),  # Purple
            (195, 35, 155),  # Mauve
            (230, 75, 215),  # Lavender (outer)
        ],
        "stem":     (55, 125, 50),
        "glow":     (190, 35, 170),
        "particle": (245, 110, 225),
        "center":   (250, 145, 255),
    },
    {   # Ring — Teal Lotus
        "name": "Teal Lotus",
        "petals": [
            (175, 175, 25),  # Deep teal (inner)
            (195, 195, 40),  # Teal
            (155, 165, 15),  # Sea green
            (215, 210, 55),  # Cyan (outer)
        ],
        "stem":     (95, 145, 40),
        "glow":     (165, 165, 25),
        "particle": (210, 235, 75),
        "center":   (195, 250, 115),
    },
    {   # Pinky — Magenta Blossom
        "name": "Magenta Blossom",
        "petals": [
            (195, 35, 215),  # Deep magenta (inner)
            (175, 55, 245),  # Hot pink
            (215, 25, 195),  # Fuchsia
            (155, 45, 235),  # Neon pink (outer)
        ],
        "stem":     (75, 115, 55),
        "glow":     (175, 25, 205),
        "particle": (195, 95, 250),
        "center":   (215, 155, 255),
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def ease_out_cubic(t):
    """Cubic ease-out: fast start, smooth deceleration."""
    return 1.0 - (1.0 - t) ** 3


def clamp_color(c):
    """Clamp a BGR color tuple to valid [0, 255] range."""
    return tuple(max(0, min(255, int(v))) for v in c)


def scale_color(color, factor):
    """Multiply a BGR color by a scalar factor, clamped to [0,255]."""
    return clamp_color(tuple(c * factor for c in color))


def bezier_quadratic(p0, p1, p2, n_points=12):
    """
    Generate points along a quadratic Bézier curve.
    Used to draw organic, curved flower stems.
    """
    points = []
    for i in range(n_points + 1):
        t = i / n_points
        inv = 1 - t
        x = inv * inv * p0[0] + 2 * inv * t * p1[0] + t * t * p2[0]
        y = inv * inv * p0[1] + 2 * inv * t * p1[1] + t * t * p2[1]
        points.append([int(x), int(y)])
    return np.array(points, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════
#  PARTICLE SYSTEM — Luminous pollen / ember trails
# ═══════════════════════════════════════════════════════════════════════════

class Particle:
    """A single luminous particle that drifts upward like floating pollen."""
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'color', 'size']

    def __init__(self, x, y, color):
        self.x = x + rng.uniform(-10, 10)
        self.y = y + rng.uniform(-10, 10)
        self.vx = rng.uniform(-0.7, 0.7)       # Slight horizontal drift
        self.vy = rng.uniform(-2.2, -0.6)       # Upward drift
        self.max_life = PARTICLE_LIFE_BASE + rng.randint(-12, 12)
        self.life = self.max_life
        self.color = color
        self.size = rng.uniform(1.5, 3.8)

    def update(self, t):
        """Advance particle physics with gentle sine-wave sway."""
        sway = math.sin(t * 3.5 + self.x * 0.015) * 0.35
        self.x += self.vx + sway
        self.y += self.vy
        self.vy *= 0.985               # Slow deceleration
        self.life -= 1
        self.size *= 0.988

    @property
    def alive(self):
        return self.life > 0 and self.size > 0.4

    @property
    def alpha(self):
        """Opacity: fades out over lifetime."""
        return max(0.0, self.life / self.max_life)


class ParticleSystem:
    """Manages the global pool of luminous particles."""

    def __init__(self):
        self.particles: list[Particle] = []

    def spawn(self, x, y, color, count=1):
        """Emit new particles at a position."""
        for _ in range(count):
            if len(self.particles) < MAX_PARTICLES:
                self.particles.append(Particle(x, y, color))

    def update(self, t):
        """Advance all particles, cull dead ones."""
        for p in self.particles:
            p.update(t)
        self.particles = [p for p in self.particles if p.alive]

    def draw(self, overlay):
        """Render all particles onto the effect overlay."""
        for p in self.particles:
            a = p.alpha
            r = max(1, int(p.size))
            ix, iy = int(p.x), int(p.y)
            # Core dot
            cv2.circle(overlay, (ix, iy), r, scale_color(p.color, a),
                        -1, cv2.LINE_AA)
            # Soft outer halo
            if r >= 1:
                cv2.circle(overlay, (ix, iy), r * 3,
                           scale_color(p.color, a * 0.25), -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
#  FINGERTIP TRAIL — Smooth tracking with motion memory
# ═══════════════════════════════════════════════════════════════════════════

class FingertipTrail:
    """
    Tracks a single fingertip with EMA smoothing and maintains
    a motion trail of the last TRAIL_LENGTH positions.
    """

    def __init__(self):
        self.positions: deque[tuple[int, int]] = deque(maxlen=TRAIL_LENGTH)
        self.smoothed = None          # Current EMA-smoothed position
        self.active = False           # Whether the finger is currently detected
        self.growth = 0.0             # 0→1 bloom animation progress
        self.palm_dir = (0.0, -1.0)   # Direction away from palm (for stems)

    def update(self, x, y, palm_cx, palm_cy):
        """Feed a new raw landmark position; smooths & stores in trail."""
        if self.smoothed is None:
            self.smoothed = np.array([x, y], dtype=np.float64)
        else:
            self.smoothed += (np.array([x, y]) - self.smoothed) * SMOOTH_FACTOR

        sx, sy = int(self.smoothed[0]), int(self.smoothed[1])
        self.positions.appendleft((sx, sy))
        self.active = True
        self.growth = min(1.0, self.growth + GROWTH_SPEED)

        # Compute direction away from palm center (for stem growth direction)
        dx, dy = sx - palm_cx, sy - palm_cy
        dist = max(1.0, math.hypot(dx, dy))
        self.palm_dir = (dx / dist, dy / dist)

    def fade(self):
        """Gracefully fade when the finger is no longer detected."""
        self.active = False
        self.growth = max(0.0, self.growth - FADE_SPEED)
        if self.growth <= 0.01:
            self.positions.clear()
            self.smoothed = None


# ═══════════════════════════════════════════════════════════════════════════
#  RENDERING — Stems, petals, halos, connections
# ═══════════════════════════════════════════════════════════════════════════

def draw_stem(overlay, base, tip, color, thickness=2, alpha=1.0):
    """
    Draw an organic curved stem from base to tip using a Bézier curve.
    A slight sine-based offset at the midpoint creates natural curvature.
    """
    col = scale_color(color, alpha)
    # Midpoint with organic curvature offset
    mx = (base[0] + tip[0]) // 2 + int(math.sin(base[1] * 0.025) * 10)
    my = (base[1] + tip[1]) // 2 + int(math.cos(base[0] * 0.025) * 6)
    curve_pts = bezier_quadratic(base, (mx, my), tip, n_points=14)
    cv2.polylines(overlay, [curve_pts], False, col,
                  max(1, thickness), cv2.LINE_AA)


def draw_petal(overlay, cx, cy, angle, length, width, color):
    """
    Draw a single elongated elliptical petal radiating from (cx, cy)
    at the given angle. Uses cv2.ellipse for smooth anti-aliased rendering.
    """
    half_len = int(length / 2)
    half_wid = int(width / 2)
    if half_len < 1 or half_wid < 1:
        return

    # Place the ellipse center halfway along the petal direction
    ecx = int(cx + math.cos(angle) * half_len)
    ecy = int(cy + math.sin(angle) * half_len)
    angle_deg = math.degrees(angle)

    cv2.ellipse(overlay, (ecx, ecy), (half_len, half_wid),
                angle_deg, 0, 360, color, -1, cv2.LINE_AA)


def draw_flower(overlay, cx, cy, t, finger_id, growth, alpha=1.0):
    """
    Render a complete multi-layered flower with pulsing glow.

    Parameters
    ----------
    overlay : np.ndarray  — black overlay for additive blending
    cx, cy  : int         — flower center pixel coordinates
    t       : float       — global animation time (seconds)
    finger_id : int       — 0-4 finger index (selects color palette)
    growth  : float       — 0→1 bloom animation progress
    alpha   : float       — overall opacity (for motion trail fade)
    """
    if growth < 0.02 or alpha < 0.02:
        return

    sp = FINGER_SPECIES[finger_id]
    eg = ease_out_cubic(growth)         # Eased growth for smooth opening
    combined = alpha * eg               # Overall visual intensity

    # ── 1. God-Ray Halo (soft radial bloom behind the flower) ──────────
    halo_r = int(48 * eg)
    if halo_r > 2:
        for frac in (1.0, 0.65, 0.35):
            r = max(1, int(halo_r * frac))
            intensity = combined * 0.20 * (1.0 - frac * 0.4)
            cv2.circle(overlay, (cx, cy), r,
                       scale_color(sp["glow"], intensity), -1, cv2.LINE_AA)

    # ── 2. Layered Petals ─────────────────────────────────────────────
    # Pulsing size modulation — breathing effect
    pulse = 1.0 + 0.10 * math.sin(t * 2.8 + finger_id * 1.3)

    for layer in range(PETAL_LAYERS):
        layer_t = layer / max(1, PETAL_LAYERS - 1)   # 0 inner → 1 outer

        petal_len = (10 + layer * 9) * eg * pulse
        petal_wid = (5 + layer * 3) * eg * pulse

        # Each ring rotates at a unique speed for visual richness
        base_rot = t * (0.35 + layer * 0.1) + finger_id * (math.pi / 5)
        # Sine-wave sway — gentle breeze oscillation
        sway = math.sin(t * 1.6 + layer * 0.6 + finger_id * 0.9) * 0.18

        petal_col = scale_color(
            sp["petals"][layer % len(sp["petals"])],
            combined * (1.0 - layer_t * 0.25),
        )

        n_petals = PETALS_PER_LAYER_BASE + layer    # More petals outward
        for i in range(n_petals):
            angle = base_rot + sway + (2 * math.pi * i / n_petals)
            draw_petal(overlay, cx, cy, angle,
                       petal_len, petal_wid, petal_col)

    # ── 3. Luminous Center ────────────────────────────────────────────
    cr = max(1, int(7 * eg * pulse))
    cv2.circle(overlay, (cx, cy), cr,
               scale_color(sp["center"], combined), -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), max(1, cr // 2),
               scale_color(sp["center"], min(1.0, combined * 1.3)),
               -1, cv2.LINE_AA)


def draw_connections(overlay, landmarks, w, h, t):
    """
    Draw shimmering iridescent connection lines between palm landmarks
    with a soft chromatic aberration glow effect.
    """
    for (i, j) in PALM_CONNECTIONS:
        x1, y1 = int(landmarks[i].x * w), int(landmarks[i].y * h)
        x2, y2 = int(landmarks[j].x * w), int(landmarks[j].y * h)

        # Shimmer: oscillating brightness
        shimmer = 0.45 + 0.35 * math.sin(t * 4.5 + i * 0.8)

        # ── Chromatic aberration: offset R, G, B channels ──
        off = 2  # pixel offset
        cv2.line(overlay, (x1 - off, y1), (x2 - off, y2),
                 (0, 0, int(160 * shimmer)), 1, cv2.LINE_AA)
        cv2.line(overlay, (x1, y1), (x2, y2),
                 (0, int(180 * shimmer), 0), 1, cv2.LINE_AA)
        cv2.line(overlay, (x1 + off, y1), (x2 + off, y2),
                 (int(160 * shimmer), 0, 0), 1, cv2.LINE_AA)

        # ── Main iridescent line (hue cycles over time) ──
        hue = int((t * 35 + i * 45) % 180)
        iridescent_hsv = np.uint8([[[hue, 170, int(190 * shimmer)]]])
        iridescent_bgr = cv2.cvtColor(iridescent_hsv, cv2.COLOR_HSV2BGR)[0][0]
        cv2.line(overlay, (x1, y1), (x2, y2),
                 tuple(int(c) for c in iridescent_bgr), 2, cv2.LINE_AA)


def create_vignette(width, height):
    """
    Pre-compute a smooth radial vignette mask.
    Returns a (H, W, 3) float32 array in [0, 1] range.
    """
    cx, cy = width / 2.0, height / 2.0
    max_dist = math.hypot(cx, cy)

    Y, X = np.ogrid[:height, :width]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)

    # Smooth power-curve falloff at the edges
    mask = 1.0 - np.clip(dist / max_dist, 0, 1) ** 1.6 * 0.55
    return np.stack([mask, mask, mask], axis=-1)   # Broadcast to 3 channels


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION LOOP
# ═══════════════════════════════════════════════════════════════════════════

def _find_model_path():
    """
    Locate the hand_landmarker.task model file.
    Searches: script directory, cwd, then offers download instructions.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"),
        os.path.join(os.getcwd(), "hand_landmarker.task"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Auto-download if not found
    dl_path = candidates[0]
    print("📥  Downloading hand_landmarker.task model...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, dl_path)
    print("✅  Model downloaded.")
    return dl_path


def main():
    """Entry point: initializes camera, MediaPipe, and runs the render loop."""

    # ── Webcam setup ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("❌  Error: Could not open webcam.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷  Camera opened at {actual_w}×{actual_h}")

    # ── MediaPipe HandLandmarker (tasks API, v0.10+) ──────────────────
    model_path = _find_model_path()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # ── Pre-computed assets ───────────────────────────────────────────
    vignette = create_vignette(actual_w, actual_h)

    # ── Per-hand, per-finger state ────────────────────────────────────
    trails = [[FingertipTrail() for _ in range(5)] for _ in range(MAX_HANDS)]
    particles = ParticleSystem()

    print("🌸  Flower Garden is blooming — press 'q' to quit")
    t0 = time.time()
    frame_count = 0
    fps_text = ""

    # ══════════════════════════════════════════════════════════════════
    #  RENDER LOOP
    # ══════════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)     # Mirror for natural interaction
        t = time.time() - t0           # Animation clock (seconds)
        h, w = frame.shape[:2]

        # ── Run hand detection (tasks API) ────────────────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(t * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # results.hand_landmarks is List[List[NormalizedLandmark]]
        detected_hands = results.hand_landmarks  # may be empty list

        # ── Update fingertip trails ───────────────────────────────────
        active_hands: set[int] = set()

        if detected_hands:
            for hi, hand_lm in enumerate(detected_hands):
                if hi >= MAX_HANDS:
                    break
                active_hands.add(hi)
                lm = hand_lm  # List[NormalizedLandmark] with .x, .y

                # Palm center (average of wrist, index-MCP, pinky-MCP)
                pcx = int(sum(lm[k].x for k in (0, 5, 17)) / 3 * w)
                pcy = int(sum(lm[k].y for k in (0, 5, 17)) / 3 * h)

                for fi, tip_id in enumerate(FINGERTIP_IDS):
                    fx = int(lm[tip_id].x * w)
                    fy = int(lm[tip_id].y * h)
                    trails[hi][fi].update(fx, fy, pcx, pcy)

        # Fade any hand that vanished
        for hi in range(MAX_HANDS):
            if hi not in active_hands:
                for fi in range(5):
                    trails[hi][fi].fade()

        # ── Darken the camera background ──────────────────────────────
        darkened = (frame.astype(np.float32) * BG_DARKEN).astype(np.uint8)

        # ── Effects overlay (black canvas — additive blending later) ──
        overlay = np.zeros_like(frame, dtype=np.uint8)

        # ── A) Connection lines between palm landmarks ────────────────
        if detected_hands:
            for hi, hand_lm in enumerate(detected_hands):
                if hi >= MAX_HANDS:
                    break
                draw_connections(overlay, hand_lm, w, h, t)

        # ── B) Stems + Flowers + Particle spawning ────────────────────
        for hi in range(MAX_HANDS):
            for fi in range(5):
                trail = trails[hi][fi]
                if not trail.positions or trail.growth < 0.02:
                    continue

                dx, dy = trail.palm_dir        # Unit vector away from palm

                # Render each position in the trail (newest → oldest)
                for pi, (px, py) in enumerate(trail.positions):
                    # Trail fade: older positions are dimmer
                    age_frac = pi / TRAIL_LENGTH
                    trail_alpha = (1.0 - age_frac) * trail.growth
                    if trail_alpha < 0.04:
                        continue

                    # Stem length shrinks for older trail entries
                    stem_len = STEM_MAX_LEN * ease_out_cubic(trail.growth) \
                               * (1.0 - age_frac * 0.5)
                    tip_x = int(px + dx * stem_len)
                    tip_y = int(py + dy * stem_len)

                    # Draw curved stem
                    draw_stem(overlay, (px, py), (tip_x, tip_y),
                              FINGER_SPECIES[fi]["stem"],
                              thickness=max(1, int(2.5 * trail_alpha)),
                              alpha=trail_alpha * 0.9)

                    # Draw the flower at the stem tip
                    flower_growth = trail.growth * (1.0 - age_frac * 0.7)
                    draw_flower(overlay, tip_x, tip_y, t, fi,
                                flower_growth, trail_alpha)

                # Spawn particles from the newest (main) flower only
                if trail.active and trail.growth > 0.3 and trail.positions:
                    mx, my = trail.positions[0]
                    s = STEM_MAX_LEN * ease_out_cubic(trail.growth)
                    ptip_x = int(mx + dx * s)
                    ptip_y = int(my + dy * s)
                    particles.spawn(ptip_x, ptip_y,
                                    FINGER_SPECIES[fi]["particle"],
                                    count=PARTICLE_SPAWN_RATE)

        # ── C) Update & draw particles ────────────────────────────────
        particles.update(t)
        particles.draw(overlay)

        # ── D) Create glow bloom (blurred copy of overlay) ────────────
        glow = cv2.GaussianBlur(overlay, (0, 0),
                                sigmaX=BLOOM_SIGMA, sigmaY=BLOOM_SIGMA)

        # ── E) Composite: dark BG + sharp overlay + soft bloom ────────
        result = cv2.add(darkened, overlay)
        result = cv2.add(result, glow)

        # ── F) Apply vignette ─────────────────────────────────────────
        result = (result.astype(np.float32) * vignette).astype(np.uint8)

        # ── FPS overlay (updated every 20 frames for stability) ───────
        frame_count += 1
        if frame_count % 20 == 0:
            elapsed = time.time() - t0
            fps_val = frame_count / elapsed if elapsed > 0 else 0
            fps_text = f"FPS: {fps_val:.0f}"

        if fps_text:
            cv2.putText(result, fps_text, (14, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (90, 240, 90), 2, cv2.LINE_AA)

        # ── Show frame ────────────────────────────────────────────────
        cv2.imshow("Flower Garden", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("🌸  Flower Garden closed. Goodbye!")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
