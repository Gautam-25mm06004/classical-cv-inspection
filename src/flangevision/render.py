"""Synthetic images of a machined flange: outer disk, central bore, 4 bolt holes on a pitch circle.
Rendered with 4x supersampling (anti-aliased edges, sub-pixel geometry), then degraded by an illumination gradient,
defocus blur and sensor noise. Ground truth is returned with every image."""
import numpy as np
import cv2

SIZE = 512
MM_PER_PX = 0.1                      # 51.2 mm field of view
NOMINAL = {"outer_d": 40.0, "bore_d": 12.0, "hole_d": 4.0, "pcd": 28.0}    # mm
CLASSES = ["good", "missing_hole", "chipped_rim", "eccentric_bore"]


def render(rng, defect="good", noise=0.02, blur=1.0, gradient=0.3, ss=4):
    S = SIZE * ss
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / ss
    cx, cy = SIZE / 2 + rng.uniform(-30, 30), SIZE / 2 + rng.uniform(-30, 30)
    theta = rng.uniform(0, 2 * np.pi)
    jitter = lambda: rng.normal(0, 0.05)                              # +-0.05 mm manufacturing variation
    R = (NOMINAL["outer_d"] + jitter()) / 2 / MM_PER_PX
    rb = (NOMINAL["bore_d"] + jitter()) / 2 / MM_PER_PX
    rh = (NOMINAL["hole_d"] + jitter()) / 2 / MM_PER_PX
    pcr = (NOMINAL["pcd"] + jitter()) / 2 / MM_PER_PX
    bx, by = cx, cy
    if defect == "eccentric_bore":
        a = rng.uniform(0, 2 * np.pi); off = rng.uniform(0.6, 1.5) / MM_PER_PX      # 0.6-1.5 mm off-centre
        bx, by = cx + off * np.cos(a), cy + off * np.sin(a)
    part = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
    part &= (xx - bx) ** 2 + (yy - by) ** 2 > rb ** 2
    holes = []
    missing = rng.integers(4) if defect == "missing_hole" else -1
    for k in range(4):
        a = theta + k * np.pi / 2
        hx, hy = cx + pcr * np.cos(a), cy + pcr * np.sin(a)
        if k != missing:
            part &= (xx - hx) ** 2 + (yy - hy) ** 2 > rh ** 2
            holes.append((hx, hy))
    if defect == "chipped_rim":
        # a circular bite of radius rc whose deepest point is `depth` inside the rim
        a = rng.uniform(0, 2 * np.pi); depth = rng.uniform(0.5, 1.5) / MM_PER_PX; rc = rng.uniform(1.5, 3.0) / MM_PER_PX
        d = R + rc - depth
        part &= (xx - cx - d * np.cos(a)) ** 2 + (yy - cy - d * np.sin(a)) ** 2 > rc ** 2
    img = part.reshape(SIZE, ss, SIZE, ss).mean(axis=(1, 3)).astype(np.float32)
    # back-lit style: dark part on bright background, uneven lighting, defocus, noise
    y, x = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    light = 0.85 + gradient * (x - 0.5) * np.cos(theta) + gradient * (y - 0.5) * np.sin(theta)
    out = light * (1 - 0.8 * img)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    out = out + rng.normal(0, noise, out.shape)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    truth = {"center": (cx, cy), "outer_d_mm": 2 * R * MM_PER_PX, "bore_d_mm": 2 * rb * MM_PER_PX, "hole_d_mm": 2 * rh * MM_PER_PX,
             "pcd_mm": 2 * pcr * MM_PER_PX, "bore_offset_mm": float(np.hypot(bx - cx, by - cy) * MM_PER_PX), "n_holes": len(holes),
             "defect": defect}
    return out, truth
