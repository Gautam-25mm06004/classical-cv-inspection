"""Classical inspection pipeline: flat-field correction -> Otsu threshold -> contour hierarchy -> circle fits ->
dimensions and rule-based defect decisions."""
import numpy as np
import cv2
from scipy.optimize import least_squares
from .render import MM_PER_PX, NOMINAL

TOL = {"outer_d": 0.2, "bore_d": 0.2, "hole_d": 0.2, "pcd": 0.2, "bore_offset": 0.3, "rim_deviation": 0.3}   # mm


def flat_field(img):
    """Fit a plane to the brightest 40% of pixels (background) and divide it out."""
    f = img.astype(np.float32)
    thr = np.percentile(f, 60)
    ys, xs = np.nonzero(f >= thr)
    A = np.c_[xs, ys, np.ones_like(xs)].astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, f[ys, xs], rcond=None)
    yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    plane = coef[0] * xx + coef[1] * yy + coef[2]
    return np.clip(f / plane * 200, 0, 255).astype(np.uint8)


def fit_circle(pts):
    """Algebraic (Kasa) fit as a start, refined by geometric least squares on radial residuals."""
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    c, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    x0, y0 = c[0], c[1]
    r0 = np.sqrt(c[2] + x0 ** 2 + y0 ** 2)
    res = least_squares(lambda p: np.hypot(x - p[0], y - p[1]) - p[2], [x0, y0, r0], loss="soft_l1", f_scale=1.0)
    return res.x


def subpixel_contour(gray, contour, thr):
    """Move each contour point along the local gradient to where the intensity crosses `thr` (linear interpolation)."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    out = []
    H, W = g.shape
    for x, y in contour.reshape(-1, 2):
        nx, ny = gx[y, x], gy[y, x]
        n = np.hypot(nx, ny)
        if n < 1e-6:
            continue
        nx, ny = nx / n, ny / n
        ts = np.linspace(-2, 2, 17)
        px, py = x + ts * nx, y + ts * ny
        if px.min() < 0 or py.min() < 0 or px.max() >= W - 1 or py.max() >= H - 1:
            continue
        vals = cv2.remap(g, px.astype(np.float32).reshape(1, -1), py.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        s = np.nonzero(np.diff(np.sign(vals - thr)))[0]
        if len(s):
            i = s[np.argmin(np.abs(ts[s]))]
            t = ts[i] + (thr - vals[i]) / (vals[i + 1] - vals[i]) * (ts[i + 1] - ts[i])
            out.append((x + t * nx, y + t * ny))
    return np.array(out) if out else contour.reshape(-1, 2).astype(float)


def inspect(img, subpixel=True):
    ff = flat_field(img)
    thr, mask = cv2.threshold(ff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"ok": False, "reason": "no part"}
    hier = hier[0]
    outer_i = max((i for i in range(len(contours)) if hier[i][3] == -1), key=lambda i: cv2.contourArea(contours[i]))
    kids = [i for i in range(len(contours)) if hier[i][3] == outer_i and cv2.contourArea(contours[i]) > 20]
    level = float(thr)
    def pts(i):
        return subpixel_contour(ff, contours[i], level) if subpixel else contours[i].reshape(-1, 2).astype(float)
    op = pts(outer_i)
    ox, oy, orad = fit_circle(op)
    rim_dev = float(np.max(np.abs(np.hypot(op[:, 0] - ox, op[:, 1] - oy) - orad)) * MM_PER_PX)
    circles = sorted((fit_circle(pts(i)) for i in kids), key=lambda c: -c[2])
    out = {"ok": True, "outer_d_mm": 2 * orad * MM_PER_PX, "rim_deviation_mm": rim_dev, "n_holes": max(0, len(circles) - 1)}
    if circles:
        bx, by, br = circles[0]
        out["bore_d_mm"] = 2 * br * MM_PER_PX
        out["bore_offset_mm"] = float(np.hypot(bx - ox, by - oy) * MM_PER_PX)
        holes = circles[1:]
        if holes:
            out["hole_d_mm"] = float(np.mean([2 * h[2] for h in holes]) * MM_PER_PX)
            out["pcd_mm"] = float(np.mean([2 * np.hypot(h[0] - ox, h[1] - oy) for h in holes]) * MM_PER_PX)
    out["verdict"] = classify(out)
    return out


def classify(m):
    if not m.get("ok") or "bore_d_mm" not in m:
        return "reject"
    if m["n_holes"] < 4:
        return "missing_hole"
    if m["rim_deviation_mm"] > TOL["rim_deviation"]:
        return "chipped_rim"
    if m["bore_offset_mm"] > TOL["bore_offset"]:
        return "eccentric_bore"
    return "good"
