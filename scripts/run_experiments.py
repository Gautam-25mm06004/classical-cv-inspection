"""Metrology accuracy vs image quality, repeatability, sub-pixel ablation, and defect classification:
classical rules vs a pixel-based random forest. Writes results/results.json, results/summary.md, figures."""
import json, sys, time
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flangevision import render, inspect, CLASSES  # noqa

OUT, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = {}
KEYS = [("outer_d_mm", "outer_d_mm"), ("bore_d_mm", "bore_d_mm"), ("hole_d_mm", "hole_d_mm"), ("pcd_mm", "pcd_mm"), ("bore_offset_mm", "bore_offset_mm")]

# 1) Metrology error vs noise and blur, sub-pixel vs pixel contours (good parts)
met = []
t_inspect = []
for noise, blur in ((0.01, 0.5), (0.03, 1.0), (0.06, 1.5), (0.10, 2.0), (0.10, 3.0)):
    rng = np.random.default_rng(int(noise * 1000 + blur * 10))
    errs = {True: {k: [] for k, _ in KEYS}, False: {k: [] for k, _ in KEYS}}
    for _ in range(40):
        img, t = render(rng, "good", noise=noise, blur=blur)
        for sub in (True, False):
            t0 = time.perf_counter(); m = inspect(img, subpixel=sub); t_inspect.append(time.perf_counter() - t0)
            for k, tk in KEYS:
                errs[sub][k].append(1000 * (m[k] - t[tk]))      # micrometres
    for sub in (True, False):
        row = {"noise_sd": noise, "blur_px": blur, "contours": "sub-pixel" if sub else "pixel"}
        for k, _ in KEYS:
            e = np.array(errs[sub][k]); row[k.replace("_mm", "") + "_MAE_um"] = round(float(np.mean(np.abs(e))), 1)
        met.append(row)
R["metrology"] = met
R["inspect_ms_mean"] = round(1000 * float(np.mean(t_inspect)), 1)

# 2) Repeatability: one part (fixed geometry and pose), 30 images with fresh noise
rep = {k: [] for k, _ in KEYS}
for i in range(30):
    rng = np.random.default_rng(123)                  # same geometry and pose
    img, t = render(rng, "good", noise=0.0, blur=1.0)
    img = np.clip(img.astype(float) + np.random.default_rng(1000 + i).normal(0, 0.03 * 255, img.shape), 0, 255).astype(np.uint8)
    m = inspect(img)
    for k, _ in KEYS:
        rep[k].append(1000 * m[k])
R["repeatability_sd_um"] = {k.replace("_mm", ""): round(float(np.std(v, ddof=1)), 2) for k, v in rep.items()}

# 3) Defect classification
def make(n_per, noise_levels, seed):
    rng = np.random.default_rng(seed)
    X, y, M = [], [], []
    for ci, c in enumerate(CLASSES):
        for _ in range(n_per):
            nz = rng.choice(noise_levels)
            img, _ = render(rng, c, noise=nz, blur=rng.uniform(0.5, 2.0))
            X.append(cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).ravel() / 255.0); y.append(ci); M.append(img)
    return np.array(X), np.array(y), M
Xtr, ytr, _ = make(150, [0.01, 0.03, 0.06], 1)
rf = RandomForestClassifier(500, random_state=0, n_jobs=-1).fit(Xtr, ytr)
cls = []
cm_rules = None
for nz in (0.01, 0.06, 0.10):
    Xte, yte, imgs = make(80, [nz], 100 + int(nz * 100))
    rules = np.array([CLASSES.index(inspect(im)["verdict"]) if inspect(im)["verdict"] in CLASSES else -1 for im in imgs])
    cls.append({"noise_sd": nz, "classical_rules_acc_%": round(100 * accuracy_score(yte, rules), 1),
                "pixel_random_forest_acc_%": round(100 * accuracy_score(yte, rf.predict(Xte)), 1)})
    if nz == 0.06:
        cm_rules = confusion_matrix(yte, rules, labels=range(4)).tolist(); cm_rf = confusion_matrix(yte, rf.predict(Xte), labels=range(4)).tolist()
R["classification"] = cls
R["confusion_noise_0.06"] = {"rules": cm_rules, "random_forest": cm_rf, "labels": CLASSES}

# Figures
rng = np.random.default_rng(5)
fig, ax = plt.subplots(1, 4, figsize=(12, 3.3))
for a, c in zip(ax, CLASSES):
    img, _ = render(rng, c, noise=0.03, blur=1.0); a.imshow(img, cmap="gray"); a.set_title(c); a.axis("off")
fig.tight_layout(); fig.savefig(FIG / "examples.png", dpi=110); plt.close(fig)
img, t = render(np.random.default_rng(9), "good", noise=0.03, blur=1.0)
from flangevision.measure import flat_field
ff = flat_field(img)
_, mask = cv2.threshold(ff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.7))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("raw (uneven light)"); ax[1].imshow(ff, cmap="gray"); ax[1].set_title("flat-field corrected")
ax[2].imshow(mask, cmap="gray"); ax[2].set_title("Otsu mask")
for a in ax: a.axis("off")
fig.tight_layout(); fig.savefig(FIG / "pipeline.png", dpi=110); plt.close(fig)

(OUT / "results.json").write_text(json.dumps(R, indent=2))
def tbl(rows):
    cols = list(rows[0]); return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)
md = "# Results (0.1 mm per pixel)\n\n## Measurement error, mean absolute (micrometres), 40 good parts per row\n\n" + tbl(met)
md += f"\nMean inspection time: {R['inspect_ms_mean']} ms per image\n\n## Repeatability (same part, 30 images, noise 0.03): standard deviation in micrometres\n\n{R['repeatability_sd_um']}\n"
md += "\n## Defect classification (80 test images per class per row)\n\n" + tbl(cls)
md += f"\nConfusion at noise 0.06 (rows = true {CLASSES}):\n\nrules: {cm_rules}\n\nrandom forest: {cm_rf}\n"
(OUT / "summary.md").write_text(md)
print(md)
