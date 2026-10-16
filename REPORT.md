# Sub-Pixel Metrology and Rule-Based Defect Detection for a Machined Flange

**Gautam Jain** — IIT Bhubaneswar

## 1. Task
Measure the outer diameter, bore diameter, bolt-hole diameter, pitch-circle diameter (PCD) and bore eccentricity of a
flange imaged at 0.1 mm/pixel, and classify each part as good, missing hole, chipped rim or eccentric bore.

## 2. Synthetic data with exact ground truth
Each image is rendered on a 4× finer grid and averaged down, so edges have physically correct partial-pixel coverage. Nominal
dimensions (Ø40 / Ø12 / 4 × Ø4 on PCD 28 mm) vary by σ = 0.05 mm; pose is random; lighting has a linear gradient of up to ±15%;
Gaussian blur σ = 0.5–3 px and noise σ = 0.01–0.10 (in units of full scale) are added.

## 3. Method
1. **Flat field**: fit a plane to the brightest 40% of pixels and divide, removing the lighting gradient before Otsu thresholding.
2. **Segmentation**: Otsu threshold, 3 × 3 opening, two-level contour hierarchy (outer contour; its children are bore and holes).
3. **Sub-pixel edges**: for each contour pixel, sample the smoothed image along the gradient direction (±2 px, bilinear) and
   interpolate the crossing of the Otsu level.
4. **Circle fitting**: algebraic (Kåsa) fit as initial guess, refined by geometric least squares with a soft-L1 loss.
5. **Rules**: fewer than four holes → missing hole; maximum radial deviation of the rim from its circle > 0.3 mm → chipped rim;
   bore centre > 0.3 mm from the outer centre → eccentric bore.

## 4. Results
See README. Sub-pixel localisation is the single most important step for diameters (error 1.6 µm vs 91 µm at low noise). Errors
grow with blur and noise, and the bore offset is the most noise-sensitive quantity because it is the difference of two fitted
centres. Rule-based classification is perfect on this data, because each defect maps to a geometric measurement with a large
margin relative to measurement noise; a pixel-based random forest fails because it would need to learn rotation and translation
invariance from only 600 examples.

## 5. Limitations
Synthetic, back-lit, single part type; real optics add lens distortion (needs camera calibration), edge diffraction, dust and
reflections. Rule thresholds would be set from a gauge R&R study on real parts.

## References
1. N. Otsu. A threshold selection method from gray-level histograms. *IEEE Trans. Systems, Man, and Cybernetics* 9(1), 62–66, 1979.
2. I. Kåsa. A circle fitting procedure and its error analysis. *IEEE Trans. Instrumentation and Measurement* IM-25(1), 8–14, 1976.
3. C. Steger. Unbiased extraction of curvilinear structures from 2D and 3D images. PhD thesis, TU Munich, 1998 (sub-pixel edge extraction).
4. C. Steger, M. Ulrich, C. Wiedemann. *Machine Vision Algorithms and Applications*, 2nd ed. Wiley-VCH, 2018.
