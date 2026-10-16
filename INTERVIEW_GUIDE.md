# Interview guide — Machine-vision flange inspection

## 30-second version
"I built a classical OpenCV inspection pipeline for a machined flange: flat-field correction, Otsu thresholding, contours, then
sub-pixel edge localisation and robust circle fits. At 100 µm per pixel it measures diameters to about 2 µm — 1/60 of a pixel —
and detects missing holes, chipped rims and eccentric bores with 100% accuracy on rendered test parts with exact ground truth.
Without sub-pixel edges the error is about one pixel, and a pixel-based ML classifier was at chance."

## Must-know
- **Otsu**: picks the threshold that maximises between-class variance of the histogram.
- **Flat-field correction**: removes uneven lighting so one global threshold works.
- **Contour hierarchy**: outer contour = part; its children = holes (RETR_CCOMP).
- **Sub-pixel edge**: the true edge is where intensity crosses the mid level; interpolate along the gradient.
- **Circle fit**: algebraic fit is linear and fast; geometric fit minimises true radial distances (more accurate, robust loss).
- **Ground truth by rendering**: supersampling gives realistic partial-pixel edges and exact dimensions to compare against.

## Likely questions
1. *Why did the random forest fail?* Defects are small and appear anywhere due to rotation/translation; 600 images is too few to learn
   invariance. A CNN with augmentation could work, but geometry gives an explainable answer with tolerances.
2. *How would you deploy on real parts?* Calibrate the camera (intrinsics, distortion, mm/px with a reference target), telecentric
   lens for metrology, back-lighting, gauge R&R to set tolerances.
3. *Why is bore offset noisier?* It is the difference between two fitted centres, so both errors add.
4. *What limits accuracy at high blur?* The edge transition widens and noise moves the crossing point more.
