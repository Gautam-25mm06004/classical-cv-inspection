# Results (0.1 mm per pixel)

## Measurement error, mean absolute (micrometres), 40 good parts per row

| noise_sd | blur_px | contours | outer_d_MAE_um | bore_d_MAE_um | hole_d_MAE_um | pcd_MAE_um | bore_offset_MAE_um |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.5 | sub-pixel | 1.6 | 1.5 | 6.0 | 0.4 | 1.9 |
| 0.01 | 0.5 | pixel | 91.3 | 91.1 | 89.7 | 3.4 | 2.8 |
| 0.03 | 1.0 | sub-pixel | 2.4 | 2.1 | 9.6 | 0.9 | 7.6 |
| 0.03 | 1.0 | pixel | 92.6 | 90.7 | 86.8 | 2.9 | 5.8 |
| 0.06 | 1.5 | sub-pixel | 6.9 | 3.5 | 11.0 | 2.1 | 22.1 |
| 0.06 | 1.5 | pixel | 113.7 | 110.0 | 99.6 | 7.2 | 15.6 |
| 0.1 | 2.0 | sub-pixel | 10.6 | 7.9 | 17.1 | 4.4 | 41.3 |
| 0.1 | 2.0 | pixel | 183.4 | 176.8 | 163.1 | 9.8 | 27.7 |
| 0.1 | 3.0 | sub-pixel | 11.7 | 12.1 | 42.6 | 6.0 | 57.7 |
| 0.1 | 3.0 | pixel | 252.1 | 238.4 | 207.2 | 15.3 | 37.2 |

Mean inspection time: 55.6 ms per image

## Repeatability (same part, 30 images, noise 0.03): standard deviation in micrometres

{'outer_d': 2.34, 'bore_d': 2.54, 'hole_d': 2.48, 'pcd': 1.03, 'bore_offset': 0.77}

## Defect classification (80 test images per class per row)

| noise_sd | classical_rules_acc_% | pixel_random_forest_acc_% |
|---|---|---|
| 0.01 | 100.0 | 26.6 |
| 0.06 | 100.0 | 23.8 |
| 0.1 | 100.0 | 25.3 |

Confusion at noise 0.06 (rows = true ['good', 'missing_hole', 'chipped_rim', 'eccentric_bore']):

rules: [[80, 0, 0, 0], [0, 80, 0, 0], [0, 0, 80, 0], [0, 0, 0, 80]]

random forest: [[19, 21, 19, 21], [18, 18, 27, 17], [31, 15, 19, 15], [23, 20, 17, 20]]
