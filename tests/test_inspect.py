import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flangevision import render, inspect, fit_circle, CLASSES  # noqa


class TestInspection(unittest.TestCase):
    def test_circle_fit_exact(self):
        a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        pts = np.c_[12.3 + 50 * np.cos(a), -4.5 + 50 * np.sin(a)]
        np.testing.assert_allclose(fit_circle(pts), [12.3, -4.5, 50], atol=1e-6)

    def test_dimensions_within_tolerance(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            img, t = render(rng, "good", noise=0.03, blur=1.0)
            m = inspect(img)
            for k in ("outer_d_mm", "bore_d_mm", "pcd_mm"):
                self.assertLess(abs(m[k] - t[k]), 0.05, k)          # 50 um = half a pixel
            self.assertEqual(m["verdict"], "good")

    def test_each_defect_detected(self):
        rng = np.random.default_rng(1)
        for c in CLASSES:
            img, _ = render(rng, c, noise=0.02, blur=1.0)
            self.assertEqual(inspect(img)["verdict"], c)

    def test_uneven_lighting_handled(self):
        rng = np.random.default_rng(2)
        img, t = render(rng, "good", noise=0.02, blur=1.0, gradient=0.6)
        self.assertLess(abs(inspect(img)["outer_d_mm"] - t["outer_d_mm"]), 0.05)


if __name__ == "__main__":
    unittest.main()
