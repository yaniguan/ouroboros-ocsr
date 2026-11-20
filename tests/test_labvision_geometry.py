import numpy as np

from ouroboros.labvision.geometry import apply_h, bilinear, fit_homography, otsu


def test_dlt_recovers_homography():
    H = np.array([[1.3, -0.4, 120.0], [0.35, 1.1, 80.0], [2e-4, -3e-4, 1.0]])
    src = np.array([[0, 0], [127.76, 0], [127.76, 85.48], [0, 85.48], [40, 30]])
    est = fit_homography(src, apply_h(H, src))
    assert np.max(np.abs(est - H)) < 1e-8


def test_bilinear_and_otsu():
    img = np.zeros((10, 10))
    img[:, 5:] = 1.0
    assert bilinear(img, np.array([4.5]), np.array([3.0]))[0] == 0.5
    assert 0.0 < otsu(img) < 1.0
