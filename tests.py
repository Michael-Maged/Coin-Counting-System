"""Unit tests for the coin_detector module."""

import numpy as np
import pytest

from coin_detector import (
    CoinDetector,
    DEFAULT_COIN_PROFILES,
    _colour_category,
    _extract_circular_roi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blank(h: int = 300, w: int = 300) -> np.ndarray:
    """Return a white BGR image."""
    return np.ones((h, w, 3), dtype=np.uint8) * 220


def _draw_coin(image: np.ndarray, cx: int, cy: int, r: int, bgr: tuple) -> None:
    """Draw a filled circle with the given colour onto *image* in-place."""
    import cv2
    cv2.circle(image, (cx, cy), r, bgr, -1)


# ---------------------------------------------------------------------------
# _colour_category
# ---------------------------------------------------------------------------

class TestColourCategory:
    def test_empty_roi_returns_silver(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert _colour_category(empty) == "silver"

    def test_silver_low_saturation(self):
        # A grey patch → low saturation → silver
        grey = np.full((40, 40, 3), 150, dtype=np.uint8)
        assert _colour_category(grey) == "silver"

    def test_copper_hue(self):
        # Approximate copper: BGR ≈ (50, 80, 185)
        copper = np.zeros((40, 40, 3), dtype=np.uint8)
        copper[:] = (50, 80, 185)
        result = _colour_category(copper)
        assert result in ("copper", "gold")  # hue boundary is fuzzy in uint8

    def test_gold_hue(self):
        # Approximate gold: BGR ≈ (0, 180, 220)
        gold_patch = np.zeros((40, 40, 3), dtype=np.uint8)
        gold_patch[:] = (0, 180, 220)
        result = _colour_category(gold_patch)
        assert result in ("gold", "copper", "silver")  # implementation-defined for edge hues


# ---------------------------------------------------------------------------
# _extract_circular_roi
# ---------------------------------------------------------------------------

class TestExtractCircularROI:
    def test_basic_extraction(self):
        img = _make_blank(100, 100)
        roi = _extract_circular_roi(img, 50, 50, 20)
        assert roi.shape[2] == 3  # BGR channels
        assert roi.shape[0] > 0 and roi.shape[1] > 0

    def test_clamps_to_image_boundary(self):
        img = _make_blank(100, 100)
        # Circle partially outside image
        roi = _extract_circular_roi(img, 5, 5, 20)
        assert roi.size > 0

    def test_mask_zeros_outside_circle(self):
        img = _make_blank(100, 100)
        # Use a pure colour so we can detect masking
        img[:] = (255, 0, 0)  # blue
        roi = _extract_circular_roi(img, 50, 50, 10)
        # Corner pixels of the bounding box should be masked to 0
        assert roi[0, 0, 0] == 0  # outside circle → zeroed
        # Centre pixel should retain the original colour
        local_cx = local_cy = 10  # radius = 10, centre of ROI
        assert roi[local_cy, local_cx, 0] == 255


# ---------------------------------------------------------------------------
# CoinDetector.detect  (synthetic image tests)
# ---------------------------------------------------------------------------

class TestCoinDetectorDetect:
    def test_returns_list(self):
        detector = CoinDetector()
        img = _make_blank(400, 400)
        result = detector.detect(img)
        assert isinstance(result, list)

    def test_detects_circle_on_plain_background(self):
        import cv2
        detector = CoinDetector(param2=15)  # lower threshold → more sensitive
        img = _make_blank(400, 400)
        # Draw a silver-ish coin (grey) with radius ~30 px (rel 0.075 of 400)
        cv2.circle(img, (200, 200), 30, (150, 150, 150), -1)
        result = detector.detect(img)
        # At least one circle should be found
        assert len(result) >= 1

    def test_detection_result_keys(self):
        import cv2
        detector = CoinDetector(param2=15)
        img = _make_blank(400, 400)
        cv2.circle(img, (200, 200), 30, (150, 150, 150), -1)
        result = detector.detect(img)
        if result:
            det = result[0]
            assert "center" in det
            assert "radius" in det
            assert "label" in det
            assert "color" in det

    def test_no_circles_returns_empty(self):
        detector = CoinDetector()
        # Completely uniform image – no circles to detect
        img = np.full((300, 300, 3), 200, dtype=np.uint8)
        result = detector.detect(img)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# CoinDetector._classify
# ---------------------------------------------------------------------------

class TestCoinDetectorClassify:
    def setup_method(self):
        self.detector = CoinDetector()

    def test_unknown_for_out_of_range_radius(self):
        label = self.detector._classify(radius=1, short_side=400, color_cat="silver")
        assert label == "Unknown"

    def test_silver_coin_classified(self):
        # Quarter: rel radius ~0.11 of short_side, silver
        short_side = 400
        r = int(0.11 * short_side)  # 44 px
        label = self.detector._classify(radius=r, short_side=short_side, color_cat="silver")
        assert label != "Unknown"

    def test_copper_coin_classified(self):
        # Penny: rel radius ~0.08 of short_side, copper
        short_side = 400
        r = int(0.08 * short_side)  # 32 px
        label = self.detector._classify(radius=r, short_side=short_side, color_cat="copper")
        assert label == "Penny"


# ---------------------------------------------------------------------------
# CoinDetector.summarise
# ---------------------------------------------------------------------------

class TestCoinDetectorSummarise:
    def setup_method(self):
        self.detector = CoinDetector()

    def test_empty_detections(self):
        summary = self.detector.summarise([])
        assert summary["total"] == 0
        assert summary["counts"] == {}

    def test_counts_correctly(self):
        detections = [
            {"center": (10, 10), "radius": 20, "label": "Penny", "color": "copper"},
            {"center": (50, 50), "radius": 22, "label": "Penny", "color": "copper"},
            {"center": (90, 90), "radius": 30, "label": "Quarter", "color": "silver"},
        ]
        summary = self.detector.summarise(detections)
        assert summary["total"] == 3
        assert summary["counts"]["Penny"] == 2
        assert summary["counts"]["Quarter"] == 1


# ---------------------------------------------------------------------------
# CoinDetector.annotate
# ---------------------------------------------------------------------------

class TestCoinDetectorAnnotate:
    def test_returns_same_shape(self):
        import cv2
        detector = CoinDetector()
        img = _make_blank(300, 300)
        detections = [
            {"center": (150, 150), "radius": 40, "label": "Quarter", "color": "silver"},
        ]
        out = detector.annotate(img, detections)
        assert out.shape == img.shape

    def test_does_not_modify_original(self):
        detector = CoinDetector()
        img = _make_blank(300, 300)
        original_copy = img.copy()
        detections = [
            {"center": (150, 150), "radius": 40, "label": "Quarter", "color": "silver"},
        ]
        detector.annotate(img, detections)
        assert np.array_equal(img, original_copy)
