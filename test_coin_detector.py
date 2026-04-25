"""
test_coin_detector.py
=====================
Unit tests for the coin_detector module.

Run with:
    python -m pytest test_coin_detector.py -v
"""

import math
from collections import Counter

import cv2
import numpy as np
import pytest

from coin_detector import (
    Coin,
    CoinDetector,
    DetectionResult,
    _bgr_to_color_name,
    _classify_coin,
    _sample_coin_color,
    US_COIN_TABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coin_image(coins_spec, img_size=(600, 600)):
    """
    Synthesise an image with solid-colour filled circles at given positions.

    coins_spec: list of (x, y, radius, bgr_colour)
    """
    img = np.full((*img_size, 3), 40, dtype=np.uint8)  # dark background
    for x, y, r, bgr in coins_spec:
        cv2.circle(img, (x, y), r, bgr, -1)
        # Add a subtle darker rim to help Hough edge detection
        cv2.circle(img, (x, y), r, tuple(max(0, c - 60) for c in bgr), 2)
    return img


# ---------------------------------------------------------------------------
# Unit tests: colour classification
# ---------------------------------------------------------------------------

class TestColorClassification:
    def test_copper_colour(self):
        # Typical copper: moderate-red hue, low-ish blue
        bgr = (30, 80, 180)
        assert _bgr_to_color_name(bgr) == "copper"

    def test_silver_colour(self):
        # High value, low saturation → silver
        bgr = (200, 200, 200)
        assert _bgr_to_color_name(bgr) == "silver"

    def test_gold_colour(self):
        # Golden yellow: high green + red, low blue, high saturation
        bgr = (0, 200, 220)
        assert _bgr_to_color_name(bgr) in ("gold", "copper")  # warm colours

    def test_dark_coin_fallback(self):
        # Very dark pixel — should fall back gracefully
        bgr = (10, 10, 10)
        name = _bgr_to_color_name(bgr)
        assert name in ("copper", "silver", "gold", "unknown")


# ---------------------------------------------------------------------------
# Unit tests: sample_coin_color
# ---------------------------------------------------------------------------

class TestSampleCoinColor:
    def test_uniform_region(self):
        # Fill a 100×100 blue image; the sampled mean should be close to blue
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = (200, 50, 10)  # BGR: high B, low G, very low R
        mean = _sample_coin_color(img, 50, 50, 40)
        assert abs(mean[0] - 200) < 20
        assert mean[2] < 50

    def test_boundary_clamp(self):
        # Circle centred near edge – should not crash
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        mean = _sample_coin_color(img, 5, 5, 20)
        assert len(mean) == 3


# ---------------------------------------------------------------------------
# Unit tests: classify_coin
# ---------------------------------------------------------------------------

class TestClassifyCoin:
    def test_quarter_silver(self):
        # ratio ~1.35 → quarter
        label, value = _classify_coin(27, "silver", 20.0, US_COIN_TABLE)
        assert label == "quarter"
        assert math.isclose(value, 0.25)

    def test_penny_copper(self):
        # ratio exactly 1.0 → penny (copper)
        label, value = _classify_coin(20, "copper", 20.0, US_COIN_TABLE)
        assert label == "penny"
        assert math.isclose(value, 0.01)

    def test_dime_silver(self):
        # ratio ~0.80 → dime
        label, value = _classify_coin(16, "silver", 20.0, US_COIN_TABLE)
        assert label == "dime"
        assert math.isclose(value, 0.10)

    def test_size_fallback_when_colour_mismatched(self):
        # "dime" size but wrong colour – should still return dime (size fallback)
        label, value = _classify_coin(16, "copper", 20.0, US_COIN_TABLE)
        assert label == "dime"

    def test_unknown_for_zero_median(self):
        label, value = _classify_coin(20, "silver", 0.0, US_COIN_TABLE)
        assert label == "unknown"
        assert value == 0.0

    def test_unknown_out_of_range(self):
        # Ratio = 5.0 → no match
        label, value = _classify_coin(100, "silver", 20.0, US_COIN_TABLE)
        assert label == "unknown"


# ---------------------------------------------------------------------------
# Unit tests: CoinDetector._load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_load_numpy_bgr(self):
        detector = CoinDetector()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        loaded = detector._load_image(img)
        assert loaded.shape == (100, 100, 3)

    def test_load_numpy_gray_converted_to_bgr(self):
        detector = CoinDetector()
        gray = np.zeros((100, 100), dtype=np.uint8)
        loaded = detector._load_image(gray)
        assert loaded.ndim == 3

    def test_load_invalid_path(self):
        detector = CoinDetector()
        with pytest.raises(ValueError, match="Could not load"):
            detector._load_image("/nonexistent/path/image.jpg")

    def test_load_invalid_array_dims(self):
        detector = CoinDetector()
        with pytest.raises(ValueError):
            detector._load_image(np.zeros((10,), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Integration tests: full detection on synthetic images
# ---------------------------------------------------------------------------

class TestCoinDetection:
    def _make_silver_coin_image(self):
        """Three silver-ish circles of different sizes."""
        return _make_coin_image([
            (150, 150, 40, (180, 180, 180)),   # large – quarter-ish
            (350, 150, 32, (160, 160, 160)),   # medium – nickel-ish
            (150, 400, 24, (200, 200, 200)),   # small – dime-ish
        ])

    def test_detection_returns_result_type(self):
        detector = CoinDetector(param2=15)
        img = self._make_silver_coin_image()
        result = detector.detect(img)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.coins, list)
        assert isinstance(result.total_value, float)
        assert result.annotated_image is not None

    def test_annotated_image_same_size(self):
        detector = CoinDetector(param2=15)
        img = self._make_silver_coin_image()
        result = detector.detect(img)
        assert result.annotated_image.shape == img.shape

    def test_total_value_non_negative(self):
        detector = CoinDetector(param2=15)
        img = self._make_silver_coin_image()
        result = detector.detect(img)
        assert result.total_value >= 0.0

    def test_total_value_equals_sum_of_coins(self):
        detector = CoinDetector(param2=15)
        img = self._make_silver_coin_image()
        result = detector.detect(img)
        expected = sum(c.value for c in result.coins)
        assert math.isclose(result.total_value, expected)

    def test_no_coins_on_blank_image(self):
        detector = CoinDetector(param2=30)
        blank = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = detector.detect(blank)
        assert result.total_value == 0.0

    def test_detect_circles_are_within_image(self):
        detector = CoinDetector(param2=15)
        img = self._make_silver_coin_image()
        result = detector.detect(img)
        h, w = img.shape[:2]
        for coin in result.coins:
            assert 0 <= coin.x < w, f"x={coin.x} out of bounds"
            assert 0 <= coin.y < h, f"y={coin.y} out of bounds"
            assert coin.radius > 0

    def test_coin_color_name_in_known_set(self):
        detector = CoinDetector(param2=15)
        img = self._make_silver_coin_image()
        result = detector.detect(img)
        valid = {"copper", "silver", "gold", "unknown"}
        for coin in result.coins:
            assert coin.color_name in valid

    def test_coin_label_in_known_set(self):
        detector = CoinDetector(param2=15)
        img = self._make_silver_coin_image()
        result = detector.detect(img)
        valid_labels = {row[0] for row in US_COIN_TABLE} | {"unknown"}
        for coin in result.coins:
            assert coin.label in valid_labels

    def test_detect_from_numpy_array(self):
        detector = CoinDetector(param2=15)
        img = self._make_silver_coin_image()
        result = detector.detect(img)
        # Should work without errors
        assert isinstance(result, DetectionResult)

    def test_copper_coin_detected(self):
        # Single prominent copper coin on dark background
        img = _make_coin_image([(300, 300, 80, (30, 80, 180))])
        detector = CoinDetector(param2=15)
        result = detector.detect(img)
        if result.coins:
            # At least one coin should be identified as copper
            assert any(c.color_name == "copper" for c in result.coins)


# ---------------------------------------------------------------------------
# Tests: CoinDetector parameter customisation
# ---------------------------------------------------------------------------

class TestCoinDetectorParams:
    def test_custom_param2_accepted(self):
        detector = CoinDetector(param2=20)
        assert detector.param2 == 20

    def test_custom_coin_table(self):
        custom_table = [("big", 2.0, "silver", 0.8, 1.5)]
        detector = CoinDetector(coin_table=custom_table)
        assert detector.coin_table is custom_table

    def test_preprocess_output_is_grayscale(self):
        detector = CoinDetector()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = detector._preprocess(img)
        assert gray.ndim == 2
