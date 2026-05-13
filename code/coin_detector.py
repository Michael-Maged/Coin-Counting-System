"""
coin_detector.py
================
Detect and classify coins in an image using the circular Hough transform
combined with size and color features.

Typical usage
-------------
>>> from coin_detector import CoinDetector
>>> detector = CoinDetector()
>>> results = detector.detect("coins.jpg")
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Coin:
    """Represents a single detected coin."""
    x: int                     # circle centre x (pixels)
    y: int                     # circle centre y (pixels)
    radius: int                # circle radius   (pixels)
    label: str = "unknown"     # coin denomination label
    color_name: str = "unknown"  # dominant color group ("gold", "silver", "copper")
    bgr_mean: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))


@dataclass
class DetectionResult:
    """Full result returned by :meth:`CoinDetector.detect`."""
    coins: List[Coin] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Coin classification tables
# ---------------------------------------------------------------------------

# Each entry: (label, value, color_group, min_radius_ratio, max_radius_ratio)
# radius_ratio = coin_radius / median_radius  – this makes classification
# robust to different image scales / resolutions.
#
# The table below covers both US coins and a generic set that can be adapted
# to other currencies by subclassing and overriding `COIN_TABLE`.

US_COIN_TABLE = [
    # label,    color,     min_r,  max_r
    ("dime",    "silver",   0.70,   0.92),
    ("penny",   "copper",   0.75,   1.15),
    ("nickel",  "silver",   0.93,   1.15),
    ("quarter", "silver",   1.10,   1.40),
    ("half",    "silver",   1.35,   1.80),
    ("dollar",  "gold",     1.50,   2.20),
]

# Color group boundaries in HSV space
# Each entry: (name, hue_low, hue_high, sat_low, val_low)
COLOR_GROUPS = [
    # copper / bronze: orange-brown hues
    ("copper",  5,  25, 50,  50),
    # gold: warm yellow hues
    ("gold",   20,  40, 60,  80),
    # silver: low saturation, high value (any hue)
    ("silver",  0, 180,  0, 120),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _bgr_to_color_name(bgr: Tuple[float, float, float]) -> str:
    """
    Map a BGR mean colour to a named colour group
    ('copper', 'gold', 'silver') using HSV thresholds.
    """
    # Convert a single pixel BGR->HSV via OpenCV
    pixel = np.array([[list(bgr)]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Check copper first (narrower hue window)
    if 5 <= h <= 25 and s >= 50 and v >= 50:
        return "copper"
    # Gold
    if 20 <= h <= 40 and s >= 60 and v >= 80:
        return "gold"
    # Silver / white-metal (low saturation)
    if s < 60 and v >= 80:
        return "silver"
    # Fallback: classify by brightness
    if v >= 120:
        return "silver"
    return "copper"


def _sample_coin_color(image_bgr: np.ndarray, x: int, y: int, radius: int,
                        sample_fraction: float = 0.6) -> Tuple[float, float, float]:
    """
    Return the mean BGR colour inside a circle, using a circular mask that
    covers *sample_fraction* of the full radius (avoids edge artefacts).
    """
    h, w = image_bgr.shape[:2]
    inner_r = max(1, int(radius * sample_fraction))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), inner_r, 255, -1)

    mean_bgr = cv2.mean(image_bgr, mask=mask)[:3]
    return (mean_bgr[0], mean_bgr[1], mean_bgr[2])


def _classify_coin(radius: int, color_name: str,
                   median_radius: float,
                   coin_table: list) -> str:
    """
    Return label by matching *radius* against the relative-size table and
    verifying color when the table specifies one.

    Strategy:
    1. Exact match: size range AND color both match.
    2. Color-only match: color matches, pick closest ratio entry.
    3. Size-only fallback: pick the entry whose range contains the ratio.
    4. Closest ratio: pick the entry with the nearest midpoint.
    """
    if median_radius <= 0:
        return "unknown"

    ratio = radius / median_radius

    # 1. Exact match (size range + color)
    for label, color_group, min_r, max_r in coin_table:
        if min_r <= ratio <= max_r and (color_group == color_name or color_group == "any"):
            return label

    # 2. Color matches — pick closest midpoint among color-matching entries
    color_candidates = [(label, (min_r + max_r) / 2)
                        for label, color_group, min_r, max_r in coin_table
                        if color_group == color_name or color_group == "any"]
    if color_candidates:
        return min(color_candidates, key=lambda t: abs(t[1] - ratio))[0]

    # 3. Size-only fallback
    for label, color_group, min_r, max_r in coin_table:
        if min_r <= ratio <= max_r:
            return label

    # 4. Closest ratio overall
    return min(coin_table, key=lambda row: abs((row[2] + row[3]) / 2 - ratio))[0]


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class CoinDetector:
    """
    Detect coins in an image using the circular Hough transform and classify
    them by relative size and colour.

    Parameters
    ----------
    coin_table : list, optional
        Classification table.  Defaults to :data:`US_COIN_TABLE`.
    dp : float
        Inverse ratio of the accumulator resolution to the image resolution.
        Higher values speed up processing at the cost of accuracy (default 1.2).
    min_dist_factor : float
        Minimum distance between circle centres as a fraction of the image
        short-side (default 0.3).
    param1 : int
        Upper Canny threshold for the internal edge detector (default 85).
    param2 : int
        Accumulator threshold for circle detection; lower finds more circles
        (default 55).
    min_radius_factor : float
        Minimum circle radius as a fraction of the image short-side (default 0.04).
    max_radius_factor : float
        Maximum circle radius as a fraction of the image short-side (default 0.4).
    blur_ksize : int
        Gaussian blur kernel size applied before Hough detection (default 7).
    median_ksize : int
        Median blur kernel size for salt-and-pepper noise removal (default 3).
    dedupe_dist_factor : float
        Center-distance fraction for duplicate suppression (default 0.8).
    dedupe_radius_factor : float
        Allowed radius difference fraction for duplicates (default 0.4).
    """

    def __init__(
        self,
        coin_table: Optional[list] = None,
        dp: float = 1.2,
        min_dist_factor: float = 0.3,
        param1: int = 85,
        param2: int = 55,
        min_radius_factor: float = 0.04,
        max_radius_factor: float = 0.4,
        blur_ksize: int = 7,
        median_ksize: int = 3,
        dedupe_dist_factor: float = 0.8,
        dedupe_radius_factor: float = 0.4,
    ):
        self.coin_table = coin_table if coin_table is not None else US_COIN_TABLE
        self.dp = dp
        self.min_dist_factor = min_dist_factor
        self.param1 = param1
        self.param2 = param2
        self.min_radius_factor = min_radius_factor
        self.max_radius_factor = max_radius_factor
        self.blur_ksize = blur_ksize
        self.median_ksize = median_ksize
        self.dedupe_dist_factor = dedupe_dist_factor
        self.dedupe_radius_factor = dedupe_radius_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _watershed_centers(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Use thresholding + distance transform + watershed to find coin
        centres. Returns (N,2) array of (x,y) or None if fewer than 1 found."""
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        n_labels, markers = cv2.connectedComponents(sure_fg)
        if n_labels < 2:
            return None
        # Return centroid of each foreground label
        centers = []
        for label in range(1, n_labels):
            ys, xs = np.where(markers == label)
            centers.append([int(xs.mean()), int(ys.mean())])
        return np.array(centers, dtype=int)

    def detect(self, image_source) -> DetectionResult:
        """
        Detect and classify all coins in *image_source*.

        Parameters
        ----------
        image_source : str | numpy.ndarray
            Either a file path to an image or a BGR numpy array.

        Returns
        -------
        DetectionResult
            A dataclass containing the list of :class:`Coin` objects and an
            annotated copy of the image.

        Raises
        ------
        ValueError
            If the image cannot be loaded or has an unexpected format.
        """
        image = self._load_image(image_source)
        gray = self._preprocess(image)
        circles = self._hough_detect(image, gray)

        if circles is None or len(circles) == 0:
            return DetectionResult(coins=[], annotated_image=image.copy())

        coins = self._build_coins(image, circles)
        annotated = self._draw_annotations(image, coins)

        return DetectionResult(coins=coins, annotated_image=annotated)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, source) -> np.ndarray:
        if isinstance(source, np.ndarray):
            if source.ndim not in (2, 3):
                raise ValueError("Image array must be 2-D (gray) or 3-D (BGR).")
            return source if source.ndim == 3 else cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        img = cv2.imread(str(source))
        if img is None:
            raise ValueError(f"Could not load image from path: {source!r}")
        return img

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and reduce noise before Hough detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        median_ksize = self.median_ksize | 1  # ensure odd
        if median_ksize > 1:
            gray = cv2.medianBlur(gray, median_ksize)
            # Second pass: suppress heavy salt-and-pepper noise
            med = cv2.medianBlur(gray, median_ksize)
            noise_mean = float(np.abs(gray.astype(np.int32) - med.astype(np.int32)).mean())
            if noise_mean > 1.5:
                gray = med
        ksize = self.blur_ksize | 1   # ensure odd
        return cv2.GaussianBlur(gray, (ksize, ksize), 2)

    def _hough_detect(self, image: np.ndarray, gray: np.ndarray):
        """Run cv2.HoughCircles, return refined (N, 3) circle array or None."""
        h, w = image.shape[:2]
        short_side = min(h, w)

        min_radius = max(1, int(short_side * self.min_radius_factor))
        max_radius = int(short_side * self.max_radius_factor)

        # Scale param2 down for small images
        param2 = max(10, int(self.param2 * (0.35 + 0.65 * min(1.0, short_side / 317))))

        # Use watershed to estimate a lower-bound on minDist, but cap it
        # so it never exceeds min_radius_factor * short_side (avoids suppressing
        # closely-packed or angled coins where watershed is unreliable)
        ws_centers = self._watershed_centers(gray)
        cap = max(1, int(short_side * self.min_dist_factor))
        if ws_centers is not None and len(ws_centers) >= 2:
            from scipy.spatial.distance import cdist
            dists = cdist(ws_centers, ws_centers)
            np.fill_diagonal(dists, np.inf)
            ws_min_dist = max(1, int(dists.min() * 0.8))
            min_dist = min(ws_min_dist, cap)
        else:
            min_dist = cap

        # Two passes: tight first pass; relaxed second pass only if first finds nothing
        all_circles = []
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=min_dist,
            param1=self.param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None:
            all_circles.append(np.round(circles[0]).astype(int))
        else:
            # Relaxed fallback: lower param2 and minDist to catch difficult images
            relaxed_dist = max(1, int(min_dist * 0.5))
            relaxed_p2   = max(10, int(param2 * 0.55))
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=self.dp,
                minDist=relaxed_dist,
                param1=self.param1,
                param2=relaxed_p2,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles is not None:
                all_circles.append(np.round(circles[0]).astype(int))

        if not all_circles:
            return None

        circles = np.vstack(all_circles)
        edges = cv2.Canny(gray, 30, 100)
        circles = self._refine_radii(edges, circles)
        circles = self._dedupe_circles(circles, edges)

        return circles

    def _circle_edge_score(self, edges: np.ndarray, x: int, y: int, r: int) -> int:
        """Return edge support along the circle circumference."""
        mask = np.zeros_like(edges)
        cv2.circle(mask, (x, y), r, 255, 2)
        return int(cv2.countNonZero(cv2.bitwise_and(edges, mask)))

    def _dedupe_circles(self, circles: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Remove near-duplicate circles based on center distance and radius."""
        if circles.size == 0:
            return circles

        scored = []
        for x, y, r in circles.tolist():
            scored.append((self._circle_edge_score(edges, x, y, r), x, y, r))

        kept = []
        for _, x, y, r in sorted(scored, key=lambda t: t[0], reverse=True):
            is_dup = False
            for kx, ky, kr in kept:
                dist = np.hypot(x - kx, y - ky)
                if dist <= self.dedupe_dist_factor * max(r, kr):
                    if abs(r - kr) <= self.dedupe_radius_factor * min(r, kr):
                        is_dup = True
                        break
            if not is_dup:
                kept.append((x, y, r))

        return np.array(kept, dtype=int)

    def _refine_radii(self, edges: np.ndarray, circles: np.ndarray) -> np.ndarray:
        """Refine each circle's radius by finding the peak edge response
        along radial samples around the detected centre."""
        refined = []
        for x, y, r in circles:
            best_r, best_score = r, -1
            for candidate_r in range(max(1, r - 8), r + 9):
                score = self._circle_edge_score(edges, int(x), int(y), int(candidate_r))
                if score > best_score:
                    best_score = score
                    best_r = candidate_r
            refined.append([x, y, best_r])
        return np.array(refined, dtype=int)

    def _build_coins(self, image: np.ndarray, circles: np.ndarray) -> List[Coin]:
        """
        Classify each detected circle and return a list of :class:`Coin`
        objects.
        """
        radii = circles[:, 2]
        median_radius = float(np.median(radii))

        coins: List[Coin] = []
        for x, y, r in circles:
            bgr_mean = _sample_coin_color(image, int(x), int(y), int(r))
            color_name = _bgr_to_color_name(bgr_mean)
            label = _classify_coin(int(r), color_name, median_radius, self.coin_table)
            coins.append(Coin(
                x=int(x), y=int(y), radius=int(r),
                label=label, color_name=color_name, bgr_mean=bgr_mean,
            ))

        return coins

    def _draw_annotations(self, image: np.ndarray, coins: List[Coin]) -> np.ndarray:
        """Return a copy of *image* with circles and labels drawn on it."""
        annotated = image.copy()

        for coin in coins:
            x, y, r = coin.x, coin.y, coin.radius

            # Colour the outline by coin colour group
            if coin.color_name == "copper":
                outline = (0, 80, 180)   # BGR – brownish-red
            elif coin.color_name == "gold":
                outline = (0, 200, 220)  # BGR – orange-yellow
            else:
                outline = (200, 200, 200)  # BGR – light grey / silver

            cv2.circle(annotated, (x, y), r, outline, 2)
            cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = r / 80
            thickness = 1

            (tw, th), _ = cv2.getTextSize(coin.label, font, font_scale, thickness)
            tx = x - tw // 2
            ty = y + th // 2
            cv2.putText(annotated, coin.label, (tx, ty), font, font_scale,
                        (255, 255, 255), thickness + 1, cv2.LINE_AA)
            cv2.putText(annotated, coin.label, (tx, ty), font, font_scale,
                        (0, 0, 0), thickness, cv2.LINE_AA)

        return annotated
