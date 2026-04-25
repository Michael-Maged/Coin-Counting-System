"""
Coin detection and classification using circular Hough transform and size/color features.

Coins are detected as circles via cv2.HoughCircles and then classified by their
relative radius and the average color of their region (to distinguish copper/gold
coins from silver coins).
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Coin denomination definitions
# Each entry maps a label to its expected normalised radius range (radius as a
# fraction of the image's shorter dimension) and colour category.
#
# Colour categories
#   "copper"  – reddish/brownish hue  (e.g. US penny, 1p / 2p)
#   "silver"  – cool grey hue         (e.g. US nickel/dime/quarter)
#   "gold"    – warm yellow hue       (e.g. £1, €1/€2)
#   "any"     – colour is not used as a discriminator
#
# The radius ranges below assume coins captured on a plain background at a
# reasonably consistent scale.  They can be overridden via CoinDetector
# constructor parameters.
# ---------------------------------------------------------------------------
DEFAULT_COIN_PROFILES: list[dict] = [
    # label,          colour,    relative-radius range (min, max)
    {"label": "Dime",    "color": "silver", "radius_range": (0.03, 0.07)},
    {"label": "Penny",   "color": "copper", "radius_range": (0.06, 0.10)},
    {"label": "Nickel",  "color": "silver", "radius_range": (0.08, 0.12)},
    {"label": "Quarter", "color": "silver", "radius_range": (0.09, 0.14)},
]

# HSV hue thresholds used to assign a colour category to a coin region.
# Hue in OpenCV is [0, 179].
_HUE_COPPER_MIN, _HUE_COPPER_MAX = 5, 20    # reddish-brown
_HUE_GOLD_MIN,   _HUE_GOLD_MAX   = 20, 40   # warm yellow / golden
# Silver coins have low saturation regardless of hue.
_SILVER_SAT_MAX = 40


def _colour_category(bgr_roi: np.ndarray) -> str:
    """Return the broad colour category ('copper', 'gold', 'silver') for a BGR patch."""
    if bgr_roi.size == 0:
        return "silver"
    hsv_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    mean_h = float(np.mean(hsv_roi[:, :, 0]))
    mean_s = float(np.mean(hsv_roi[:, :, 1]))

    if mean_s < _SILVER_SAT_MAX:
        return "silver"
    if _HUE_COPPER_MIN <= mean_h <= _HUE_COPPER_MAX:
        return "copper"
    if _HUE_GOLD_MIN <= mean_h <= _HUE_GOLD_MAX:
        return "gold"
    return "silver"


def _extract_circular_roi(image: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
    """Return the pixels inside the circle (cx, cy, radius) as a rectangular BGR patch."""
    h, w = image.shape[:2]
    x1 = max(cx - radius, 0)
    y1 = max(cy - radius, 0)
    x2 = min(cx + radius, w)
    y2 = min(cy + radius, h)

    roi = image[y1:y2, x1:x2].copy()

    # Build a circular mask so background pixels do not skew the colour average.
    local_cx = cx - x1
    local_cy = cy - y1
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (local_cx, local_cy), radius, 255, -1)
    roi[mask == 0] = 0
    return roi


class CoinDetector:
    """Detect and classify coins in an image.

    Parameters
    ----------
    coin_profiles:
        List of dicts, each with keys ``label``, ``color``, and
        ``radius_range`` (relative to the image's shorter dimension).
        Defaults to :data:`DEFAULT_COIN_PROFILES`.
    dp:
        Inverse ratio of the accumulator resolution to the image resolution
        passed to :func:`cv2.HoughCircles`.  Default ``1.2``.
    min_dist_factor:
        Minimum distance between detected circle centres as a fraction of the
        image's shorter dimension.  Default ``0.06``.
    param1:
        Upper threshold for the Canny edge detector inside HoughCircles.
        Default ``100``.
    param2:
        Accumulator threshold for circle detection – lower values detect more
        (possibly false) circles.  Default ``30``.
    blur_ksize:
        Kernel size for the Gaussian blur applied before Hough detection.
        Must be odd and ≥ 1.  Default ``7``.
    """

    def __init__(
        self,
        coin_profiles: list[dict] | None = None,
        dp: float = 1.2,
        min_dist_factor: float = 0.06,
        param1: float = 100,
        param2: float = 30,
        blur_ksize: int = 7,
    ) -> None:
        self.coin_profiles = coin_profiles if coin_profiles is not None else DEFAULT_COIN_PROFILES
        self.dp = dp
        self.min_dist_factor = min_dist_factor
        self.param1 = param1
        self.param2 = param2
        self.blur_ksize = blur_ksize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[dict]:
        """Detect and classify coins in *image*.

        Parameters
        ----------
        image:
            BGR image as a NumPy array (as returned by ``cv2.imread``).

        Returns
        -------
        list of dict
            Each dict has the keys:

            ``center``  – ``(x, y)`` pixel coordinates of the circle centre.
            ``radius``  – radius in pixels.
            ``label``   – coin denomination string or ``"Unknown"``.
            ``color``   – detected colour category string.
        """
        h, w = image.shape[:2]
        short_side = min(h, w)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 2)

        min_r = int(short_side * min(p["radius_range"][0] for p in self.coin_profiles))
        max_r = int(short_side * max(p["radius_range"][1] for p in self.coin_profiles))
        min_dist = int(short_side * self.min_dist_factor)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=max(min_r, 1),
            maxRadius=max_r,
        )

        results: list[dict] = []
        if circles is None:
            return results

        circles = np.round(circles[0]).astype(int)
        for cx, cy, r in circles:
            roi = _extract_circular_roi(image, cx, cy, r)
            color_cat = _colour_category(roi)
            label = self._classify(r, short_side, color_cat)
            results.append(
                {
                    "center": (int(cx), int(cy)),
                    "radius": int(r),
                    "label": label,
                    "color": color_cat,
                }
            )

        return results

    def annotate(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw detection results onto *image* and return the annotated copy.

        Parameters
        ----------
        image:
            Original BGR image.
        detections:
            Output from :meth:`detect`.

        Returns
        -------
        np.ndarray
            Annotated BGR image.
        """
        output = image.copy()
        colour_map = {
            "copper": (30, 100, 200),
            "gold":   (0, 200, 220),
            "silver": (200, 200, 200),
        }

        for det in detections:
            cx, cy = det["center"]
            r = det["radius"]
            draw_colour = colour_map.get(det["color"], (0, 255, 0))

            cv2.circle(output, (cx, cy), r, draw_colour, 2)
            cv2.circle(output, (cx, cy), 3, draw_colour, -1)

            label_text = det["label"]
            font_scale = max(0.4, r / 60)
            thickness = max(1, int(font_scale * 2))
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            tx = cx - tw // 2
            ty = cy + th // 2
            cv2.putText(
                output,
                label_text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                draw_colour,
                thickness,
                cv2.LINE_AA,
            )

        return output

    def summarise(self, detections: list[dict]) -> dict:
        """Return a summary of detected coins.

        Parameters
        ----------
        detections:
            Output from :meth:`detect`.

        Returns
        -------
        dict
            ``counts`` – mapping of label → count.
            ``total``  – total number of coins detected.
        """
        counts: dict[str, int] = {}
        for det in detections:
            counts[det["label"]] = counts.get(det["label"], 0) + 1
        return {"counts": counts, "total": len(detections)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self, radius: int, short_side: int, color_cat: str) -> str:
        """Map a detected circle to a coin denomination label.

        The best-matching profile is chosen by the combination of relative
        radius overlap and colour agreement.  If no profile matches, the
        function returns ``"Unknown"``.
        """
        rel_radius = radius / short_side
        best_label = "Unknown"
        best_score = -1.0

        for profile in self.coin_profiles:
            r_min, r_max = profile["radius_range"]
            if rel_radius < r_min or rel_radius > r_max:
                continue

            # Score is 1.0 if colour matches, 0.5 if profile accepts any colour.
            colour_score = 1.0 if profile["color"] in (color_cat, "any") else 0.0
            if colour_score == 0.0:
                # Colour mismatch – allow a weak match only when no better
                # candidate exists.
                colour_score = 0.1

            # Prefer profiles whose midpoint is closest to the detected radius.
            r_mid = (r_min + r_max) / 2
            r_half_width = (r_max - r_min) / 2
            proximity = 1.0 - abs(rel_radius - r_mid) / r_half_width

            score = colour_score * proximity
            if score > best_score:
                best_score = score
                best_label = profile["label"]

        return best_label
