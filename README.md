# Coin Counting System

Detect and classify coins in an image using the **circular Hough transform** and
**size / colour features**.

## How it works

1. **Preprocessing** – The input image is converted to greyscale and blurred
   with a Gaussian kernel to suppress noise.
2. **Circle detection** – `cv2.HoughCircles` (gradient method) finds circular
   regions that match the expected coin radius range.
3. **Feature extraction** – For each detected circle the mean HSV colour of the
   enclosed region is computed.
4. **Classification** – Each circle is matched to the closest coin profile using
   a combined score of:
   - *Relative radius* (circle radius ÷ image short-side)
   - *Colour category* – `copper` (pennies), `silver` (nickels / dimes /
     quarters), or `gold` (£1, €1 …)
5. **Annotation** – Detected coins are drawn on the image with their label.

## Repository structure

```
coin_detector.py   Core detection / classification module
main.py            CLI entry point
tests.py           Unit tests (pytest)
requirements.txt   Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py <image_path> [--output result.jpg] [--no-display] [--param2 30]
```

| Argument | Description |
|----------|-------------|
| `image`  | Path to the input image (JPEG / PNG / …) |
| `--output` / `-o` | Save annotated image to this path |
| `--no-display` | Skip GUI window (useful in headless environments) |
| `--param2` | HoughCircles accumulator threshold – lower values detect more circles (default: `30`) |

### Example

```bash
python main.py coins.jpg --output result.jpg --no-display
```

Expected output:

```
Coins detected: 5
Breakdown:
  Dime: 1
  Penny: 2
  Quarter: 2
Annotated image saved to: result.jpg
```

## Python API

```python
import cv2
from coin_detector import CoinDetector

image = cv2.imread("coins.jpg")
detector = CoinDetector()

detections = detector.detect(image)   # list of dicts
summary    = detector.summarise(detections)
annotated  = detector.annotate(image, detections)

print(summary)
# {'counts': {'Penny': 2, 'Quarter': 2, 'Dime': 1}, 'total': 5}
cv2.imwrite("result.jpg", annotated)
```

Each detection dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `center` | `(int, int)` | `(x, y)` pixel coordinates of the circle centre |
| `radius` | `int` | Circle radius in pixels |
| `label` | `str` | Coin denomination or `"Unknown"` |
| `color` | `str` | `"copper"`, `"silver"`, or `"gold"` |

## Running tests

```bash
pip install pytest
pytest tests.py -v
```

## Customising coin profiles

Pass your own list of profiles to `CoinDetector`:

```python
profiles = [
    {"label": "1p",  "color": "copper", "radius_range": (0.04, 0.08)},
    {"label": "2p",  "color": "copper", "radius_range": (0.07, 0.11)},
    {"label": "10p", "color": "silver", "radius_range": (0.08, 0.13)},
    {"label": "£1",  "color": "gold",   "radius_range": (0.09, 0.14)},
]
detector = CoinDetector(coin_profiles=profiles)
```
