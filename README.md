# Coin Counting System

Detect and classify coins in an image using the **circular Hough transform**
combined with **size** and **color** features.

## Features

- Detects circular coin regions via `cv2.HoughCircles`
- Classifies each coin by its **relative radius** (robust to scale) and
  **dominant colour** (copper / gold / silver)
- Supports US coins out of the box: penny, nickel, dime, quarter, half-dollar,
  dollar — with an easy-to-extend classification table
- Computes the **total monetary value** of all detected coins
- Produces an **annotated image** with labelled circles

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command line

```bash
python main.py <image_path> [--output <result.jpg>] [--no-display] [--param2 30]
```

| Option | Default | Description |
|--------|---------|-------------|
| `image` | *(required)* | Path to input image |
| `--output` / `-o` | — | Save annotated image to this path |
| `--no-display` | off | Skip the interactive preview window |
| `--param2` | 30 | Hough accumulator threshold (lower => more circles) |

Example:

```bash
python main.py coins.jpg --output annotated.jpg --no-display
```

Output:

```
==================================================
  Coin Detection Results
==================================================
  Coins found : 4
    dime         x1  ($0.10 each)
    nickel       x1  ($0.05 each)
    penny        x1  ($0.01 each)
    quarter      x1  ($0.25 each)
  Total value : $0.41
==================================================
```

### Python API

```python
from coin_detector import CoinDetector

detector = CoinDetector()
result = detector.detect("coins.jpg")

print(f"Found {len(result.coins)} coins")
print(f"Total value: ${result.total_value:.2f}")

for coin in result.coins:
    print(f"  {coin.label} at ({coin.x}, {coin.y}) r={coin.radius}px  colour={coin.color_name}")
```

## How it works

1. **Pre-processing** — the input image is converted to grayscale and
   smoothed with a Gaussian blur to reduce noise.
2. **Hough circle detection** — `cv2.HoughCircles` is applied with
   scale-relative `minDist`, `minRadius`, and `maxRadius` parameters
   so that the detector adapts automatically to different image sizes.
3. **Colour sampling** — the mean BGR colour is sampled from the inner
   60 % of each detected circle (avoiding rim artefacts) and mapped to
   *copper*, *gold*, or *silver* using HSV thresholds.
4. **Classification** — each coin's radius is compared to the median
   radius of all detected coins (the *radius ratio*).  Combined with the
   colour group, the closest entry in the classification table determines
   the denomination.  A size-only fallback is used when the colour does
   not match any table entry.

## Customising for other currencies

Pass a custom `coin_table` to `CoinDetector`:

```python
from coin_detector import CoinDetector

MY_TABLE = [
    # label,     value,   color,     min_ratio, max_ratio
    ("1 EGP",    1.00,   "silver",   0.90,      1.10),
    ("50 pt",    0.50,   "silver",   0.75,      0.89),
    ("25 pt",    0.25,   "copper",   0.60,      0.74),
]

detector = CoinDetector(coin_table=MY_TABLE)
result = detector.detect("egyptian_coins.jpg")
```

## Running tests

```bash
pip install pytest
python -m pytest test_coin_detector.py -v
```

## Dependencies

- [OpenCV](https://opencv.org/) (`opencv-python-headless`)
- [NumPy](https://numpy.org/)
