"""
main.py
=======
Command-line interface for the Coin Counting System.

Usage
-----
    python main.py <image_path> [--output <output_path>] [--no-display]
    python main.py <folder_path>   # batch mode -> saves to results/

Examples
--------
    python main.py coins.jpg
    python main.py coins.jpg --output result.jpg --no-display
    python main.py tests/
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import cv2

from coin_detector import CoinDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect and classify coins using circular Hough transform."
    )
    parser.add_argument("image", nargs="?", default="tests", help="Path to an input image or a folder of images. Defaults to tests/.")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save the annotated output image (single-image mode only).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an interactive window to display results.",
    )
    parser.add_argument(
        "--param2",
        type=int,
        default=70,
        help="Hough accumulator threshold (lower = more circles found). Default: 70.",
    )
    return parser.parse_args(argv)


def _print_summary(image_name, result):
    print(f"\n{'='*50}")
    print(f"  {image_name}")
    print(f"{'='*50}")
    print(f"  Coins found : {len(result.coins)}")
    if result.coins:
        counts = Counter(c.label for c in result.coins)
        for label, count in sorted(counts.items()):
            unit_value = next(c.value for c in result.coins if c.label == label)
            print(f"    {label:12s} x{count}  (${unit_value:.2f} each)")
    print(f"  Total value : ${result.total_value:.2f}")
    print(f"{'='*50}")


def main(argv=None):
    args = parse_args(argv)
    detector = CoinDetector(param2=args.param2)
    input_path = Path(args.image)

    # ------------------------------------------------------------------ #
    # Batch mode: input is a folder                                        #
    # ------------------------------------------------------------------ #
    if input_path.is_dir():
        images = sorted(f for f in input_path.iterdir()
                        if f.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            print(f"No images found in {input_path}", file=sys.stderr)
            return 1

        results_dir = input_path.parent / "results"
        results_dir.mkdir(exist_ok=True)

        for img_path in images:
            try:
                result = detector.detect(str(img_path))
            except ValueError as exc:
                print(f"Error processing {img_path.name}: {exc}", file=sys.stderr)
                continue
            _print_summary(img_path.name, result)
            out_path = results_dir / img_path.name
            cv2.imwrite(str(out_path), result.annotated_image)  # type: ignore
            print(f"  Saved -> {out_path}")

        return 0

    # ------------------------------------------------------------------ #
    # Single image mode                                                    #
    # ------------------------------------------------------------------ #
    try:
        result = detector.detect(str(input_path))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(input_path.name, result)

    if args.output:
        cv2.imwrite(args.output, result.annotated_image)  # type: ignore
        print(f"Annotated image saved to: {args.output}")

    if not args.no_display:
        cv2.imshow("Coin Detection", result.annotated_image)  # type: ignore
        print("Press any key to close the window …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
