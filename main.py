"""
main.py
=======
Command-line interface for the Coin Counting System.

Usage
-----
    python main.py <image_path> [--output <output_path>] [--no-display]

Examples
--------
    python main.py coins.jpg
    python main.py coins.jpg --output result.jpg --no-display
"""

import argparse
import sys

import cv2

from coin_detector import CoinDetector


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect and classify coins using circular Hough transform."
    )
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save the annotated output image (optional).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an interactive window to display results.",
    )
    parser.add_argument(
        "--param2",
        type=int,
        default=30,
        help="Hough accumulator threshold (lower = more circles found). Default: 30.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    detector = CoinDetector(param2=args.param2)

    try:
        result = detector.detect(args.image)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Print summary to stdout                                              #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*50}")
    print(f"  Coin Detection Results")
    print(f"{'='*50}")
    print(f"  Coins found : {len(result.coins)}")

    if result.coins:
        # Count by denomination
        from collections import Counter
        counts = Counter(c.label for c in result.coins)
        for label, count in sorted(counts.items()):
            unit_value = next(c.value for c in result.coins if c.label == label)
            print(f"    {label:12s} x{count}  (${unit_value:.2f} each)")

    print(f"  Total value : ${result.total_value:.2f}")
    print(f"{'='*50}\n")

    # ------------------------------------------------------------------ #
    # Save / display annotated image                                       #
    # ------------------------------------------------------------------ #
    if args.output:
        cv2.imwrite(args.output, result.annotated_image)
        print(f"Annotated image saved to: {args.output}")

    if not args.no_display:
        cv2.imshow("Coin Detection", result.annotated_image)
        print("Press any key to close the window …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
