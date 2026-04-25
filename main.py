"""Command-line interface for the Coin Counting System.

Usage
-----
    python main.py <image_path> [--output <output_path>] [--no-display]

Arguments
---------
image_path
    Path to the input image (JPEG, PNG, …).
--output, -o
    Path where the annotated image should be saved.  Optional.
--no-display
    Do not attempt to show the result in a GUI window.

Example
-------
    python main.py coins.jpg --output result.jpg
"""

import argparse
import sys

import cv2

from coin_detector import CoinDetector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and classify coins in an image using circular Hough transform."
    )
    parser.add_argument("image", help="Path to the input image file.")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save the annotated image to this path.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not display the result in a GUI window.",
    )
    parser.add_argument(
        "--param2",
        type=float,
        default=30,
        help="Accumulator threshold for HoughCircles (lower = more detections). Default: 30.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: could not load image '{args.image}'", file=sys.stderr)
        return 1

    detector = CoinDetector(param2=args.param2)

    detections = detector.detect(image)
    summary = detector.summarise(detections)
    annotated = detector.annotate(image, detections)

    # Print summary to stdout.
    print(f"Coins detected: {summary['total']}")
    if summary["counts"]:
        print("Breakdown:")
        for label, count in sorted(summary["counts"].items()):
            print(f"  {label}: {count}")
    else:
        print("No coins were detected.")

    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"Annotated image saved to: {args.output}")

    if not args.no_display:
        cv2.imshow("Coin Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
