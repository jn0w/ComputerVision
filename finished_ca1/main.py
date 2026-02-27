"""
program for O-ring inspection

commands to run the code:
    python main.py --input ./Orings --output ./out --save

This script reads all images from the input folder, runs the inspection
pipeline on each image, classifies every O-ring as PASS or FAIL, and
prints one CSV-style line per image with the main metrics. It can also
save annotated images to disk.
"""

import argparse
import os
import sys
import time
import glob

import cv2

from oring import inspect_image
from utils import annotate_image


def main():
    """Parse command-line arguments and run the inspection on a folder."""
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="O-ring inspection – classify ring images as PASS/FAIL")
    # Folder with input images
    parser.add_argument("--input", type=str, default="./Orings",
                        help="Path to folder containing O-ring images")
    # Where to save annotated output images
    parser.add_argument("--output", type=str, default="./out",
                        help="Path to folder for annotated output images")
    # If given, save annotated images
    parser.add_argument("--save", action="store_true",
                        help="Save annotated images to the output folder")
    args = parser.parse_args()

    # Collect image paths from the input directory
    input_dir = args.input
    if not os.path.isdir(input_dir):
        print(f"Error: input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # support common image filename extensions
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths.sort()

    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        sys.exit(1)

    # Prepare output directory to save imagese
    if args.save:
        os.makedirs(args.output, exist_ok=True)

    # Print CSV header (column names)
    header = ("filename,threshold,area,thickness_mean,thickness_std,"
              "thickness_cv,missing_frac,max_dip,holes_ratio,blobs,result,ms")
    print(header)

    # Process each image in turn
    for path in image_paths:
        # Just the file name, without the folder path
        filename = os.path.basename(path)

        # Measure processing time for this image
        t_start = time.perf_counter()
        result = inspect_image(path)
        t_ms = (time.perf_counter() - t_start) * 1000.0

        # Build one CSV line with the most important metrics
        line = (f"{filename},"
                f"{result['threshold']},"
                f"{result['area']},"
                f"{result['thickness_mean']:.2f},"
                f"{result['thickness_std']:.2f},"
                f"{result['thickness_cv']:.3f},"
                f"{result['missing_fraction']:.3f},"
                f"{result.get('max_dip', 0.0):.1f},"
                f"{result['holes_ratio']:.3f},"
                f"{result['extra_blobs']},"
                f"{result['result']},"
                f"{t_ms:.1f}")
        print(line)
        # If the ring failed, print the reasons on the next line
        reasons = result.get('reasons', [])
        if result['result'] == 'FAIL' and reasons:
            print("    -> " + "; ".join(reasons))

        # Annotate and optionally save (PNG keeps text sharp; JPEG blurs it)
        if args.save:
            annotated = annotate_image(
                result['img'], filename, result['result'], t_ms, result)
            base, _ = os.path.splitext(filename)
            out_path = os.path.join(args.output, base + ".png")
            cv2.imwrite(out_path, annotated)

    if args.save:
        print(f"\nAnnotated images saved to '{args.output}/'.")


if __name__ == "__main__":
    main()

