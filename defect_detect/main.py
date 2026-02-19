#!/usr/bin/env python3

import os
import glob
import time
import cv2

from oring import inspect_image


def main():
    # Where to find images and save results
    image_dir = "./Orings"
    out_dir = "./out"
    os.makedirs(out_dir, exist_ok=True)

    # Get all JPG images from the folder
    paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not paths:
        print("No images found in", image_dir)
        return

    # Print table header
    print(f"{'filename':<16s} {'thresh':>6s} {'area':>7s} {'th_cv':>7s} "
          f"{'miss':>6s} {'holes':>7s} {'blobs':>6s} {'result':>7s} {'ms':>7s}")
    print("-" * 75)

    # Go through each image
    for path in paths:
        filename = os.path.basename(path)

        # Inspect the image and measure how long it takes
        t0 = time.perf_counter()
        result = inspect_image(path)
        dt = (time.perf_counter() - t0) * 1000

        # Print the results
        print(f"  {filename:<14s} {result['threshold']:>6d} {result['area']:>7d} "
              f"{result['thickness_cv']:>7.3f} {result['missing_fraction']:>6.3f} "
              f"{result['holes_ratio']:>7.3f} {result['extra_blobs']:>6d} "
              f"{result['result']:>7s} {dt:>6.1f}")

        # Save annotated image with result text
        out = cv2.cvtColor(result['img'], cv2.COLOR_GRAY2BGR)
        colour = (0, 200, 0) if result['result'] == "PASS" else (0, 0, 220)
        cv2.putText(out, result['result'], (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, filename), out)

    print(f"\nAnnotated images saved to '{out_dir}/'")


if __name__ == "__main__":
    main()

