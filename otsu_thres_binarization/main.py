import os
import glob
import cv2
import numpy as np

from oring import otsu_threshold, binarize


def main():
    # Set input and output directories
    image_dir = "./Orings"
    out_dir = "./out"
    os.makedirs(out_dir, exist_ok=True)

    # Find all JPG images in the input directory
    paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not paths:
        print("No images found in", image_dir)
        return

    print(f"Found {len(paths)} images\n")
    # Print header for results table
    print(f"{'filename':<16s} {'threshold':>9s} {'fg_pixels':>10s}")
    print("-" * 38)

    # Process each image
    for path in paths:
        filename = os.path.basename(path)
        
        # Load image as grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Step 1: Compute optimal threshold using Otsu's method
        t = otsu_threshold(img)
        
        # Step 2: Convert grayscale to binary mask (True = foreground/O-ring)
        binary = binarize(img, t)

        # Count foreground pixels (True values in binary mask)
        fg_count = int(binary.sum())
        print(f"  {filename:<14s} {t:>9d} {fg_count:>10d}")

        # Save binary image for visual inspection
        # Convert boolean to uint8: True -> 255 (white), False -> 0 (black)
        binary_img = (binary.astype(np.uint8)) * 255
        cv2.imwrite(os.path.join(out_dir, f"bin_{filename}"), binary_img)

    print(f"\nBinary images saved to '{out_dir}/'")


if __name__ == "__main__":
    main()

