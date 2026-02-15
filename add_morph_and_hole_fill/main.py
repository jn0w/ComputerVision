

import os
import glob
import cv2
import numpy as np

from oring import otsu_threshold, binarize, opening, closing, fill_holes


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

    # Small 3x3 square used for cleaning up the image
    se = np.ones((3, 3), dtype=bool)

    # Print table header
    print(f"{'filename':<16s} {'thresh':>6s} {'raw_fg':>8s} {'cleaned_fg':>11s} {'filled_fg':>10s}")
    print("-" * 55)

    # Go through each image
    for path in paths:
        filename = os.path.basename(path)
        
        # Load the image (convert to black and white)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Step 1: Find the best threshold value to separate O-ring from background
        t = otsu_threshold(img)
        
        # Step 2: Turn the image into black and white (O-ring = white, background = black)
        binary = binarize(img, t)
        
        # Count how many white pixels we have (the O-ring)
        raw_fg = int(binary.sum())

        # Step 3: Clean up the image
        cleaned = opening(binary, se)   # Remove tiny noise spots
        cleaned = closing(cleaned, se)  # Fill in small gaps
        cleaned_fg = int(cleaned.sum())

        # Step 4: Fill any holes inside the O-ring
        filled = fill_holes(cleaned)
        filled_fg = int(filled.sum())

        print(f"  {filename:<14s} {t:>6d} {raw_fg:>8d} {cleaned_fg:>11d} {filled_fg:>10d}")

        # Save the cleaned image
        # Convert True/False to white (255) and black (0) so we can see it
        out_img = (cleaned.astype(np.uint8)) * 255
        cv2.imwrite(os.path.join(out_dir, f"clean_{filename}"), out_img)

    print(f"\nCleaned binary images saved to '{out_dir}/'")


if __name__ == "__main__":
    main()

