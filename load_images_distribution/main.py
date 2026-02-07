#main.py – Load O-ring images, print info about the images 

import os
import glob
import cv2
import numpy as np


def main():
    # directory for images
    image_dir = "Orings"

    # get all images in the directory
    paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not paths:
        print("No images found in", image_dir)
        return

    print(f"Found {len(paths)} images\n")

    # Loop through each image file
    for path in paths:
        # Extract just the filename from the file path
        filename = os.path.basename(path)
        
        # Load the image as grayscale for simpler processing
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  {filename}: FAILED to load")  
            continue

        # Print basic statistics about the image
        # dimensions of the image height, width
        # dtype: data type unsigned integer, 0-255 or 8 bits
        # min/max: darkest and brightest pixel 
        # mean: average intensity across all pixels
        print(f"  {filename}: shape={img.shape}  dtype={img.dtype}  "
              f"min={img.min()}  max={img.max()}  mean={img.mean():.1f}")

    
    print("\n--- Histogram of first image ---")
    
    # Reload the first image to analyze its intensity distribution
    img = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    
    # Count how many pixels have each intensity value 
    # bincount creates a 256-element array where each index represents
    # how many pixels have that intensity value
    hist = np.bincount(img.ravel(), minlength=256)

    # Print histogram in 8 bins (each bin covers 32 intensity levels)
    # This gives us a view of the intensity distribution
    for start in range(0, 256, 32):
        end = min(start + 32, 256)
        # Sum all pixel counts in this intensity range
        count = hist[start:end].sum()
        # Create a visual bar: each '#' represents ~200 pixels
        bar = "#" * (count // 200)
        print(f"  [{start:3d}-{end:3d}): {count:6d} {bar}")


if __name__ == "__main__":
    main()