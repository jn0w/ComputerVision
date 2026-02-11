"""
oring.py – Core pipeline functions for O-ring inspection.

All image processing is implemented from scratch using NumPy.
OpenCV is used ONLY for loading images (cv2.imread).
"""

import numpy as np
import cv2


# ===================================================================
# 1. Histogram-based automatic threshold (Otsu's method)
# ===================================================================

def otsu_threshold(img: np.ndarray) -> int:
    """Compute the optimal binarization threshold using Otsu's method.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image (uint8, 2-D).

    Returns
    -------
    int
        Optimal threshold value in [0, 255].
    """
    # Step 1: Build 256-bin histogram (count pixels at each intensity 0-255)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total_pixels = img.size

    # Step 2: Pre-compute cumulative sums for efficiency
    # cum_sum: cumulative count of pixels up to each threshold
    # cum_mean: cumulative sum of intensity values up to each threshold
    cum_sum = np.cumsum(hist)                       # ω(t) * N
    cum_mean = np.cumsum(hist * np.arange(256))     # μ(t) * N

    global_mean = cum_mean[-1]  # total intensity sum (mean of entire image)

    # Step 3: Try every possible threshold (0-255) and find the best one
    best_t = 0
    best_var = -1.0

    for t in range(256):
        # Split pixels into two groups: below threshold (w0) and above (w1)
        w0 = cum_sum[t]  # number of pixels <= threshold
        w1 = total_pixels - w0  # number of pixels > threshold
        
        # Skip if one group is empty (invalid threshold)
        if w0 == 0 or w1 == 0:
            continue

        # Calculate mean intensity for each group
        mu0 = cum_mean[t] / w0  # mean of pixels <= threshold
        mu1 = (global_mean - cum_mean[t]) / w1  # mean of pixels > threshold

        # Otsu's criterion: maximize between-class variance
        # Higher variance = better separation between foreground and background
        between_var = w0 * w1 * (mu0 - mu1) ** 2
        if between_var > best_var:
            best_var = between_var
            best_t = t

    return int(best_t)


# ===================================================================
# 2. Binarization with automatic polarity detection
# ===================================================================

def binarize(img: np.ndarray, threshold: int) -> np.ndarray:
    """Threshold image to binary, auto-detecting foreground polarity.

    Compares mean intensity of a 10-pixel border strip with the overall
    mean.  If the border is brighter, the O-ring is dark and we set
    foreground = (img <= threshold).

    Returns
    -------
    np.ndarray
        Boolean mask where True = foreground (O-ring).
    """
    border_width = 10
    h, w = img.shape

    # Collect border pixels (top, bottom, left, right strips)
    # Border pixels are likely background, so we use them to determine polarity
    border_pixels = np.concatenate([
        img[:border_width, :].ravel(),  # top border
        img[-border_width:, :].ravel(),  # bottom border
        img[border_width:-border_width, :border_width].ravel(),  # left border
        img[border_width:-border_width, -border_width:].ravel(),  # right border
    ])

    # Compare border intensity with overall image intensity
    border_mean = border_pixels.mean()
    overall_mean = img.mean()

    # Auto-detect polarity: if border is brighter than average,
    # then the O-ring is darker (foreground = pixels below threshold)
    # Otherwise, O-ring is brighter (foreground = pixels above threshold)
    if border_mean >= overall_mean:
        # Dark ring on light background: foreground is below threshold
        binary = img <= threshold
    else:
        # Light ring on dark background: foreground is above threshold
        binary = img >= threshold

    return binary

