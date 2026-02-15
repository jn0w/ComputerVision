
import numpy as np
import cv2


# 1. Histogram-based automatic threshold (Otsu's method)

def otsu_threshold(img: np.ndarray) -> int:
    """Compute the optimal binarization threshold using Otsu's method.

    Parameters
    img : np.ndarray
        Grayscale image (uint8, 2-D).

    Returns
    int
        Optimal threshold value in [0, 255].
    """
    # Step 1: Count how many pixels have each brightness level (0 to 255)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total_pixels = img.size

    # Step 2: Pre-calculate running totals to speed things up
    # cum_sum: how many pixels are at or below each brightness level
    # cum_mean: total brightness of all pixels at or below each level
    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))

    global_mean = cum_mean[-1]  # total brightness of the whole image

    # Step 3: Try every possible threshold value and pick the best one
    best_t = 0
    best_var = -1.0

    for t in range(256):
        # Split pixels into two groups: dark ones (w0) and bright ones (w1)
        w0 = cum_sum[t]  # how many pixels are dark (at or below threshold)
        w1 = total_pixels - w0  # how many pixels are bright (above threshold)
        
        # Skip if one group is empty (can't use this threshold)
        if w0 == 0 or w1 == 0:
            continue

        # Find average brightness for each group
        mu0 = cum_mean[t] / w0  # average brightness of dark pixels
        mu1 = (global_mean - cum_mean[t]) / w1  # average brightness of bright pixels

        # Pick the threshold that best separates dark from bright
        # Higher variance means better separation
        between_var = w0 * w1 * (mu0 - mu1) ** 2
        if between_var > best_var:
            best_var = between_var
            best_t = t

    return int(best_t)



# 2. Binarization with automatic polarity detection


def binarize(img: np.ndarray, threshold: int) -> np.ndarray:
    """Threshold image to binary, auto-detecting foreground polarity.

    Compares mean intensity of a 10-pixel border strip with the overall
    mean.  If the border is brighter, the O-ring is dark and we set
    foreground = (img <= threshold).

    Returns
    np.ndarray
        Boolean mask where True = foreground (O-ring).
    """
    border_width = 10
    h, w = img.shape

    # Look at the edges of the image (top, bottom, left, right)
    # The edges are usually background, so we can use them to figure out
    # whether the O-ring is darker or brighter than the background
    border_pixels = np.concatenate([
        img[:border_width, :].ravel(),  # top edge
        img[-border_width:, :].ravel(),  # bottom edge
        img[border_width:-border_width, :border_width].ravel(),  # left edge
        img[border_width:-border_width, -border_width:].ravel(),  # right edge
    ])

    # Compare edge brightness with overall image brightness
    border_mean = border_pixels.mean()
    overall_mean = img.mean()

    # Figure out if O-ring is dark or bright:
    # If edges are bright, O-ring is probably dark (use pixels below threshold)
    # If edges are dark, O-ring is probably bright (use pixels above threshold)
    if border_mean >= overall_mean:
        # Dark O-ring on light background
        binary = img <= threshold
    else:
        # Light O-ring on dark background
        binary = img >= threshold

    return binary


# 3. Binary morphology (from scratch)

def erode(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Erode a binary mask with the given structuring element.

    A foreground pixel survives only if ALL positions under the SE are
    foreground.
    """
    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2

    padded = np.pad(mask, ((ph, ph), (pw, pw)), mode='constant',
                    constant_values=False)

    result = np.ones_like(mask, dtype=bool)
    for dr in range(sh):
        for dc in range(sw):
            if se[dr, dc]:
                result &= padded[dr:dr + mask.shape[0],
                                 dc:dc + mask.shape[1]]
    return result


def dilate(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Dilate a binary mask with the given structuring element.

    A pixel becomes foreground if ANY position under the SE is foreground.
    """
    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2

    padded = np.pad(mask, ((ph, ph), (pw, pw)), mode='constant',
                    constant_values=False)

    result = np.zeros_like(mask, dtype=bool)
    for dr in range(sh):
        for dc in range(sw):
            if se[dr, dc]:
                result |= padded[dr:dr + mask.shape[0],
                                 dc:dc + mask.shape[1]]
    return result


def closing(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Morphological closing: dilate then erode.  Closes small holes/gaps."""
    return erode(dilate(mask, se), se)


def opening(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Morphological opening: erode then dilate.  Removes small noise."""
    return dilate(erode(mask, se), se)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask using border flood-fill.

    Algorithm:
      1. Start from every background pixel on the image border.
      2. BFS/flood-fill (4-connected) on the *background*.
      3. Everything NOT reached is either foreground or an interior hole.
      4. Return original foreground UNION interior holes.
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)

    # Start from all background pixels on the edges
    stack = []
    for r in range(h):
        for c in [0, w - 1]:
            if not mask[r, c] and not visited[r, c]:
                visited[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if not mask[r, c] and not visited[r, c]:
                visited[r, c] = True
                stack.append((r, c))

    # Spread out from edges, marking all background pixels we can reach
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and not mask[nr, nc]:
                visited[nr, nc] = True
                stack.append((nr, nc))

    # Any pixel we couldn't reach from the edges is either the O-ring or a hole inside it
    # Fill in those holes by making them part of the O-ring
    filled = mask | ~visited
    return filled

