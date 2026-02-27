"""
Main code for inspecting O-ring images.

All image processing steps, such as thresholding, morphology,
feature extraction and the decision rules, are written using
plain NumPy. OpenCV is only used to read images from disk
with cv2.imread.
"""

import numpy as np
import cv2
import math

# Classification thresholds
# THRESH_THICKNESS_CV controls how much the ring thickness is allowed
# to change as we go around the circle. Bigger values = more change
# allowed before we call the ring defective.
THRESH_THICKNESS_CV = 0.08

# THRESH_MISSING_FRAC is starting from the center of the ring,
# at each angle of the circle the thickness is measured
# then we get the average thickness so if more than 3% of the angles are missing = fail
THRESH_MISSING_FRAC = 0.03

# THRESH_MAX_DIP is the largest drop in thickness (in pixels) that we
# allow at any angle compared to a smoothed, expected thickness. Bigger
# drops usually mean a deep nick or missing chunk.
THRESH_MAX_DIP = 2.5

# THRESH_HOLES_RATIO says how much of the ring band is allowed to be
# empty (background) instead of ring material. Higher values mean more
# holes in the ring are allowed.
THRESH_HOLES_RATIO = 0.05

# MIN_BLOB_AREA is the smallest size (in pixels) for extra blobs that we
# treat as real debris. Tiny specks smaller than this are ignored.
MIN_BLOB_AREA = 30

# DIP_SMOOTH_WINDOW says how wide the angle window is when we smooth
# the thickness curve. For each angle we average over +/- this many
# degrees to get a smooth “expected” thickness.
DIP_SMOOTH_WINDOW = 15


# 1. Histogram-based automatic threshold using Otsu's method

def otsu_threshold(img: np.ndarray) -> int:
    """Find a good gray-level threshold using Otsu's method.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image (uint8, 2-D).

    This automatically picks a threshold that best separates darker
    and brighter pixels in the image.

    Returns
    -------
    int
        Threshold value in [0, 255].
    """
    # Step 1: build a histogram of all gray values (0..255) in the image.
    # hist[g] tells us how many pixels in the image have value g.
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total_pixels = img.size

    # Step 2: make running sums across gray levels.
    # running_counts[t] = number of pixels with value 0..t
    # running_sums[t]   = sum of (gray value * count) for 0..t
    running_counts = np.cumsum(hist)
    running_sums = np.cumsum(hist * np.arange(256))

    # total_intensity_sum is the sum of all pixel values in the image.
    total_intensity_sum = running_sums[-1]

    # We will search all possible thresholds t in [0, 255] and keep the one
    # that best separates dark and bright pixels (largest between variance)
    best_t = 0
    best_var = -1.0

    for t in range(256):
        # w0 = number of pixels on the "dark" side (<= t)
        # w1 = number of pixels on the "bright" side (> t)
        w0 = running_counts[t]
        w1 = total_pixels - w0
        # If all pixels are on one side, this t is useless; skip it.
        if w0 == 0 or w1 == 0:
            continue

        # mu0 = average gray value of the dark pixels (0..t)
        # mu1 = average gray value of the bright pixels (t+1..255)
        mu0 = running_sums[t] / w0
        mu1 = (total_intensity_sum - running_sums[t]) / w1

        # between_var measures how far apart these two averages are,
        # weighted by how many pixels are in each group.
        between_var = w0 * w1 * (mu0 - mu1) ** 2

        # If this threshold gives a better separation, remember it.
        if between_var > best_var:
            best_var = between_var
            best_t = t

    # best_t is the gray value that best separates dark and bright pixels.
    return int(best_t)


# 2. Binarization with automatic foreground/background detection

def binarize(img: np.ndarray, threshold: int) -> np.ndarray:
    """Convert the grayscale image into a binary mask.

    We first check whether the background is brighter or darker than the
    O-ring by comparing the border pixels to the global mean.  If the
    border is brighter, the O-ring is dark and we use (img <= threshold)
    as foreground; otherwise we use (img >= threshold).

    Returns
    -------
    np.ndarray
        Boolean mask where True = foreground (O-ring).
    """
    # Step 1: decide how thick a frame around the image we treat as "border".
    border_width = 10
    h, w = img.shape

    # Step 2: collect all pixels from the image border
    # (top row, bottom row, left strip, right strip).
    border_pixels = np.concatenate([
        img[:border_width, :].ravel(),                # top
        img[-border_width:, :].ravel(),               # bottom
        img[border_width:-border_width, :border_width].ravel(),   # left
        img[border_width:-border_width, -border_width:].ravel(),  # right
    ])

    # Step 3: compare the average brightness of the border to the whole image.
    border_mean = border_pixels.mean()
    overall_mean = img.mean()

    # Step 4: decide which side is the ring based on brightness.
    # If the border is brighter than the whole image, the ring is darker,
    # so we treat pixels <= threshold as ring (foreground).
    # Otherwise, the ring is brighter, so we treat pixels >= threshold as ring.
    if border_mean >= overall_mean:
        binary = img <= threshold
    else:
        binary = img >= threshold

    # Step 5: return a True/False mask where True marks the ring pixels.
    return binary


# 3. Binary morphology 
#true = ring, false = background

def erode(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Shrink foreground regions using the given structuring element.

    A foreground pixel stays 1 only if ALL 1-positions of the SE
    overlap foreground pixels in the mask.
    """
    # se is a small True/False pattern (structuring element).
    # We slide this pattern over the mask and only keep pixels where
    # the full pattern fits completely inside the foreground.
    se_height, se_width = se.shape
    pad_rows, pad_cols = se_height // 2, se_width // 2

    # Pad the mask so we can slide the SE near the edges without
    # running out of image.
    padded = np.pad(mask, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='constant',
                    constant_values=False)

    # Start with everything set to True and AND in each shifted view
    # wherever the SE has a True.
    result = np.ones_like(mask, dtype=bool)
    for shift_row in range(se_height):
        for shift_col in range(se_width):
            if se[shift_row, shift_col]:
                result &= padded[shift_row:shift_row + mask.shape[0],
                                 shift_col:shift_col + mask.shape[1]]
    return result


def dilate(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Grow foreground regions using the given structuring element.

    A pixel becomes 1 if ANY 1-position of the SE overlaps a foreground
    pixel in the mask.
    """
    # Here we also slide the SE over the image, but now we mark a pixel
    # as foreground if at least one position of the SE hits a True pixel.
    se_height, se_width = se.shape
    pad_rows, pad_cols = se_height // 2, se_width // 2

    # Pad the mask as before so we can look around every pixel.
    padded = np.pad(mask, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='constant',
                    constant_values=False)

    # Start with everything set to False and OR in each shifted view
    # wherever the SE has a True.
    result = np.zeros_like(mask, dtype=bool)
    for shift_row in range(se_height):
        for shift_col in range(se_width):
            if se[shift_row, shift_col]:
                result |= padded[shift_row:shift_row + mask.shape[0],
                                 shift_col:shift_col + mask.shape[1]]
    return result


def closing(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Morphological closing: dilate then erode.

    This tends to close small holes and small gaps in the ring.
    """
    # First grow the foreground a bit (dilate), then shrink it back
    # (erode). Small gaps and tiny holes get filled in during this.
    return erode(dilate(mask, se), se)


def opening(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Morphological opening: erode then dilate.

    This removes small isolated noise specks.
    """
    # First shrink the foreground (erode), which kills small isolated
    # dots, then grow it back (dilate) so the main shapes return to
    # roughly their original size.
    return dilate(erode(mask, se), se)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask using a border flood-fill.

    The idea is to start from background pixels on the image border,
    flood-fill all background pixels that connect to this border, and
    then treat any remaining background pixels as holes inside the
    object. Those holes are turned into foreground pixels, and the
    filled mask is returned.
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)

    # Step 1: push all background pixels that lie on the image border
    # onto a stack. These are background pixels that are definitely not
    # holes (they touch the outside of the object).
    stack = []
    for row in range(h):
        for col in [0, w - 1]:
            if not mask[row, col] and not visited[row, col]:
                visited[row, col] = True
                stack.append((row, col))
    for col in range(w):
        for row in [0, h - 1]:
            if not mask[row, col] and not visited[row, col]:
                visited[row, col] = True
                stack.append((row, col))

    # Step 2: flood-fill all background pixels connected to this border
    # using 4-neighbour connectivity (up, down, left, right).
    while stack:
        row, col = stack.pop()
        for row_offset, col_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_row = row + row_offset
            neighbor_col = col + col_offset
            if 0 <= neighbor_row < h and 0 <= neighbor_col < w and not visited[neighbor_row, neighbor_col] and not mask[neighbor_row, neighbor_col]:
                visited[neighbor_row, neighbor_col] = True
                stack.append((neighbor_row, neighbor_col))

    # Step 3: any background pixel that we did NOT reach from the border
    # must be an interior hole. We turn those into foreground.
    # visited == True  -> background connected to outside
    # visited == False -> either ring or interior hole
    filled = mask | ~visited
    return filled


# 4. Connected component labeling with an eight-neighbour search

def connected_components(mask: np.ndarray):
    """Label 8-connected blobs in a boolean mask.

    Returns
    -------
    labels : np.ndarray (int32)
        Label image; 0 = background, 1..N are object ids.
    components : list[dict]
        For each object we store: label id, pixel area, and bounding box
        (rmin, rmax, cmin, cmax).
    """
    # Get image height and width.
    h, w = mask.shape
    # labels will store an integer id for each blob (0 = background).
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    components = []

    # 8-neighbour offsets: (row_offset, col_offset) for up/down/left/right and the 4 diagonals.
    neighbour_offsets_8 = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),           (0, 1),
                           (1, -1),  (1, 0),  (1, 1)]

    # Scan every pixel in the image.
    for row in range(h):
        for col in range(w):
            # If this pixel is foreground and not yet labelled, we start
            # a new blob from here.
            if mask[row, col] and labels[row, col] == 0:
                current_label += 1
                stack = [(row, col)]
                labels[row, col] = current_label
                area = 0
                # Initialise bounding box for this blob.
                rmin, rmax, cmin, cmax = row, row, col, col

                # Depth-first search (DFS) using a stack to grow the blob.
                while stack:
                    current_row, current_col = stack.pop()
                    area += 1
                    # Update bounding box as we visit new pixels.
                    if current_row < rmin:
                        rmin = current_row
                    if current_row > rmax:
                        rmax = current_row
                    if current_col < cmin:
                        cmin = current_col
                    if current_col > cmax:
                        cmax = current_col

                    # Look at all 8 neighbours of the current pixel.
                    for row_offset, col_offset in neighbour_offsets_8:
                        neighbor_row = current_row + row_offset
                        neighbor_col = current_col + col_offset
                        # If neighbour is inside image, foreground, and
                        # not yet labelled, add it to this blob.
                        if 0 <= neighbor_row < h and 0 <= neighbor_col < w and mask[neighbor_row, neighbor_col] and labels[neighbor_row, neighbor_col] == 0:
                            labels[neighbor_row, neighbor_col] = current_label
                            stack.append((neighbor_row, neighbor_col))

                # After the blob is fully explored, save its info.
                components.append({
                    'label': current_label,
                    'area': area,
                    'bbox': (rmin, rmax, cmin, cmax),
                })

    # labels: image with blob ids, components: list with stats per blob.
    return labels, components


# 5. Feature extraction

def extract_features(mask: np.ndarray, component: dict,
                     filled_mask: np.ndarray,
                     labels: np.ndarray,
                     all_components: list,
                     extra_blobs_pre: int) -> dict:
    """Compute features that describe the quality of the main O-ring.

    Parameters
    ----------
    mask            : cleaned binary mask (after morphology)
    component       : dict for the largest component
    filled_mask     : mask after hole-filling
    labels          : label image from CCL on cleaned mask
    all_components  : list of all component dicts
    extra_blobs_pre : number of extra blobs detected on the pre-opening
                      binary mask (for debris detection)

    Returns
    -------
    dict of features.
    """
    lbl = component['label']
    comp_mask = labels == lbl

    # --- Area ---
    area = int(comp_mask.sum())

    # --- Centroid ---
    rows, cols = np.where(comp_mask)
    centroid_r = float(rows.mean())
    centroid_c = float(cols.mean())

    # --- Bounding box ---
    bbox = component['bbox']

    # --- Radial thickness profile ---
    # For each angle from 0..359 degrees we shoot a ray from the centroid
    # and record where we first enter and finally leave the ring.  The
    # difference is the local thickness at that angle.
    h, w = mask.shape
    max_radius = int(math.sqrt(h ** 2 + w ** 2))
    n_angles = 360
    thicknesses = np.zeros(n_angles, dtype=np.float64)
    r_inners = np.zeros(n_angles, dtype=np.float64)
    r_outers = np.zeros(n_angles, dtype=np.float64)

    for i in range(n_angles):
        angle_rad = math.radians(i)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        r_inner = -1
        r_outer = -1

        for radius in range(1, max_radius):
            pr = int(round(centroid_r + radius * cos_a))
            pc = int(round(centroid_c + radius * sin_a))
            if pr < 0 or pr >= h or pc < 0 or pc >= w:
                break
            if comp_mask[pr, pc]:
                if r_inner < 0:
                    r_inner = radius
                r_outer = radius

        if r_inner > 0 and r_outer > 0:
            thicknesses[i] = r_outer - r_inner
            r_inners[i] = r_inner
            r_outers[i] = r_outer
        else:
            thicknesses[i] = 0.0
            r_inners[i] = 0.0
            r_outers[i] = 0.0

    # --- Global thickness statistics ---
    # thickness_mean: average radial thickness around the whole ring.
    # thickness_cv   : relative variation of thickness (std / mean).
    thickness_mean = float(thicknesses.mean()) if thicknesses.sum() > 0 else 0.0
    thickness_std = float(thicknesses.std()) if thicknesses.sum() > 0 else 0.0
    thickness_cv = (thickness_std / thickness_mean) if thickness_mean > 0 else 999.0

    # --- Missing count: angles where thickness is near-zero ---
    # If a long arc of the ring is missing, many angles will have very
    # small thickness values.
    if thickness_mean > 0:
        missing_threshold = 0.25 * thickness_mean
        missing_count = int(np.sum(thicknesses < missing_threshold))
    else:
        missing_count = n_angles
    missing_fraction = missing_count / n_angles

    # --- Max dip: maximum local drop compared to smoothed profile ---
    # We build a smoothed version of the thickness profile which acts as
    # the "expected" thickness at each angle.  The max_dip is the largest
    # amount by which the real thickness falls below this expected value.
    # This catches localised nicks/chips even when global stats look fine.
    smoothed = np.zeros(n_angles, dtype=np.float64)
    hw = DIP_SMOOTH_WINDOW
    for a in range(n_angles):
        indices = [(a + j) % n_angles for j in range(-hw, hw + 1)]
        smoothed[a] = np.mean(thicknesses[indices])

    dip_profile = np.maximum(0.0, smoothed - thicknesses)
    max_dip = float(dip_profile.max())

    # --- Holes ratio ---
    # Use the median inner/outer edge of the ring so that we know 
    # where the ring material should be.  Then count how many pixels
    # inside this area are background instead of ring.  This ignores
    # the central hole of the ring and focuses on holes in the material.
    valid_rays = thicknesses > 0
    if valid_rays.sum() > 0:
        med_r_inner = float(np.median(r_inners[valid_rays]))
        med_r_outer = float(np.median(r_outers[valid_rays]))

        rr, cc = np.ogrid[:h, :w]
        dist_map = np.sqrt((rr - centroid_r) ** 2 + (cc - centroid_c) ** 2)
        ring_edge = (dist_map >= med_r_inner) & (dist_map <= med_r_outer)
        ring_edge_area = int(ring_edge.sum())

        if ring_edge_area > 0:
            holes_in_ring_edge = int((ring_edge & ~comp_mask).sum())
            holes_ratio = holes_in_ring_edge / ring_edge_area
        else:
            holes_ratio = 0.0
    else:
        holes_ratio = 0.0

    # --- Extra blobs ---
    # extra_blobs_pre: blobs found before morphology (so small debris is
    # not erased).  Here we also count blobs after morphology and take
    # the maximum of both counts.
    extra_blobs_post = 0
    for comp in all_components:
        if comp['label'] != lbl and comp['area'] >= MIN_BLOB_AREA:
            extra_blobs_post += 1

    extra_blobs = max(extra_blobs_pre, extra_blobs_post)

    return {
        'area': area,
        'centroid_r': centroid_r,
        'centroid_c': centroid_c,
        'bbox': bbox,
        'thickness_mean': thickness_mean,
        'thickness_std': thickness_std,
        'thickness_cv': thickness_cv,
        'missing_count': missing_count,
        'missing_fraction': missing_fraction,
        'max_dip': max_dip,
        'holes_ratio': holes_ratio,
        'extra_blobs': extra_blobs,
    }


# 6. Classification

def classify(features: dict):
    """Decide whether the O-ring is PASS or FAIL.

    We use a few simple rules on the features (thickness variation,
    missing sections, deep nicks, holes and extra blobs).  Each rule
    that fails adds a fail reason string.

    Returns
    -------
    result  : str  ("PASS" or "FAIL")
    reasons : list[str]  (empty if PASS)
    """
    reasons = []

    if features['thickness_cv'] > THRESH_THICKNESS_CV:
        reasons.append(
            f"thickness_cv={features['thickness_cv']:.3f} > {THRESH_THICKNESS_CV}")

    if features['missing_fraction'] > THRESH_MISSING_FRAC:
        reasons.append(
            f"missing_fraction={features['missing_fraction']:.3f} > {THRESH_MISSING_FRAC}")

    if features['max_dip'] > THRESH_MAX_DIP:
        reasons.append(
            f"max_dip={features['max_dip']:.1f} > {THRESH_MAX_DIP}")

    if features['holes_ratio'] > THRESH_HOLES_RATIO:
        reasons.append(
            f"holes_ratio={features['holes_ratio']:.3f} > {THRESH_HOLES_RATIO}")

    if features['extra_blobs'] > 0:
        reasons.append(
            f"extra_blobs={features['extra_blobs']}")

    result = "FAIL" if reasons else "PASS"
    return result, reasons


# 7. Full inspection pipeline

def inspect_image(path: str) -> dict:
    """Run the full O-ring inspection pipeline on a single image.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    dict
        Dictionary with all metrics, the PASS/FAIL result, the chosen
        threshold, and the original grayscale image.
    """
    # --- Load ---
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = img.astype(np.uint8)

    # --- Otsu threshold ---
    t = otsu_threshold(img)

    # --- Binarize ---
    binary = binarize(img, t)

    # --- Blob detection on raw binary (before morphological cleaning) ---
    # Opening can erase small debris pieces, so we run connected
    # components here first to count them.
    labels_raw, comps_raw = connected_components(binary)
    if comps_raw:
        largest_raw = max(comps_raw, key=lambda c: c['area'])
        extra_blobs_pre = sum(
            1 for c in comps_raw
            if c['label'] != largest_raw['label'] and c['area'] >= MIN_BLOB_AREA
        )
    else:
        extra_blobs_pre = 0

    # --- Morphology (clean up the binary mask) ---
    se = np.ones((3, 3), dtype=bool)

    # Opening removes small noise specks
    cleaned = opening(binary, se)

    # Closing fills small gaps in the ring boundary
    cleaned = closing(cleaned, se)

    # Fill interior holes (used for the holes_ratio feature)
    filled = fill_holes(cleaned)

    # --- Connected component labeling on cleaned mask ---
    labels, components = connected_components(cleaned)

    if not components:
        return {
            'img': img,
            'threshold': t,
            'area': 0,
            'centroid_r': 0, 'centroid_c': 0,
            'bbox': (0, 0, 0, 0),
            'thickness_mean': 0, 'thickness_std': 0, 'thickness_cv': 999,
            'missing_count': 360, 'missing_fraction': 1.0,
            'max_dip': 0,
            'holes_ratio': 0,
            'extra_blobs': extra_blobs_pre,
            'result': 'FAIL',
            'reasons': ['No foreground detected'],
        }

    # Select the largest component as the O-ring
    largest = max(components, key=lambda c: c['area'])

    # --- Feature extraction ---
    features = extract_features(
        cleaned, largest, filled, labels, components, extra_blobs_pre)

    # --- Classification ---
    result, reasons = classify(features)

    return {
        'img': img,
        'threshold': t,
        **features,
        'result': result,
        'reasons': reasons,
    }
