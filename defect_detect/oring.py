import numpy as np
import cv2
import math


# Configurable classification thresholds
THRESH_THICKNESS_CV = 0.35    # coefficient of variation of radial thickness
THRESH_MISSING_FRAC = 0.03    # fraction of rays with near zero thickness
THRESH_HOLES_RATIO = 0.02     # ratio of holes inside ring annulus
MIN_BLOB_AREA = 50            # min pixel area for extra blob to count


# 1. Histogram based automatic threshold Otsus method

def otsu_threshold(img: np.ndarray) -> int:
    """Compute the optimal binarization threshold using Otsu's method."""
    # Step 1 Count how many pixels have each brightness level from 0 to 255
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total_pixels = img.size

    # Step 2 Pre calculate running totals to speed things up
    # cum_sum how many pixels are at or below each brightness level
    # cum_mean total brightness of all pixels at or below each level
    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))
    global_mean = cum_mean[-1]  # total brightness of the whole image

    # Step 3 Try every possible threshold value and pick the best one
    best_t = 0
    best_var = -1.0

    for t in range(256):
        # Split pixels into two groups dark ones w0 and bright ones w1
        w0 = cum_sum[t]  # how many pixels are dark at or below threshold
        w1 = total_pixels - w0  # how many pixels are bright above threshold
        
        # Skip if one group is empty cant use this threshold
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
    """Threshold image to binary, auto-detecting foreground polarity."""
    border_width = 10
    h, w = img.shape

    # Look at the edges of the image top bottom left right
    # The edges are usually background so we can use them to figure out
    # whether the O ring is darker or brighter than the background
    border_pixels = np.concatenate([
        img[:border_width, :].ravel(),  # top edge
        img[-border_width:, :].ravel(),  # bottom edge
        img[border_width:-border_width, :border_width].ravel(),  # left edge
        img[border_width:-border_width, -border_width:].ravel(),  # right edge
    ])

    # Compare edge brightness with overall image brightness
    border_mean = border_pixels.mean()
    overall_mean = img.mean()

    # Figure out if O ring is dark or bright
    # If edges are bright O ring is probably dark use pixels below threshold
    # If edges are dark O ring is probably bright use pixels above threshold
    if border_mean >= overall_mean:
        # Dark O ring on light background
        binary = img <= threshold
    else:
        # Light O ring on dark background
        binary = img >= threshold

    return binary


# 3. Binary morphology from scratch

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


def closing(mask, se):
    """Morphological closing dilate then erode.  Closes small holes gaps."""
    return erode(dilate(mask, se), se)


def opening(mask, se):
    """Morphological opening erode then dilate.  Removes small noise."""
    return dilate(erode(mask, se), se)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask using border flood-fill.
    
    Algorithm:
      1. Start from every background pixel on the image border.
      2. BFS flood fill 4 connected on the background.
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

    # Spread out from edges marking all background pixels we can reach
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and not mask[nr, nc]:
                visited[nr, nc] = True
                stack.append((nr, nc))

    # Any pixel we couldnt reach from the edges is either the O ring or a hole inside it
    # Fill in those holes by making them part of the O ring
    filled = mask | ~visited
    return filled


# 4. Connected component labeling 8 connected stack based BFS

def connected_components(mask: np.ndarray):
    """Label 8-connected components in a boolean mask."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    components = []

    # Check all 8 neighbors including diagonals
    neighbours_8 = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

    # Go through each pixel
    for r in range(h):
        for c in range(w):
            # Found a new foreground pixel that hasnt been labeled yet
            if mask[r, c] and labels[r, c] == 0:
                current_label += 1
                stack = [(r, c)]
                labels[r, c] = current_label
                area = 0
                rmin, rmax, cmin, cmax = r, r, c, c

                # Use BFS to find all connected pixels
                while stack:
                    cr, cc = stack.pop()
                    area += 1
                    # Track bounding box
                    if cr < rmin: rmin = cr
                    if cr > rmax: rmax = cr
                    if cc < cmin: cmin = cc
                    if cc > cmax: cmax = cc

                    # Check all 8 neighbors
                    for dr, dc in neighbours_8:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = current_label
                            stack.append((nr, nc))

                # Save info about this component
                components.append({
                    'label': current_label,
                    'area': area,
                    'bbox': (rmin, rmax, cmin, cmax),
                })

    return labels, components


# 5. Feature extraction

def extract_features(mask, component, filled_mask, labels, all_components):
    """Compute defect-relevant features for the largest O-ring component."""
    lbl = component['label']
    comp_mask = labels == lbl

    # Basic area measurement
    area = int(comp_mask.sum())

    # Find the center of the O ring
    rows, cols = np.where(comp_mask)
    centroid_r = float(rows.mean())
    centroid_c = float(cols.mean())

    bbox = component['bbox']

    # Measure thickness by shooting rays from the center in all directions
    h, w = mask.shape
    max_radius = int(math.sqrt(h ** 2 + w ** 2))
    n_angles = 360
    thicknesses = np.zeros(n_angles, dtype=np.float64)
    r_inners = np.zeros(n_angles, dtype=np.float64)
    r_outers = np.zeros(n_angles, dtype=np.float64)

    # For each angle find inner and outer radius of the O ring
    for i in range(n_angles):
        angle_rad = math.radians(i)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        r_inner = -1
        r_outer = -1

        # Walk outward from center along this ray
        for radius in range(1, max_radius):
            pr = int(round(centroid_r + radius * cos_a))
            pc = int(round(centroid_c + radius * sin_a))
            if pr < 0 or pr >= h or pc < 0 or pc >= w:
                break
            if comp_mask[pr, pc]:
                if r_inner < 0:
                    r_inner = radius  # first time we hit the O ring
                r_outer = radius  # keep updating outer edge

        # Calculate thickness for this angle
        if r_inner > 0 and r_outer > 0:
            thicknesses[i] = r_outer - r_inner
            r_inners[i] = r_inner
            r_outers[i] = r_outer

    # Calculate statistics about thickness variation
    thickness_mean = float(thicknesses.mean()) if thicknesses.sum() > 0 else 0.0
    thickness_std = float(thicknesses.std()) if thicknesses.sum() > 0 else 0.0
    thickness_cv = (thickness_std / thickness_mean) if thickness_mean > 0 else 999.0

    # Count how many rays show missing material very thin sections
    if thickness_mean > 0:
        missing_threshold = 0.25 * thickness_mean
        missing_count = int(np.sum(thicknesses < missing_threshold))
    else:
        missing_count = n_angles
    missing_fraction = missing_count / n_angles

    # Find holes inside the O ring by looking at the annulus region
    valid_rays = thicknesses > 0
    if valid_rays.sum() > 0:
        med_r_inner = float(np.median(r_inners[valid_rays]))
        med_r_outer = float(np.median(r_outers[valid_rays]))
        rr, cc = np.ogrid[:h, :w]
        dist_map = np.sqrt((rr - centroid_r) ** 2 + (cc - centroid_c) ** 2)
        annulus = (dist_map >= med_r_inner) & (dist_map <= med_r_outer)
        annulus_area = int(annulus.sum())
        if annulus_area > 0:
            holes_in_annulus = int((annulus & ~comp_mask).sum())
            holes_ratio = holes_in_annulus / annulus_area
        else:
            holes_ratio = 0.0
    else:
        holes_ratio = 0.0

    # Count extra blobs other components besides the main O ring
    extra_blobs = 0
    for comp in all_components:
        if comp['label'] != lbl and comp['area'] >= MIN_BLOB_AREA:
            extra_blobs += 1

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
        'holes_ratio': holes_ratio,
        'extra_blobs': extra_blobs,
    }


# 6. Classification

def classify(features):
    """Classify O-ring as PASS or FAIL based on extracted features."""
    reasons = []

    # Check if thickness varies too much
    if features['thickness_cv'] > THRESH_THICKNESS_CV:
        reasons.append(
            f"thickness_cv={features['thickness_cv']:.3f} > {THRESH_THICKNESS_CV}")

    # Check if too much material is missing
    if features['missing_fraction'] > THRESH_MISSING_FRAC:
        reasons.append(
            f"missing_fraction={features['missing_fraction']:.3f} > {THRESH_MISSING_FRAC}")

    # Check if there are too many holes
    if features['holes_ratio'] > THRESH_HOLES_RATIO:
        reasons.append(
            f"holes_ratio={features['holes_ratio']:.3f} > {THRESH_HOLES_RATIO}")

    # Check if there are extra blobs contamination
    if features['extra_blobs'] > 0:
        reasons.append(f"extra_blobs={features['extra_blobs']}")

    result = "FAIL" if reasons else "PASS"
    return result, reasons


# 7. Full inspection pipeline

def inspect_image(path: str) -> dict:
    """Run the full O-ring inspection pipeline on a single image."""
    # Load the image convert to black and white
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = img.astype(np.uint8)

    # Step 1 Find the best threshold value to separate O ring from background
    t = otsu_threshold(img)
    
    # Step 2 Turn the image into black and white O ring is white background is black
    binary = binarize(img, t)

    # Step 3 Clean up the image
    se = np.ones((3, 3), dtype=bool)
    cleaned = opening(binary, se)   # Remove tiny noise spots
    cleaned = closing(cleaned, se)  # Fill in small gaps
    filled = fill_holes(cleaned)   # Fill any holes inside the O ring

    # Step 4 Find all connected components
    labels, components = connected_components(cleaned)

    # If no components found return failure
    if not components:
        return {
            'img': img, 'threshold': t,
            'area': 0, 'centroid_r': 0, 'centroid_c': 0,
            'bbox': (0, 0, 0, 0),
            'thickness_mean': 0, 'thickness_std': 0, 'thickness_cv': 999,
            'missing_count': 360, 'missing_fraction': 1.0,
            'holes_ratio': 0, 'extra_blobs': 0,
            'result': 'FAIL', 'reasons': ['No foreground detected'],
        }

    # Step 5 Extract features from the largest component the O ring
    largest = max(components, key=lambda c: c['area'])
    features = extract_features(cleaned, largest, filled, labels, components)

    # Step 6 Classify as PASS or FAIL
    result, reasons = classify(features)

    return {
        'img': img, 'threshold': t,
        **features,
        'result': result, 'reasons': reasons,
    }

