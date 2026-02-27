"""
Utility functions used by the O-ring inspection code.

This file contains the annotate_image function, which draws the
inspection results on top of an image using OpenCV.
"""

import numpy as np
import cv2
import math


# ===================================================================
# Annotation cv2 drawing
# ===================================================================

def annotate_image(img_gray: np.ndarray, filename: str, result: str,
                   time_ms: float, features: dict) -> np.ndarray:
    """Create a colour image with the inspection results drawn on top.

    The function draws the file name, a PASS or FAIL label, the processing
    time in milliseconds, the bounding box of the detected O-ring and a
    small marker at the ring centroid.

    Parameters
    ----------
    img_gray : np.ndarray  (H, W) uint8 grayscale input image
    filename : str         name of the file (for display)
    result   : str         "PASS" or "FAIL"
    time_ms  : float       processing time in milliseconds
    features : dict        feature/result dictionary from inspect_image()

    Returns
    -------
    np.ndarray (H, W, 3) uint8 BGR annotated image
    """
    # Convert grayscale to BGR so that we can draw in colour
    out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    h, w = out.shape[:2]
    # Pick font size based on image size so text stays readable
    font_scale = max(0.9, min(h, w) / 400.0)
    thickness = max(1, int(font_scale * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Colours
    green = (0, 200, 0)
    red = (0, 0, 220)
    white = (255, 255, 255)
    cyan = (255, 255, 0)

    colour = green if result == "PASS" else red

    # Filename
    y_offset = int(25 * font_scale)
    cv2.putText(out, filename, (5, y_offset), font, font_scale * 0.6,
                white, thickness, cv2.LINE_AA)

    # PASS / FAIL
    y_offset += int(30 * font_scale)
    cv2.putText(out, result, (5, y_offset), font, font_scale * 1.2,
                colour, thickness + 1, cv2.LINE_AA)

    # Processing time
    y_offset += int(25 * font_scale)
    cv2.putText(out, f"t = {time_ms:.1f} ms", (5, y_offset), font,
                font_scale * 0.55, white, thickness, cv2.LINE_AA)

    # Bounding box
    bbox = features.get('bbox', None)
    if bbox and bbox != (0, 0, 0, 0):
        rmin, rmax, cmin, cmax = bbox
        cv2.rectangle(out, (cmin, rmin), (cmax, rmax), cyan, 1)

    # Centroid
    cr = features.get('centroid_r', None)
    cc = features.get('centroid_c', None)
    if cr is not None and cc is not None:
        cv2.circle(out, (int(cc), int(cr)), 4, cyan, -1)

    # Failure reasons (small text at the bottom of the image)
    reasons = features.get('reasons', [])
    if reasons:
        black = (0, 0, 0)
        for i, reason in enumerate(reasons):
            y_pos = h - 10 - i * int(18 * font_scale)
            if y_pos < 0:
                break
            cv2.putText(out, reason, (5, y_pos), font, font_scale * 0.5,
                        black, 1, cv2.LINE_AA)

    return out

