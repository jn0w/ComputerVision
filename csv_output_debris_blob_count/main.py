

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

    # Print CSV header
    header = ("filename,threshold,area,thickness_mean,thickness_std,"
              "thickness_cv,missing_frac,max_dip,holes_ratio,blobs,result,ms")
    print(header)

    # Go through each image
    for path in paths:
        filename = os.path.basename(path)

        # Inspect the image and measure how long it takes
        t0 = time.perf_counter()
        result = inspect_image(path)
        dt = (time.perf_counter() - t0) * 1000

        # Print the results as a CSV line
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
                f"{dt:.1f}")
        print(line)

        # Save annotated image with result text 
        out = cv2.cvtColor(result['img'], cv2.COLOR_GRAY2BGR)
        colour = (0, 200, 0) if result['result'] == "PASS" else (0, 0, 220)
        cv2.putText(out, result['result'], (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
        bbox = result.get('bbox')
        if bbox and bbox != (0, 0, 0, 0):
            rmin, rmax, cmin, cmax = bbox
            cv2.rectangle(out, (cmin, rmin), (cmax, rmax), (255, 255, 0), 1)
        cv2.imwrite(os.path.join(out_dir, filename), out)

    print(f"\nAnnotated images saved to '{out_dir}/'")


if __name__ == "__main__":
    main()

