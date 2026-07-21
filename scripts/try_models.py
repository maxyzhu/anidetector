"""
Run:  uv run python scripts/try_model.py <path-to-image>

Goal: confirm the model loads, runs on one image, and see the exact shape of
what single_image_detection returns (bbox / category / confidence), so that a
later failure is unambiguously a *Django* problem, not a *model* problem.
"""

import sys

import numpy as np
from PIL import Image

def main(image_path: str):
    from PytorchWildlife.models import detection as pw_detection

    pil = Image.open(image_path).convert("RGB")
    img_arr = np.array(pil)
    print(f"loaded {image_path} size={pil.size} arr={img_arr.shape}")

    detector = pw_detection.MegaDetectorV6(
        device="cpu", pretrained=True, version="MDV6-yolov9-c"
    )

    results = detector.single_image_detection(img_arr)

    # Print the raw result keys and detections
    print("\n--- raw result keys ---")
    print(type(results), list(results.keys()) if hasattr(results, "keys") else results)
    print("\n--- detections ---")
    print(results)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/try_model.py scripts/01.webp or <your-image-path>")
        sys.exit(1)
    main(sys.argv[1])