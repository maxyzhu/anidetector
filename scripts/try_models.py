"""
Run:  uv run python scripts/try_models.py <path-to-image>

Goal: confirm the detector loads and runs *through the inference package*, and
print both the raw PytorchWildlife result and what Detector._parse_one made of
it — so a later failure is unambiguously a *Django* problem, not a *model*
problem, and a change in the library's return shape shows up here first.
"""

import sys
from pathlib import Path

# Running a script puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import get_detector, load_image_array

CONF = 0.5


def main(image_path: str):
    img_arr, width, height = load_image_array(image_path)
    print(f"loaded {image_path} size=({width}, {height}) arr={img_arr.shape}")

    detector = get_detector(device="cpu")
    print(f"variant={detector.version} licence={detector.licence}")

    raw = detector.raw_batch([img_arr], CONF)[0]
    print("\n--- raw result ---")
    print(type(raw))
    print(raw)

    parsed = detector.detect_batch([img_arr], conf_threshold=CONF)[0]
    print(f"\n--- parsed: {len(parsed)} detection(s) at conf >= {CONF} ---")
    for d in parsed:
        box = tuple(round(v, 4) for v in d.bbox)
        print(f"  {d.category:8} {d.confidence:0.3f}  bbox={box}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/try_models.py example_images/01.webp or <your-image-path>")
        sys.exit(1)
    main(sys.argv[1])
