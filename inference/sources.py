"""Frame producers.

The FrameSource seam is the pivot of the whole refactor: stills come from a
directory walk, video frames come from a PyAV decoder, and the models cannot tell
the difference. A live edge stream would just be a third implementation.

NOTE: ImageDirectorySource is not wired into `manage.py ingest` yet. Ingest
registers Image rows first and then batches over the pending ones; converting it
to consume frames is Step 2 of the refactor plan, not this step. The protocol is
declared now so video/decode.py has something to conform to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from inference.types import Frame

# Only try to decode formats the model + PIL reliably handle.
IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def load_image_array(path):
    """Explicitly read one image to an RGB array, returning (array, width, height)."""
    with PILImage.open(path) as img:
        return np.array(img.convert("RGB")), img.width, img.height


class ImageDirectorySource:
    """FrameSource over a directory tree of still images.

    Stills have no timeline, so ``timestamp`` and ``frame_index`` are just the
    position in the sorted walk, and ``media_id`` is the resolved file path
    (core.Media ids arrive in Step 3).
    """

    def __init__(self, folder, suffixes=IMAGE_SUFFIXES):
        self.folder = Path(folder)
        self.suffixes = suffixes

    def paths(self):
        """Every image file under the folder, in a stable order."""
        return sorted(
            p for p in self.folder.rglob("*") if p.suffix.lower() in self.suffixes
        )

    def __iter__(self):
        for index, path in enumerate(self.paths()):
            array, _width, _height = load_image_array(path)
            yield Frame(
                array=array,
                timestamp=float(index),
                frame_index=index,
                media_id=str(path.resolve()),
            )
