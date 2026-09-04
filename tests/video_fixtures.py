"""Synthetic clips with known ground truth.

Field footage has no ground truth, so the tracker and the motion signals get
verified against a rectangle moving at a speed we chose.
"""

import av
import numpy as np


def write_moving_rectangle(path, frames=100, fps=25, gop=10, size=(160, 120),
                           box=(20, 20), x_per_frame=1.0):
    """Write a clip of one bright rectangle sliding left to right.

    Returns the per-frame top-left x positions, so a test can compare against
    the trajectory it asked for.
    """
    width, height = size
    box_w, box_h = box
    y = (height - box_h) // 2
    positions = []

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = gop

        for i in range(frames):
            x = int(i * x_per_frame)
            positions.append(x)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[y:y + box_h, x:min(x + box_w, width)] = 255
            frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():   # flush
            container.mux(packet)

    return positions