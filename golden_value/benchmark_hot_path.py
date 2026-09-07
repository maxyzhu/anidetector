"""
Run:  uv run python golden_value/benchmark_hot_path.py <path-to-video>

Times the decode -> sample -> detect loop, decode separately from detection, so
the answer to "what is the throughput ceiling" is measured rather than assumed.
It was assumed once: the roadmap called the Python loop the ceiling, and on the
pet set decode runs ~60x realtime while the detector forward pass is ~97% of the
loop, which caps a loop rewrite at a few percent.

No database and no Django: decode_frames only carries media_id into the Frame,
so a path and a dummy id are enough, and the number is reproducible anywhere the
file is.

  --write <file>   record the run as a baseline
  --check <file>   re-run and compare against one, non-zero exit if it regressed
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

# Running a script puts golden_value/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import get_detector
from video.decode import decode_frames, probe
from video.sampling import sample_at_fps

# Fraction the throughput may fall before --check fails. Wide because this is a
# wall-clock number on a shared machine, not a unit test.
TOLERANCE = 0.20


def _batches(frames, size):
    batch = []
    for frame in frames:
        batch.append(frame)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _sampled(path, media_id, fps):
    return sample_at_fps(decode_frames(path, media_id), fps)


def measure(path, device, fps, batch, conf, warmup):
    duration = probe(path).duration

    start = time.perf_counter()
    detector = get_detector(device=device)
    load_seconds = time.perf_counter() - start

    # Lazy init happens on the first inference; folding it into the loop would
    # understate throughput by a whole model warm-up.
    for index, frame in enumerate(_sampled(path, 0, fps)):
        if index >= warmup:
            break
        detector.detect_batch([frame.array], conf)

    start = time.perf_counter()
    decoded = sum(1 for _ in _sampled(path, 0, fps))
    decode_seconds = time.perf_counter() - start

    start = time.perf_counter()
    detected = 0
    for group in _batches(_sampled(path, 0, fps), batch):
        detector.detect_batch([frame.array for frame in group], conf)
        detected += len(group)
    total_seconds = time.perf_counter() - start

    detect_seconds = total_seconds - decode_seconds
    return {
        "video": Path(path).name,
        "video_seconds": duration,
        "frames": detected,
        "device": device,
        "sample_fps": fps,
        "batch": batch,
        "detector_load_seconds": load_seconds,
        "decode_seconds": decode_seconds,
        "detect_seconds": detect_seconds,
        "total_seconds": total_seconds,
        "decode_fps": decoded / decode_seconds if decode_seconds else 0.0,
        "throughput_fps": detected / total_seconds if total_seconds else 0.0,
        "realtime_factor": duration / total_seconds if total_seconds else 0.0,
        "detect_ms_per_frame": detect_seconds / detected * 1000 if detected else 0.0,
        "detect_share": detect_seconds / total_seconds if total_seconds else 0.0,
        "machine": f"{platform.system()} {platform.machine()}",
    }


def report(result):
    print(f"\n{result['video']}  {result['video_seconds']:.1f}s of footage"
          f"  ·  {result['device']}  ·  {result['sample_fps']} fps sampling"
          f"  ·  batch {result['batch']}")
    print(f"  detector load        {result['detector_load_seconds']:8.1f}s   (one-off, outside the loop)")
    print(f"  decode + sample      {result['decode_seconds']:8.1f}s"
          f"   {result['decode_fps']:7.1f} fps")
    print(f"  + detect             {result['total_seconds']:8.1f}s"
          f"   {result['throughput_fps']:7.1f} fps"
          f"   {result['realtime_factor']:5.2f}x realtime")
    print(f"\n  detect is {result['detect_share']*100:.0f}% of the loop"
          f"  ·  {result['detect_ms_per_frame']:.0f} ms/frame")
    hours = 12 * 3600 / result["realtime_factor"] / 3600 if result["realtime_factor"] else 0
    print(f"  a 12 h night takes {hours:.1f} h\n")


def check(result, baseline_path):
    baseline = json.loads(Path(baseline_path).read_text())
    if baseline["machine"] != result["machine"] or baseline["device"] != result["device"]:
        print(f"baseline is from {baseline['machine']}/{baseline['device']}, "
              f"this is {result['machine']}/{result['device']} — not comparable")
        return 0

    failed = False
    for key in ("throughput_fps", "decode_fps"):
        floor = baseline[key] * (1 - TOLERANCE)
        ok = result[key] >= floor
        failed |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {key}: {result[key]:.1f} "
              f"vs baseline {baseline[key]:.1f} (floor {floor:.1f})")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--fps", type=float, default=5.0, help="sampling rate")
    parser.add_argument("--batch", type=int, default=1,
                        help="frames per detect_batch call; services.py uses 1")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--write", metavar="FILE", help="record this run as a baseline")
    parser.add_argument("--check", metavar="FILE", help="compare against a baseline")
    opts = parser.parse_args()

    result = measure(opts.video, opts.device, opts.fps, opts.batch,
                     opts.conf, opts.warmup)
    report(result)

    if opts.write:
        Path(opts.write).write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote baseline to {opts.write}")
    if opts.check:
        sys.exit(check(result, opts.check))


if __name__ == "__main__":
    main()
