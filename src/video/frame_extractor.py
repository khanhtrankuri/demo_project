from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ExtractedFrame:
    frame_id: int
    timestamp_sec: float
    image_path: str


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float = 2.0,
) -> list[ExtractedFrame]:
    """Decode a video and sample frames at approximately sample_fps.

    OpenCV is imported lazily so the rest of the package remains usable without
    video dependencies until this feature is invoked.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video frame extraction requires opencv-python") from exc

    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = 30.0

    interval = max(1, round(native_fps / sample_fps))
    results: list[ExtractedFrame] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                timestamp_sec = frame_index / native_fps
                target = output_dir / f"frame_{frame_index:08d}.jpg"
                if not cv2.imwrite(str(target), frame):
                    raise RuntimeError(f"Could not write sampled frame: {target}")
                results.append(
                    ExtractedFrame(
                        frame_id=frame_index,
                        timestamp_sec=timestamp_sec,
                        image_path=str(target.resolve()),
                    )
                )
            frame_index += 1
    finally:
        capture.release()

    if not results:
        raise RuntimeError("No frames were extracted from the uploaded video")
    return results
