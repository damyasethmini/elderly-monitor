"""Sample still frames from a video and save a timestamp manifest.

This module is deliberately independent of the vision models. Later pipeline
stages can use the saved images and timestamps for person/pose detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm (for readable logs and manifests)."""
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def sample_video(
    video_path: Path,
    frames_dir: Path,
    sampling_fps: float = 1.0,
    max_frames: int | None = None,
    jpeg_quality: int = 92,
) -> dict[str, Any]:
    """Save approximately ``sampling_fps`` images per second of source video.

    Timestamps are estimated from the source frame index and reported source
    FPS. This is appropriate for ordinary constant-frame-rate MP4 test videos.
    The manifest's ``timestamp_sec`` values remain numeric for later duration
    calculations.
    """
    video_path = Path(video_path)
    frames_dir = Path(frames_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if sampling_fps <= 0:
        raise ValueError("sampling_fps must be greater than 0")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than 0 when provided")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {video_path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if source_fps <= 0:
            raise ValueError(
                "Could not read a valid FPS value from this video. "
                "Try converting it to a standard MP4 video first."
            )

        # Sampling faster than the source cannot create new information, so
        # cap the effective rate at the source FPS.
        effective_sampling_fps = min(float(sampling_fps), source_fps)
        frame_interval = source_fps / effective_sampling_fps

        frames_dir.mkdir(parents=True, exist_ok=True)
        # Remove frames from a previous run so a lower --max-frames value does
        # not leave stale JPEGs beside the new manifest.
        for old_frame in frames_dir.glob("frame_*.jpg"):
            old_frame.unlink()

        samples: list[dict[str, Any]] = []
        frame_index = 0
        next_sample_frame = 0.0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index + 1e-9 >= next_sample_frame:
                timestamp_sec = frame_index / source_fps
                sample_number = len(samples) + 1
                filename = (
                    f"frame_{sample_number:06d}_t{timestamp_sec:09.3f}s.jpg"
                )
                image_path = frames_dir / filename
                written = cv2.imwrite(
                    str(image_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not written:
                    raise OSError(f"Could not write sampled frame: {image_path}")

                samples.append(
                    {
                        "sample_number": sample_number,
                        "source_frame_index": frame_index,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "timestamp": format_timestamp(timestamp_sec),
                        "file": f"frames/{filename}",
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    }
                )
                next_sample_frame += frame_interval

                if max_frames is not None and len(samples) >= max_frames:
                    break

            frame_index += 1

        actual_frame_count = frame_index + (1 if max_frames is not None and len(samples) >= max_frames else 0)
        # OpenCV's reported count is usually reliable. Keep both counts in the
        # manifest so videos with imperfect metadata are easier to diagnose.
        duration_sec = (
            reported_frame_count / source_fps
            if reported_frame_count > 0
            else actual_frame_count / source_fps
        )

        return {
            "source_video": video_path.name,
            "source_video_path": str(video_path.resolve()),
            "width": width,
            "height": height,
            "source_fps": round(source_fps, 3),
            "reported_frame_count": reported_frame_count,
            "duration_sec": round(duration_sec, 3),
            "sampling_fps_requested": round(float(sampling_fps), 3),
            "sampling_fps_effective": round(effective_sampling_fps, 3),
            "sampled_frame_count": len(samples),
            "max_frames_limit": max_frames,
            "frames_directory": str(frames_dir.resolve()),
            "frames": samples,
        }
    finally:
        capture.release()


def write_manifest(manifest: dict[str, Any], manifest_path: Path) -> Path:
    """Write the frame manifest as a UTF-8 JSON file."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return manifest_path
