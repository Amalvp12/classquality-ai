"""Technical video measurements. These are NOT teaching-quality scores.

Brightness and sharpness cannot show teacher visibility, background suitability, audio
clarity or teaching effectiveness. The app reports them separately for context only.
"""
from __future__ import annotations

import os

ALLOWED_EXTENSIONS = (".mp4", ".mov", ".avi")


class VideoError(RuntimeError):
    pass


def analyze_video(path: str, max_frames: int = 30) -> dict:
    """Sample up to `max_frames` evenly spaced frames and measure them.

    Returns duration (s), frames sampled, mean brightness (0-255 grey level) and sharpness
    (variance of the Laplacian; higher = more edge detail, resolution- and content-dependent).
    Raises VideoError for missing, unreadable or empty videos.
    """
    if not path or not os.path.exists(path):
        raise VideoError("Video file not found.")
    if not path.lower().endswith(ALLOWED_EXTENSIONS):
        raise VideoError("Unsupported video type. Use MP4, MOV or AVI.")
    if os.path.getsize(path) == 0:
        raise VideoError("The video file is empty.")
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise VideoError("opencv-python and numpy are required for video checks.") from exc

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoError("The video could not be opened (corrupt or unsupported codec).")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if total <= 0 or fps <= 0:
            raise VideoError("The video reports no frames or an invalid frame rate.")
        duration = total / fps
        n = max(1, min(max_frames, total))
        indices = sorted({int(i * (total - 1) / max(1, n - 1)) for i in range(n)}) if n > 1 else [0]
        brightness: list[float] = []
        sharpness: list[float] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(gray.mean()))
            sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if not brightness:
            raise VideoError("No frames could be decoded from the video.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    return {
        "duration_seconds": round(duration, 1),
        "frames_sampled": len(brightness),
        "resolution": f"{width}x{height}" if width and height else "unknown",
        "mean_brightness": round(float(np.mean(brightness)), 1),
        "brightness_range": [round(min(brightness), 1), round(max(brightness), 1)],
        "mean_sharpness": round(float(np.mean(sharpness)), 1),
        "min_sharpness": round(min(sharpness), 1),
        "note": ("Frame-sampling measurements only. They do not show teacher visibility, background, "
                 "audio quality or teaching quality."),
    }
