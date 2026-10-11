
import base64
import binascii
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Literal
from uuid import uuid4

import cv2
import mediapipe as mp
import numpy as np
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent
MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b")
API_KEY = os.getenv("OLLAMA_API_KEY", "")
logger = logging.getLogger("skipsense")
logger.setLevel(logging.INFO)
DEBUG_SAVE_FRAMES = os.getenv("SKIPSENSE_DEBUG_FRAMES", "").lower() in {
    "1",
    "true",
    "yes",
}
DEBUG_FRAME_DIR = Path(
    os.getenv("SKIPSENSE_DEBUG_DIR", str(ROOT / "debug_frames"))
).expanduser().resolve()

if DEBUG_SAVE_FRAMES:
    try:
        DEBUG_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Could not create debug frame directory %s", DEBUG_FRAME_DIR)

logger.info(
    "SkipSense debug frames: enabled=%s output_dir=%s",
    DEBUG_SAVE_FRAMES,
    DEBUG_FRAME_DIR,
)

app = FastAPI(title="SkipSense AI Jump Counter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class CountRequest(BaseModel):
    frames: List[str] = Field(min_length=8, max_length=2000)
    interval_ms: int = Field(default=100, ge=1, le=2000)
    duration_seconds: float | None = Field(default=None, gt=0, le=3600)
    requested_fps: Literal[15, 30, 60] = 15


class CoachRequest(BaseModel):
    completed_skips: int = Field(ge=0)
    duration_seconds: float = Field(gt=0, le=3600)
    rest_seconds: float = Field(default=0, ge=0, le=3600)


class LiveFrameRequest(BaseModel):
    session_id: str = Field(min_length=16, max_length=64)
    frame: str = Field(min_length=32)
    elapsed_seconds: float = Field(ge=0, le=3600)


class LiveStopRequest(BaseModel):
    session_id: str = Field(min_length=16, max_length=64)


def detect_jump_cycles(signal: np.ndarray, interval_ms: int) -> tuple[np.ndarray, dict]:
    """Return separated upward motion peaks and basic signal quality metrics."""
    finite = np.isfinite(signal)
    if finite.sum() < 3:
        return np.array([], dtype=int), {"valid_fraction": 0.0}

    indexes = np.arange(len(signal))
    interpolated = np.interp(indexes, indexes[finite], signal[finite])
    if len(interpolated) >= 5:
        kernel = np.ones(5, dtype=float) / 5
        smoothed = np.convolve(np.pad(interpolated, (2, 2), mode="edge"), kernel, mode="valid")
    else:
        smoothed = interpolated

    motion = -smoothed
    amplitude = float(np.percentile(motion, 95) - np.percentile(motion, 5))
    signal_min = float(np.min(interpolated))
    signal_max = float(np.max(interpolated))
    # A relative threshold handles different camera distances and resolutions.
    prominence = max(0.008, amplitude * 0.18)
    min_distance = max(2, round(280 / interval_ms))
    peaks, properties = find_peaks(
        motion,
        distance=min_distance,
        prominence=prominence,
    )

    gaps = np.diff(peaks) * interval_ms / 1000 if len(peaks) > 1 else np.array([])
    regularity = 1.0
    if len(gaps) > 1 and np.mean(gaps) > 0:
        regularity = max(0.0, 1.0 - float(np.std(gaps) / np.mean(gaps)))
    quality = {
        "valid_fraction": round(float(finite.mean()), 3),
        "signal_min": round(signal_min, 4),
        "signal_max": round(signal_max, 4),
        "signal_range": round(signal_max - signal_min, 4),
        "signal_definition": "(ankle_y_or_hip_y - shoulder_y) / shoulder_width",
        "signal_direction": "lower values are upward; detector negates signal",
        "motion_amplitude": round(amplitude, 4),
        "cycle_regularity": round(regularity, 3),
        "median_peak_prominence": round(float(np.median(properties["prominences"])), 4)
        if len(peaks) else 0.0,
        "peak_timestamps_seconds": [
            round(float(index * interval_ms / 1000), 3) for index in peaks
        ],
        "peak_prominences": [
            round(float(prominence), 4)
            for prominence in properties["prominences"]
        ],
    }
    return peaks, quality


def align_signal_sources(
    signal: np.ndarray, sources: np.ndarray
) -> tuple[np.ndarray, float]:
    """Align hip fallback values to the ankle signal's baseline."""
    aligned = signal.copy()
    ankle_values = aligned[(sources == "ankle") & np.isfinite(aligned)]
    hip_values = aligned[(sources == "hip") & np.isfinite(aligned)]
    if not len(ankle_values) or not len(hip_values):
        return aligned, 0.0

    offset = float(np.median(ankle_values) - np.median(hip_values))
    aligned[sources == "hip"] += offset
    return aligned, offset


class LiveSession:
    def __init__(self):
        self.pose_module = mp.solutions.pose
        self.pose = self.pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.signals = []
        self.sources = []
        self.timestamps = []
        self.lock = threading.Lock()

    def close(self):
        self.pose.close()

    def process(self, frame: str, elapsed_seconds: float) -> dict:
        with self.lock:
            image = decode_frame(frame)
            result = self.pose.process(image)
            landmarks = []
            if result.pose_landmarks:
                landmarks = [
                    {
                        "x": round(float(point.x), 4),
                        "y": round(float(point.y), 4),
                        "visibility": round(float(point.visibility), 3),
                    }
                    for point in result.pose_landmarks.landmark
                ]

            signal = np.nan
            source = None
            guidance = "Keep your full body, shoulders, and ankles visible."
            if result.pose_landmarks:
                points = result.pose_landmarks.landmark
                left = points[self.pose_module.PoseLandmark.LEFT_HIP]
                right = points[self.pose_module.PoseLandmark.RIGHT_HIP]
                left_shoulder = points[self.pose_module.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = points[self.pose_module.PoseLandmark.RIGHT_SHOULDER]
                left_ankle = points[self.pose_module.PoseLandmark.LEFT_ANKLE]
                right_ankle = points[self.pose_module.PoseLandmark.RIGHT_ANKLE]
                shoulders_visible = min(
                    left_shoulder.visibility,
                    right_shoulder.visibility,
                ) >= 0.35
                ankles_visible = min(
                    left_ankle.visibility,
                    right_ankle.visibility,
                ) >= 0.35
                hips_visible = min(left.visibility, right.visibility) >= 0.35
                body_scale = float(
                    np.hypot(
                        left_shoulder.x - right_shoulder.x,
                        left_shoulder.y - right_shoulder.y,
                    )
                )
                if shoulders_visible and (ankles_visible or hips_visible) and body_scale >= 0.03:
                    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                    if ankles_visible:
                        movement_y = (left_ankle.y + right_ankle.y) / 2
                        source = "ankle"
                    else:
                        movement_y = (left.y + right.y) / 2
                        source = "hip"
                    signal = (movement_y - shoulder_y) / body_scale
                    guidance = "Tracking body motion."
                else:
                    guidance = "Step back and keep both shoulders and ankles visible."
            else:
                guidance = "No pose detected. Face the camera and show your full body."

            self.signals.append(signal)
            self.sources.append(source)
            self.timestamps.append(elapsed_seconds)
            signal_array, _ = align_signal_sources(
                np.asarray(self.signals, dtype=float),
                np.asarray(self.sources, dtype=object),
            )
            finite_count = int(np.isfinite(signal_array).sum())
            interval_ms = 200.0
            if len(self.timestamps) > 1:
                interval_ms = max(
                    1.0,
                    float(np.median(np.diff(self.timestamps[-10:]))) * 1000,
                )
            peaks, quality = detect_jump_cycles(signal_array, interval_ms)
            count = int(len(peaks)) if finite_count >= 6 else 0
            elapsed = max(0.001, elapsed_seconds)
            return {
                "completed_skips": count,
                "elapsed_seconds": round(elapsed, 2),
                "estimated_skips_per_minute": round(count * 60 / elapsed, 1),
                "pose_available": bool(result.pose_landmarks),
                "landmarks": landmarks,
                "guidance": guidance,
                "valid_signal_samples": finite_count,
                "frames_analyzed": len(self.signals),
                "signal_used": "ankle_with_hip_fallback"
                if "ankle" in self.sources and "hip" in self.sources
                else (
                    "ankle"
                    if "ankle" in self.sources
                    else ("hip" if "hip" in self.sources else "none")
                ),
                "signal_quality": quality,
            }


LIVE_SESSIONS: dict[str, LiveSession] = {}
LIVE_SESSIONS_LOCK = threading.Lock()


def decode_frame(data_url: str) -> np.ndarray:
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    encoded = encoded.strip()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image frame.") from exc

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode an image frame.")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def count_jumps(
    frames: List[str], interval_ms: int, duration_seconds: float | None = None
) -> dict:
    """Estimate jump cycles from normalized vertical hip motion."""
    pose_module = mp.solutions.pose
    hip_positions = []
    signal_sources = []
    valid_frames = 0
    pose_detected_frames = 0
    visibility_rejected_frames = 0
    scale_rejected_frames = 0
    ankle_signal_frames = 0
    hip_fallback_frames = 0
    image_size = None
    debug_frame_files = []
    processing_exceptions = []
    debug_indexes = {0, len(frames) // 2, len(frames) - 1}

    diagnostics = {
        "decoded_frames": 0,
        "pose_detected_frames": 0,
        "pose_missing_frames": 0,
        "visibility_rejected_frames": 0,
        "scale_rejected_frames": 0,
        "ankle_signal_frames": 0,
        "hip_fallback_frames": 0,
        "image_size": None,
        "color_conversion": "OpenCV BGR decode converted to RGB",
        "input_prefix": None,
        "input_encoded_chars": None,
        "debug_frame_files": debug_frame_files,
        "processing_exceptions": processing_exceptions,
    }
    if DEBUG_SAVE_FRAMES:
        DEBUG_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Counting %d frames at %d ms intervals", len(frames), interval_ms)

    with pose_module.Pose(
        # Uploaded videos are sampled with seeks, so tracking state can be
        # stale between frames. Independent detection is more reliable here.
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    ) as pose:
        for frame_index, frame_data in enumerate(frames):
            if frame_index == 0 and isinstance(frame_data, str):
                prefix, separator, encoded = frame_data.partition(",")
                diagnostics["input_prefix"] = prefix[:80] if separator else "raw-base64"
                diagnostics["input_encoded_chars"] = len(encoded if separator else frame_data)
                logger.info(
                    "First frame input: prefix=%s encoded_chars=%d",
                    diagnostics["input_prefix"],
                    diagnostics["input_encoded_chars"],
                )
            try:
                image = decode_frame(frame_data)
            except Exception as exc:
                details = f"frame {frame_index}: {type(exc).__name__}: {str(exc)[:160]}"
                processing_exceptions.append(details)
                logger.exception("Frame %d could not be decoded", frame_index)
                raise

            diagnostics["decoded_frames"] += 1
            if image_size is None:
                image_size = [int(image.shape[1]), int(image.shape[0])]
                diagnostics["image_size"] = image_size

            if DEBUG_SAVE_FRAMES and frame_index in debug_indexes:
                debug_path = DEBUG_FRAME_DIR / (
                    f"{uuid4().hex[:12]}_frame_{frame_index}.jpg"
                )
                try:
                    saved = cv2.imwrite(
                        str(debug_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    )
                    if not saved:
                        raise OSError("cv2.imwrite returned false")
                    debug_frame_files.append(str(debug_path))
                    logger.info("Saved debug frame %s", debug_path)
                except Exception as exc:
                    details = (
                        f"frame {frame_index} debug save: "
                        f"{type(exc).__name__}: {str(exc)[:160]}"
                    )
                    processing_exceptions.append(details)
                    logger.exception("Could not save debug frame %d", frame_index)

            try:
                result = pose.process(image)
            except Exception as exc:
                details = f"frame {frame_index} pose: {type(exc).__name__}: {str(exc)[:160]}"
                processing_exceptions.append(details)
                logger.exception("Pose processing failed for frame %d", frame_index)
                hip_positions.append(np.nan)
                signal_sources.append(None)
                continue

            if not result.pose_landmarks:
                diagnostics["pose_missing_frames"] += 1
                hip_positions.append(np.nan)
                signal_sources.append(None)
                continue

            pose_detected_frames += 1
            diagnostics["pose_detected_frames"] = pose_detected_frames
            landmarks = result.pose_landmarks.landmark
            left = landmarks[pose_module.PoseLandmark.LEFT_HIP]
            right = landmarks[pose_module.PoseLandmark.RIGHT_HIP]
            left_shoulder = landmarks[pose_module.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[pose_module.PoseLandmark.RIGHT_SHOULDER]
            left_ankle = landmarks[pose_module.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[pose_module.PoseLandmark.RIGHT_ANKLE]

            shoulders_visible = min(
                left_shoulder.visibility,
                right_shoulder.visibility,
            ) >= 0.35
            lower_body_visible = (
                min(left.visibility, right.visibility) >= 0.35
                or min(left_ankle.visibility, right_ankle.visibility) >= 0.35
            )
            if not shoulders_visible or not lower_body_visible:
                visibility_rejected_frames += 1
                diagnostics["visibility_rejected_frames"] = visibility_rejected_frames
                hip_positions.append(np.nan)
                signal_sources.append(None)
                continue

            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            body_scale = float(
                np.hypot(
                    left_shoulder.x - right_shoulder.x,
                    left_shoulder.y - right_shoulder.y,
                )
            )
            if body_scale < 0.03:
                scale_rejected_frames += 1
                diagnostics["scale_rejected_frames"] = scale_rejected_frames
                hip_positions.append(np.nan)
                signal_sources.append(None)
                continue

            if min(left_ankle.visibility, right_ankle.visibility) >= 0.35:
                movement_y = (left_ankle.y + right_ankle.y) / 2
                signal_sources.append("ankle")
                ankle_signal_frames += 1
                diagnostics["ankle_signal_frames"] = ankle_signal_frames
            else:
                movement_y = (left.y + right.y) / 2
                signal_sources.append("hip")
                hip_fallback_frames += 1
                diagnostics["hip_fallback_frames"] = hip_fallback_frames

            # Remove camera translation and normalize for camera distance.
            # Lower values mean the lower body rose relative to the shoulders.
            hip_positions.append((movement_y - shoulder_y) / body_scale)
            valid_frames += 1

    logger.info(
        "Pose diagnostics: decoded=%d detected=%d missing=%d visibility_rejected=%d scale_rejected=%d image_size=%s",
        diagnostics["decoded_frames"],
        diagnostics["pose_detected_frames"],
        diagnostics["pose_missing_frames"],
        diagnostics["visibility_rejected_frames"],
        diagnostics["scale_rejected_frames"],
        diagnostics["image_size"],
    )

    total_frames = len(frames)
    effective_interval_ms = (
        duration_seconds * 1000 / max(1, total_frames - 1)
        if duration_seconds is not None
        else interval_ms
    )
    valid_fraction = valid_frames / total_frames
    if ankle_signal_frames and hip_fallback_frames:
        signal_used = "ankle_with_hip_fallback"
    elif ankle_signal_frames:
        signal_used = "ankle"
    elif hip_fallback_frames:
        signal_used = "hip"
    else:
        signal_used = "none"
    diagnostics["signal_used"] = signal_used
    diagnostics["valid_signal_samples"] = valid_frames
    diagnostics["detected_peaks"] = 0

    if valid_frames < max(6, int(total_frames * 0.5)):
        return {
            "completed_skips": None,
            "confidence": "low",
            "reason": "The body was not detected reliably enough.",
            "limitations": (
                "Keep the full body visible, especially the hips and legs."
            ),
            "frames_analyzed": total_frames,
            "valid_pose_frames": valid_frames,
            "sampling_interval_ms": round(effective_interval_ms, 2),
            "sampling_fps": round(1000 / effective_interval_ms, 3),
            "duration_seconds": round(
                max(0.001, duration_seconds or (total_frames - 1) * interval_ms / 1000),
                2,
            ),
            "frames_analyzed": total_frames,
            "estimated_skips_per_minute": None,
            "signal_quality": {
                "valid_fraction": round(valid_fraction, 3),
                "motion_amplitude": 0.0,
                "cycle_regularity": 0.0,
            },
            "detection_diagnostics": diagnostics,
        }

    signal = np.asarray(hip_positions, dtype=float)
    signal, source_offset = align_signal_sources(
        signal, np.asarray(signal_sources, dtype=object)
    )
    diagnostics["source_baseline_offset"] = round(source_offset, 4)
    peaks, quality = detect_jump_cycles(signal, effective_interval_ms)
    diagnostics["detected_peaks"] = int(len(peaks))
    logger.info(
        "Signal diagnostics: source=%s valid_samples=%d range=%.4f peaks=%s prominences=%s",
        signal_used,
        valid_frames,
        quality["signal_range"],
        quality["peak_timestamps_seconds"],
        quality["peak_prominences"],
    )

    duration = max(0.001, duration_seconds or (total_frames - 1) * interval_ms / 1000)
    count = int(len(peaks))
    confidence = "low"
    if valid_fraction >= 0.8 and quality["motion_amplitude"] >= 0.03:
        confidence = "medium"
    if (
        valid_fraction >= 0.9
        and quality["motion_amplitude"] >= 0.06
        and quality["cycle_regularity"] >= 0.55
        and count >= 2
    ):
        confidence = "higher"

    return {
        "completed_skips": count,
        "confidence": confidence,
        "reason": (
            "Estimated from separated, repeated upward lower-body movements "
            f"using the {signal_used} signal."
        ),
        "limitations": (
            "Lower-body movement is not direct rope-rotation detection. "
            "The estimate can miss small jumps or count other bouncing. "
            "Compare it with a manually counted video."
        ),
        "duration_seconds": round(duration, 2),
        "estimated_skips_per_minute": round(count * 60 / duration, 1),
        "frames_analyzed": total_frames,
        "valid_pose_frames": valid_frames,
        "sampling_interval_ms": round(effective_interval_ms, 2),
        "sampling_fps": round(1000 / effective_interval_ms, 3),
        "signal_quality": quality,
        "detection_diagnostics": diagnostics,
    }


@app.get("/")
def home():
    file_path = ROOT / "index.html"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(file_path)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "counter": "MediaPipe Pose + peak detection",
        "gemma_model": MODEL,
        "cloud_key_configured": bool(API_KEY),
        "debug_frames_enabled": DEBUG_SAVE_FRAMES,
        "debug_frame_dir": str(DEBUG_FRAME_DIR),
    }


@app.post("/api/live/start")
def live_start():
    session_id = uuid4().hex
    with LIVE_SESSIONS_LOCK:
        LIVE_SESSIONS[session_id] = LiveSession()
    return {"session_id": session_id, "sample_fps": 5}


@app.post("/api/live/frame")
def live_frame(payload: LiveFrameRequest):
    with LIVE_SESSIONS_LOCK:
        session = LIVE_SESSIONS.get(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Live workout session not found.")
    started = time.perf_counter()
    try:
        result = session.process(payload.frame, payload.elapsed_seconds)
        result["processing_time_seconds"] = round(time.perf_counter() - started, 3)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Live frame processing failed")
        raise HTTPException(
            status_code=502,
            detail=f"Live processing failed ({type(exc).__name__}).",
        ) from exc


@app.post("/api/live/stop")
def live_stop(payload: LiveStopRequest):
    with LIVE_SESSIONS_LOCK:
        session = LIVE_SESSIONS.pop(payload.session_id, None)
    if session is not None:
        session.close()
    return {"ok": True}


@app.post("/api/count")
def count_skips(payload: CountRequest):
    started = time.perf_counter()
    try:
        result = count_jumps(
            payload.frames,
            payload.interval_ms,
            payload.duration_seconds,
        )
        result["requested_fps"] = payload.requested_fps
        result["processing_time_seconds"] = round(time.perf_counter() - started, 3)
        return result
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Counting failed ({type(exc).__name__}). Check server logs.",
        ) from exc


@app.post("/api/coach")
def coach(payload: CoachRequest):
    """Ask Gemma to interpret measured workout statistics."""
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OLLAMA_API_KEY is not configured on the server.",
        )

    pace = payload.completed_skips * 60 / payload.duration_seconds
    stats = {
        "completed_skips": payload.completed_skips,
        "duration_seconds": payload.duration_seconds,
        "rest_seconds": payload.rest_seconds,
        "estimated_skips_per_minute": round(pace, 1),
    }

    prompt = (
        "You are SkipSense, a concise jump-rope workout coach. "
        "Interpret only the supplied statistics. Do not invent heart rate, "
        "calories, fatigue, medical facts, or trends from a single session. "
        "Explain what the metrics show, give 1-2 practical suggestions, "
        "and state that fatigue cannot be measured directly from these "
        "statistics. Return a short response in plain text.\n\n"
        f"Workout statistics: {json.dumps(stats)}"
    )

    try:
        client = ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=120.0,
        )
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return {
            "model": MODEL,
            "statistics": stats,
            "analysis": response.message.content,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemma request failed ({type(exc).__name__}).",
        ) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
