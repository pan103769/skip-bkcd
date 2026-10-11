import base64
import types
import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app
from app import CountRequest, align_signal_sources, detect_jump_cycles


class JumpCycleTests(unittest.TestCase):
    def test_realistic_jump_frequencies(self):
        interval_ms = 187
        frame_count = 120
        time = np.arange(frame_count) * interval_ms / 1000

        for frequency, expected in ((0.6, 13), (0.7, 15), (0.8, 17)):
            signal = 0.08 * np.sin(2 * np.pi * frequency * time)
            peaks, quality = detect_jump_cycles(signal, interval_ms)
            self.assertTrue(expected - 2 <= len(peaks) <= expected + 2)
            self.assertGreater(quality["cycle_regularity"], 0.75)

    def test_slow_movement_is_not_counted_as_jumps(self):
        interval_ms = 187
        time = np.arange(120) * interval_ms / 1000
        signal = 0.05 * np.sin(2 * np.pi * 0.08 * time)
        peaks, _ = detect_jump_cycles(signal, interval_ms)
        self.assertLessEqual(len(peaks), 3)

    def test_fast_normal_and_slow_live_rates(self):
        interval_ms = 125
        time = np.arange(160) * interval_ms / 1000
        expected_ranges = ((0.5, 7, 11), (0.8, 12, 18), (1.2, 19, 27))
        for frequency, minimum, maximum in expected_ranges:
            signal = 0.08 * np.sin(2 * np.pi * frequency * time)
            peaks, quality = detect_jump_cycles(signal, interval_ms)
            self.assertTrue(minimum <= len(peaks) <= maximum)
            self.assertGreater(quality["cycle_regularity"], 0.8)

    def test_missing_samples_are_interpolated(self):
        interval_ms = 187
        time = np.arange(120) * interval_ms / 1000
        signal = 0.08 * np.sin(2 * np.pi * 0.7 * time)
        signal[30:33] = np.nan
        peaks, quality = detect_jump_cycles(signal, interval_ms)
        self.assertTrue(13 <= len(peaks) <= 17)
        self.assertAlmostEqual(quality["valid_fraction"], 0.975, places=3)

    def test_mixed_ankle_hip_baseline_is_aligned(self):
        interval_ms = 22.62 * 1000 / 678
        time = np.arange(679) * interval_ms / 1000
        ankle_signal = 2.2 - 0.16 * np.sin(2 * np.pi * 0.67 * time)
        mixed_signal = ankle_signal.copy()
        sources = np.full(679, "ankle", dtype=object)
        sources[300:375] = "hip"
        mixed_signal[300:375] -= 1.05

        aligned, offset = align_signal_sources(mixed_signal, sources)
        peaks, quality = detect_jump_cycles(aligned, interval_ms)

        self.assertTrue(1.0 < offset < 1.1)
        self.assertTrue(13 <= len(peaks) <= 17)
        self.assertGreater(quality["cycle_regularity"], 0.9)


class SamplingContractTests(unittest.TestCase):
    def test_all_supported_rates_and_largest_30_second_payload(self):
        for requested_fps in (15, 30, 60):
            payload = CountRequest(
                frames=["frame"] * 1800,
                interval_ms=17,
                duration_seconds=30,
                requested_fps=requested_fps,
            )
            self.assertEqual(len(payload.frames), 1800)
            self.assertEqual(payload.requested_fps, requested_fps)

        with self.assertRaises(ValidationError):
            CountRequest(frames=["frame"] * 2001, requested_fps=60)

    def test_count_endpoint_reports_processing_metadata(self):
        original = app.count_jumps

        def fake_count(frames, interval_ms, duration_seconds=None):
            return {
                "frames_analyzed": len(frames),
                "sampling_fps": 60.0,
                "completed_skips": 15,
            }

        app.count_jumps = fake_count
        try:
            response = TestClient(app.app).post(
                "/api/count",
                json={
                    "frames": ["frame"] * 1800,
                    "interval_ms": 17,
                    "duration_seconds": 30,
                    "requested_fps": 60,
                },
            )
        finally:
            app.count_jumps = original

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["frames_analyzed"], 1800)
        self.assertEqual(data["requested_fps"], 60)
        self.assertGreaterEqual(data["processing_time_seconds"], 0)


class LiveSessionTests(unittest.TestCase):
    def test_live_start_advertises_bounded_sample_rate(self):
        response = TestClient(app.app).post("/api/live/start")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sample_fps"], 8)
        stop = TestClient(app.app).post(
            "/api/live/stop", json={"session_id": data["session_id"]}
        )
        self.assertEqual(stop.status_code, 200)

    def test_live_session_counts_synthetic_ankle_motion(self):
        class PoseLandmark:
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28

        class FakePose:
            def __init__(self, **kwargs):
                self.index = 0

            def process(self, image):
                elapsed = self.index * 0.2
                self.index += 1
                points = [types.SimpleNamespace(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]
                points[11] = types.SimpleNamespace(x=0.4, y=0.3, visibility=1.0)
                points[12] = types.SimpleNamespace(x=0.6, y=0.3, visibility=1.0)
                ankle_y = 0.75 - 0.08 * np.sin(2 * np.pi * 0.7 * elapsed)
                points[23] = types.SimpleNamespace(x=0.45, y=0.55, visibility=1.0)
                points[24] = types.SimpleNamespace(x=0.55, y=0.55, visibility=1.0)
                points[27] = types.SimpleNamespace(x=0.45, y=ankle_y, visibility=1.0)
                points[28] = types.SimpleNamespace(x=0.55, y=ankle_y, visibility=1.0)
                return types.SimpleNamespace(
                    pose_landmarks=types.SimpleNamespace(landmark=points)
                )

            def close(self):
                pass

        original_pose = app.mp.solutions.pose
        app.mp.solutions.pose = types.SimpleNamespace(
            Pose=FakePose,
            PoseLandmark=PoseLandmark,
        )
        try:
            session = app.LiveSession()
            ok, encoded = cv2.imencode(
                ".jpg", np.zeros((32, 32, 3), dtype=np.uint8)
            )
            self.assertTrue(ok)
            frame = "data:image/jpeg;base64," + base64.b64encode(
                encoded.tobytes()
            ).decode()
            result = None
            for index in range(80):
                result = session.process(frame, index * 0.2)
            session.close()
        finally:
            app.mp.solutions.pose = original_pose

        self.assertGreaterEqual(result["completed_skips"], 8)
        self.assertEqual(result["signal_used"], "ankle")
        self.assertEqual(result["valid_signal_samples"], 80)
        self.assertEqual(len(result["landmarks"]), 33)


if __name__ == "__main__":
    unittest.main()
