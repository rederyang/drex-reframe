from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest

from demo import backend
from reframe import media, pipeline


def response_for(count=8):
    answers = {}
    for k in range(1, count + 1):
        q = pipeline.kf_questions(k)
        choices = {"left": "3", "right": "12", "top": "C", "bottom": "H", "focus_x": "5", "focus_y": "E", "shot": "wide"}
        for name, question in q.items():
            labels = list(question["criteria"])
            choice = choices[name.split("_", 1)[1]]
            probabilities = {label: 1 - 0.01 * (len(labels) - 1) if label == choice else 0.01 for label in labels}
            answers[name] = {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": 0.9}
    return {"answers": answers, "_latency_s": 0.2, "request_id": "test-request", "usage": {"input_tokens": 100}}


def frames_for(start=12):
    return [(start + i * 0.5, Image.new("RGB", (320, 180), (int(start) % 255, 65 + i * 10, 40))) for i in range(8)]


class MainPipelineTests(unittest.TestCase):
    def test_refactor_matches_unmodified_main_stage3_payload_and_output(self):
        # Compare against the actual main implementation at the verified base,
        # including its floor subtraction and newly added focus points.
        baseline_git = subprocess.run(["git", "show", "ab86fd8:reframe/pipeline.py"], cwd=backend.ROOT, capture_output=True, text=True)
        if baseline_git.returncode:
            self.skipTest("The main baseline commit is unavailable in this source export or shallow clone")
        source = baseline_git.stdout
        baseline = types.ModuleType("reframe._baseline_pipeline")
        baseline.__file__ = str(backend.ROOT / "reframe" / "pipeline.py")
        baseline.__package__ = "reframe"
        exec(compile(source, "main-baseline", "exec"), baseline.__dict__)
        frames = frames_for()
        calls = []
        def capture(state, questions, images, **kwargs):
            calls.append({"state": state, "questions": questions,
                          "images": [backend.nace.image_to_b64(im) for im in images], "kwargs": kwargs})
            return response_for()
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / "old", Path(directory) / "new"
            old.mkdir(); new.mkdir()
            events = [{"clip": [12, 16]}]
            with patch.object(baseline.media, "grab_many", return_value=frames), patch.object(baseline, "drex", side_effect=capture):
                expected = baseline.stage3(Path("unused"), {}, events, old)
            with patch.object(pipeline.media, "grab_many", return_value=frames), patch.object(pipeline, "drex", side_effect=capture):
                actual = pipeline.stage3(Path("unused"), {}, events, new)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(expected, actual)
        self.assertEqual(len(calls[1]["questions"]), 56)
        self.assertIn("focus_kf", actual["0"][0])

    def test_custom_goal_changes_only_state_and_keeps_main_decoding(self):
        frames = frames_for()
        default = pipeline.prepare_keyframe_batch(frames)
        custom = pipeline.prepare_keyframe_batch(frames, "Keep the visible ball in view.")
        self.assertEqual(default["questions"], custom["questions"])
        self.assertIn("Keep the visible ball in view.", custom["state"])
        self.assertTrue(custom["state"].startswith(default["state"]))
        with patch.object(pipeline, "drex", return_value=response_for()) as call:
            result = pipeline.detect_keyframe_batch(frames, "Keep the visible ball in view.")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(len(result["decoded"]), 8)
        self.assertAlmostEqual(result["decoded"][0]["region"][0], 60)
        self.assertAlmostEqual(result["decoded"][0]["focus"]["x"], 100)
        self.assertEqual(call.call_args.kwargs["tag"], "s3_keyframes")

    def test_invalid_batch_or_goal_is_rejected_before_api(self):
        with patch.object(pipeline, "drex") as call:
            for frames in [[], frames_for() + frames_for()]:
                with self.assertRaises(ValueError):
                    pipeline.detect_keyframe_batch(frames)
            with self.assertRaises(ValueError):
                pipeline.detect_keyframe_batch(frames_for(), "x" * 2001)
            call.assert_not_called()

    def test_scene_adapter_uses_bundled_clip_and_main_media(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "clip.mp4").touch()
            (folder / "manifest.json").write_text(json.dumps({"pipeline_base": "test", "clips": [{"id": "wing", "file": "clip.mp4",
                "W": 320, "H": 180, "fps": 25, "duration": 4, "scene_start": 28, "source_start": 4378, "cuts": [0.24]}]}))
            with patch.object(subprocess, "check_output", side_effect=AssertionError("No Git or external asset lookup")):
                scene = backend.load_scene(assets_root=folder)
            self.assertEqual(scene["start"], 0)
            self.assertAlmostEqual(scene["cuts"][0], 0.24)
            self.assertEqual(scene["source_start"], 4378)
            with patch.object(media, "grab_many", return_value=frames_for(0)) as grab:
                backend.extract_batch(scene, 0)
            self.assertEqual(grab.call_args.args[1:], (0.0, 4.0, 0.5))
            for invalid in [-1, 0.5, float("nan"), True]:
                with self.assertRaises(ValueError):
                    backend.extract_batch(scene, invalid)

    def test_mismatched_preview_stops_network_call(self):
        frames = frames_for()
        prepared = pipeline.prepare_keyframe_batch(frames, "A different goal")
        with tempfile.TemporaryDirectory() as directory, patch.object(backend, "load_credentials"), patch.object(pipeline, "drex") as call:
            with self.assertRaisesRegex(ValueError, "Preview and actual"):
                backend.execute({}, frames, "actual goal", prepared, run_root=directory)
            call.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_full_stage_outputs_real_video_and_audio_using_main_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25",
                            "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=16000", "-t", "4",
                            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(video)], check=True)
            scene = {"video": video, "W": 320, "H": 180, "fps": 25, "dur": 4, "start": 0, "source_start": 4350, "cuts": []}
            frames = backend.extract_batch(scene, 0)
            prepared = pipeline.prepare_keyframe_batch(frames)
            events = []
            with patch.object(backend, "load_credentials"), patch.object(pipeline, "drex", return_value=response_for()) as api:
                result = backend.execute(scene, frames, "", prepared, lambda event, data: events.append(event), root / "runs")
            self.assertEqual(api.call_count, 1)
            self.assertEqual(events, ["prepared", "response", "decoded", "planning", "rendering"])
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", result["vertical"]]))
            output = next(s for s in info["streams"] if s["codec_type"] == "video")
            self.assertEqual((output["width"], output["height"], int(output["nb_frames"])), (720, 1280, 100))
            self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
            saved = json.loads((Path(result["directory"]) / "request.json").read_text())
            self.assertEqual(saved, backend.body_for(prepared))
            self.assertEqual(len(saved["questions"]), 56)
            self.assertEqual(backend.frame_crop(frames[0][1], result, 0)[0].size, (720, 1280))


class AppTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        video = root / "source.mp4"
        video.touch()
        self.scene = {"video": video, "W": 320, "H": 180, "fps": 25, "dur": 1800,
                      "start": 0, "source_start": 4350, "cuts": []}
        for patcher in [patch.object(backend, "load_scene", return_value=self.scene),
                        patch.object(backend, "extract_batch", side_effect=lambda scene, start: frames_for(start))]:
            patcher.start(); self.addCleanup(patcher.stop)
        self.app = AppTest.from_file(str(backend.ROOT / "demo" / "app.py"))
        self.root = root

    def fake_execute(self, scene, frames, goal, prepared, on_event=None):
        with patch.object(pipeline, "drex", return_value=response_for()):
            result = pipeline.detect_keyframe_batch(frames, goal or None, on_event=on_event)
        event = {"clip": [frames[0][0], frames[0][0] + 4]}
        plan = pipeline.plan_clip(scene, event, result["keyframes"])
        vertical, debug = self.root / "vertical.mp4", self.root / "debug.mp4"
        vertical.write_bytes(b"test only"); debug.write_bytes(b"test only")
        (self.root / "run.json").write_text("{}")
        return {**result, "plan": plan, "vertical": str(vertical), "debug": str(debug), "directory": str(self.root),
                "pipeline_revision": "test", "request_sha256": backend.digest(prepared), "total_s": 0.3}

    def test_english_custom_goal_and_local_view_changes(self):
        with patch.object(backend, "execute", side_effect=self.fake_execute) as run:
            app = self.app.run(timeout=30)
            self.assertFalse(app.exception)
            run.assert_not_called()
            app.selectbox(key="goal_preset").select("Passing options").run()
            self.assertIn("passing", app.text_input(key="goal").value)
            app.text_input(key="goal").set_value("Keep the ball and the nearest receiver").run()
            run.assert_not_called()
            app.button(key="run").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(run.call_args.args[2], "Keep the ball and the nearest receiver")
            self.assertEqual(run.call_count, 1)
            for view in ["Model input", "Frames", "Video"]:
                app.session_state["view"] = view
                app.run(timeout=30)
                self.assertFalse(app.exception)
            app.session_state["frame"] = 4
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(run.call_count, 1)
            self.assertTrue(all(not any("\u4e00" <= c <= "\u9fff" for c in entry.value) for entry in app.caption))

    def test_changed_clip_or_goal_invalidates_old_result(self):
        with patch.object(backend, "execute", side_effect=self.fake_execute) as run:
            app = self.app.run(timeout=30)
            app.button(key="run").click().run(timeout=30)
            self.assertIn("result", app.session_state)
            app.text_input(key="goal").set_value("Visible goalmouth action").run()
            self.assertNotIn("result", app.session_state)
            app.button(key="run").click().run(timeout=30)
            app.selectbox(key="clip").select("Touchline · 01:00").run()
            self.assertNotIn("result", app.session_state)
            self.assertEqual(run.call_count, 2)

    def test_error_is_english_and_never_shows_fabricated_output(self):
        with patch.object(backend, "execute", side_effect=RuntimeError("NACE unavailable")) as run:
            app = self.app.run(timeout=30)
            app.button(key="run").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any("NACE unavailable" in error.value for error in app.error))
            self.assertNotIn("result", app.session_state)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
