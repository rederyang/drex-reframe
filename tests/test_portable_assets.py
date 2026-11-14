"""Verify the checkout is sufficient for both live input and offline replay."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from demo import backend
from reframe import media
from reframe.grid import decode_region
from reframe.pipeline import GRID


class PortableAssetsTests(unittest.TestCase):
    def test_all_six_bundled_clips_match_manifest_and_are_four_seconds(self):
        manifest = json.loads((backend.ASSETS / "manifest.json").read_text())
        self.assertEqual(len(manifest["clips"]), 6)
        self.assertEqual(set(backend.CLIPS.values()), {c["id"] for c in manifest["clips"]})
        for clip in manifest["clips"]:
            scene = backend.load_scene(clip["id"])
            self.assertTrue(scene["video"].is_relative_to(backend.ASSETS))
            data = scene["video"].read_bytes()
            self.assertLess(len(data), 10 * 1024 * 1024)
            self.assertEqual(hashlib.sha256(data).hexdigest(), clip["sha256"])
            info = media.probe(scene["video"])
            self.assertAlmostEqual(info["duration"], 4, delta=0.15)
            self.assertEqual((info["W"], info["H"], info["fps"]), (clip["W"], clip["H"], clip["fps"]))
            self.assertTrue(all(0 < cut < 4 for cut in scene["cuts"]))

    def test_bundled_clip_decodes_without_original_scene_or_git(self):
        with patch.dict("os.environ", {"DREX_SCENE1_DIR": "/does-not-exist"}), patch.object(backend, "common_repository", create=True, side_effect=AssertionError("No external repository")):
            for identity in backend.CLIPS.values():
                scene = backend.load_scene(identity)
                frames = backend.extract_batch(scene, 0)
                self.assertEqual(len(frames), 8)
                self.assertEqual([t for t, _ in frames], [i * 0.5 for i in range(8)])
                self.assertEqual(frames[0][1].size, (1280, 720))

    def test_offline_records_have_every_asset_and_exact_main_decoding(self):
        root = backend.ROOT / "demo" / "replay"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(len(manifest["cases"]), 4)
        for clip in manifest["clips"]:
            self.assertTrue((root / clip["video"]).resolve().is_file())
        for case in manifest["cases"]:
            path = root / case["record"]
            text = path.read_text()
            self.assertNotIn("/Users/", text)
            self.assertNotIn("NACE_API_KEY", text)
            record = json.loads(text)
            self.assertEqual(len(record["request"]["questions"]), 56)
            self.assertEqual(len(record["frames"]), 8)
            for i, frame in enumerate(record["frames"]):
                for field in ("source", "input", "portrait"):
                    resolved = (root / frame[field]).resolve()
                    self.assertTrue(resolved.is_relative_to(backend.ASSETS))
                    self.assertTrue(resolved.is_file())
                request_image = record["request"]["images"][i]
                self.assertEqual(hashlib.sha256((root / request_image["file"]).read_bytes()).hexdigest(), request_image["sha256"])
                self.assertEqual(decode_region(record["response"]["answers"], GRID, 1280, 720, prefix=f"f{i+1}_"), record["decoded"][i])
            for field in ("video", "debug_video"):
                info = media.probe((root / record[field]).resolve())
                self.assertAlmostEqual(info["duration"], 4, delta=0.15)
            vertical = media.probe((root / record["video"]).resolve())
            self.assertEqual((vertical["W"], vertical["H"]), (720, 1280))

    def test_credentials_are_read_from_this_clone_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("NACE_API_KEY=test-only-placeholder\n")
            with patch.object(backend, "ROOT", root), patch.dict("os.environ", {"NACE_API_KEY": ""}):
                backend.load_credentials()
                self.assertEqual(backend.os.environ["NACE_API_KEY"], "test-only-placeholder")


if __name__ == "__main__":
    unittest.main()
