import unittest

from reframe.gemini_baseline import validate_anchors, validate_events


class BaselineValidationTests(unittest.TestCase):
    def test_normalized_coordinates_and_clip_time_are_mapped_to_source(self):
        rows = [{"t_s": t, "shot": "wide", "region": [0.25, 0.2, 0.75, 0.8],
                 "focus": [0.6, 0.4], "focus_description": "ball", "uncertain": False}
                for t in [0, 0.5]]
        kfs = validate_anchors({"anchors": rows}, 1, 120, 1920, 1080)
        self.assertEqual(kfs[1]["t"], 120.5)
        self.assertEqual(kfs[0]["region_kf"], [480, 216, 1440, 864])
        self.assertEqual(kfs[0]["focus_kf"], {"x": 1152, "y": 432})

    def test_missing_temporal_coverage_is_not_silently_interpolated(self):
        with self.assertRaises(ValueError):
            validate_anchors({"anchors": []}, 10, 0, 1280, 720)

    def test_bad_coordinates_are_not_silently_clamped(self):
        row = {"t_s": 0, "shot": "wide", "region": [0, 0, 2, 1], "focus": [0.5, 0.5],
               "focus_description": "ball", "uncertain": False}
        with self.assertRaises(ValueError):
            validate_anchors({"anchors": [row]}, 0.5, 0, 1280, 720)

    def test_events_cannot_extend_beyond_uploaded_video(self):
        with self.assertRaises(ValueError):
            validate_events({"events": [{"start_s": 95, "peak_s": 99, "end_s": 105}]}, 100, 3, 45)

    def test_overlapping_events_rejected(self):
        with self.assertRaises(ValueError):
            validate_events({"events": [{"start_s": 10, "peak_s": 15, "end_s": 25},
                                         {"start_s": 20, "peak_s": 26, "end_s": 30}]}, 100, 3, 45)


if __name__ == "__main__":
    unittest.main()
