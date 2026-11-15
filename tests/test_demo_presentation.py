"""The display must preserve main's continuous coordinates and raw evidence."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from PIL import Image

from demo import presentation


class PresentationTests(unittest.TestCase):
    def setUp(self):
        replay = Path(__file__).resolve().parents[1] / "demo" / "replay"
        self.result = json.loads((replay / "records" / "wing-context.json").read_text())
        frame = self.result["keyframes"][0]
        with Image.open(replay / self.result["frames"][0]["source"]) as image:
            self.image = image.convert("RGB").resize((frame["kf_W"], frame["kf_H"]))
        self.decoded = self.result["decoded"][0]
        self.crop = self.result["frames"][0]["crop_box"]

    def test_overlays_preserve_inputs_and_original_view_is_unmodified(self):
        before, pixels = deepcopy(self.result), self.image.tobytes()
        for mode in ("combined", "vertical", "horizontal", "original"):
            rendered = presentation.render_detection(self.image, self.decoded, self.crop, mode)
            if mode == "original":
                self.assertIsNot(rendered, self.image)
                self.assertEqual(rendered.tobytes(), pixels)
        self.assertEqual(self.result, before)
        self.assertEqual(self.image.tobytes(), pixels)

    def test_overlay_uses_continuous_decoded_edge_and_exact_planner_crop(self):
        rendered = presentation.render_detection(self.image, self.decoded, self.crop, selected="left")
        x0, y0, x1, y1 = self.decoded["region"]
        # The external rulers offset the picture, never its model coordinates.
        self.assertEqual(rendered.getpixel((44 + int(x0), 38 + int((y0 + y1) / 2))), presentation.ORANGE)
        self.assertEqual(rendered.getpixel((44 + int(x1), 38 + int(y0 + 70))), presentation.GREEN)
        self.assertEqual(rendered.getpixel((44 + int(self.crop[2]), 38 + 430)), presentation.BLUE)
        # main decodes a corrected expectation: it must not snap to argmax line 6.
        argmax_x = self.decoded["edges"]["left"]["argmax_idx"] / 16 * self.image.width
        self.assertNotEqual(int(x0), int(argmax_x))
        self.assertNotEqual(rendered.getpixel((44 + int(argmax_x), 38 + int((y0 + y1) / 2))), presentation.ORANGE)

    def test_probability_chart_uses_raw_grid_line_choices(self):
        before = deepcopy(self.result)
        for boundary in ("left", "right", "top", "bottom"):
            rows, labels, edge = presentation.boundary_distribution(self.result, 0, boundary)
            raw = self.result["response"]["answers"][f"f1_{boundary}"]["probabilities"]
            self.assertEqual([r["Probability"] for r in rows], [raw[label] for label in labels])
            self.assertEqual(labels[0], "0" if boundary in ("left", "right") else "A")
            self.assertEqual(edge, self.decoded["edges"][boundary])
        self.assertEqual(self.result, before)


if __name__ == "__main__":
    unittest.main()
