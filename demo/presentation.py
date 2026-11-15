"""POC-inspired views over main's unchanged region and camera geometry."""
from PIL import Image, ImageDraw, ImageFont
from reframe.pipeline import GRID

VIEWS = {"Region": "combined", "X boundaries": "vertical", "Y boundaries": "horizontal", "Original": "original"}
GREEN, BLUE, ORANGE, DARK = (41, 215, 137), (61, 178, 255), (255, 169, 72), (20, 31, 26)


def font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def tag(draw, text, x, y, color, size, bounds):
    face = font(size)
    box = draw.textbbox((0, 0), text, font=face)
    width, height = box[2] - box[0] + 16, box[3] - box[1] + 12
    x, y = max(2, min(x, bounds[0] - width - 2)), max(2, min(y, bounds[1] - height - 2))
    draw.rounded_rectangle((x, y, x + width, y + height), radius=5, fill=color)
    draw.text((x + 8, y + 5 - box[1]), text, font=face, fill=DARK)


def render_detection(image, decoded, crop_box, mode="combined", selected="left"):
    """Only render stored coordinates; never re-score, snap or re-decode them."""
    if mode == "original":
        return image.copy()
    if mode not in ("combined", "vertical", "horizontal"):
        raise ValueError("Unknown result view")
    W, H = image.size
    x0, y0, x1, y1 = decoded["region"]
    keep = (x0, 0, x1, H) if mode == "vertical" else (0, y0, W, y1) if mode == "horizontal" else (x0, y0, x1, y1)
    mask = Image.new("L", (W, H), 116)
    shade = ImageDraw.Draw(mask)
    if mode == "combined":
        shade.rectangle(crop_box, fill=52)
    shade.rectangle(keep, fill=0)
    source = Image.composite(Image.new("RGB", (W, H), DARK), image.convert("RGB"), mask)
    tint = Image.new("RGBA", (W, H))
    ImageDraw.Draw(tint).rectangle(keep, fill=(*GREEN, 25))
    source = Image.alpha_composite(source.convert("RGBA"), tint).convert("RGB")
    # These are main's grid LINE labels, rather than POC's one-based strips.
    left, top = (44 if mode != "vertical" else 12), (38 if mode != "horizontal" else 12)
    canvas = Image.new("RGB", (W + left + 12, H + top + 12), (240, 244, 237))
    canvas.paste(source, (left, top))
    draw = ImageDraw.Draw(canvas)
    text_size = max(16, W // 55)
    label_font = font(text_size)
    if mode != "horizontal":
        for i, label in enumerate(GRID.xlabels):
            x = left + GRID.x_of(i, W)
            if mode == "vertical":
                draw.line((x, top, x, top + H), fill=(106, 130, 114), width=1)
            draw.text((min(max(x, left + 8), left + W - 10), top - 15), label, font=label_font, fill=(94, 114, 100), anchor="mm")
    if mode != "vertical":
        for i, label in enumerate(GRID.ylabels):
            y = top + GRID.y_of(i, H)
            if mode == "horizontal":
                draw.line((left, y, left + W, y), fill=(106, 130, 114), width=1)
            draw.text((left - 18, min(max(y, top + 8), top + H - 10)), label, font=label_font, fill=(94, 114, 100), anchor="mm")
    line_width = max(3, W // 300)
    if mode == "combined":
        draw.rectangle((left + x0, top + y0, left + x1, top + y1), outline=GREEN, width=line_width + 1)
        tag(draw, "KEY REGION", left + x0, top + y0 - text_size - 16, GREEN, text_size, canvas.size)
    else:
        for edge in (("left", "right") if mode == "vertical" else ("top", "bottom")):
            value = decoded["edges"][edge]["px"]
            if mode == "vertical":
                x = left + value
                draw.line((x, top, x, top + H), fill=GREEN, width=line_width + 1)
                tag(draw, f"{edge.upper()}  {value:.0f}px", x + 5, top + 8, GREEN, text_size, canvas.size)
            else:
                y = top + value
                draw.line((left, y, left + W, y), fill=GREEN, width=line_width + 1)
                tag(draw, f"{edge.upper()}  {value:.0f}px", left + 8, y + 5, GREEN, text_size, canvas.size)
    cx0, cy0, cx1, cy1 = crop_box
    draw.rectangle((left + cx0, top + cy0, left + cx1, top + cy1), outline=BLUE, width=line_width)
    tag(draw, "9:16 CROP", left + cx0, top + cy0 + 7, BLUE, text_size, canvas.size)
    if selected in decoded["edges"]:
        value = decoded["edges"][selected]["px"]
        if selected in ("left", "right") and mode != "horizontal":
            x = left + value
            draw.line((x, top + y0, x, top + y1), fill=ORANGE, width=line_width + 1)
            draw.ellipse((x - 6, top - 9, x + 6, top + 3), fill=ORANGE)
        elif selected in ("top", "bottom") and mode != "vertical":
            y = top + value
            draw.line((left + x0, y, left + x1, y), fill=ORANGE, width=line_width + 1)
            draw.ellipse((left - 9, y - 6, left + 3, y + 6), fill=ORANGE)
    focus = decoded.get("focus")
    if focus:
        x, y = left + focus["x"], top + focus["y"]
        draw.ellipse((x - 15, y - 15, x + 15, y + 15), outline=DARK, width=5)
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=(255, 248, 224), width=3)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=ORANGE)
    return canvas


def boundary_distribution(result, index, boundary):
    """Keep raw Choice probabilities separate from main's decoded coordinate."""
    labels = GRID.xlabels if boundary in ("left", "right") else GRID.ylabels
    answer = result["response"]["answers"][f"f{index + 1}_{boundary}"]
    edge = result["decoded"][index]["edges"][boundary]
    rows = [{"Index": i, "Label": label, "Probability": answer.get("probabilities", {}).get(label, 0)} for i, label in enumerate(labels)]
    return rows, labels, edge
