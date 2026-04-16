from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "poster-img"
ROC_SRC = ROOT / "outputs" / "figures" / "resnet50_transfer_full_v1" / "test_roc_curve.png"
GRADCAM_DIR = ROOT / "outputs" / "figures" / "resnet50_transfer_full_v1" / "gradcam"
RAW_DIR = ROOT / "data" / "raw"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates += [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
        ]
    else:
        candidates += [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def find_image(name: str) -> Path:
    matches = list(RAW_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find image: {name}")
    return matches[0]


def draw_centered(draw: ImageDraw.ImageDraw, box, text, font, fill, multiline=False, spacing=8):
    x0, y0, x1, y1 = box
    if multiline:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.multiline_text(((x0 + x1 - tw) / 2, (y0 + y1 - th) / 2), text, font=font, fill=fill, spacing=spacing, align="center")
    else:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((x0 + x1 - tw) / 2, (y0 + y1 - th) / 2), text, font=font, fill=fill)


def generate_roc_poster():
    OUT_DIR.mkdir(exist_ok=True)
    roc = Image.open(ROC_SRC).convert("RGB")
    canvas = Image.new("RGB", (2400, 1800), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(64, bold=True)
    subtitle_font = load_font(34, bold=False)
    body_font = load_font(30, bold=False)

    draw_centered(
        draw,
        (80, 40, 2320, 180),
        "ROC Curve: ResNet50 with Transfer Learning",
        title_font,
        (0, 0, 0),
    )

    draw_centered(
        draw,
        (120, 180, 2280, 300),
        "Selected best-balance model for the poster analysis",
        subtitle_font,
        (60, 60, 60),
    )

    framed = fit_image(roc, 1800, 1150)
    img_x = (2400 - framed.width) // 2
    img_y = 330
    canvas.paste(framed, (img_x, img_y))

    draw.rectangle((img_x - 8, img_y - 8, img_x + framed.width + 8, img_y + framed.height + 8), outline=(60, 60, 60), width=3)

    footer = (
        "Poster note: test AUC = 0.7454, accuracy = 0.6955, F1-score = 0.6547. "
        "This model was selected because it provided the strongest overall balance "
        "between classification metrics."
    )
    draw.multiline_text((140, 1540), footer, font=body_font, fill=(0, 0, 0), spacing=10)

    canvas.save(OUT_DIR / "roc_curve_poster.png", dpi=(300, 300))


def generate_gradcam_poster():
    OUT_DIR.mkdir(exist_ok=True)
    summary = json.loads((GRADCAM_DIR / "gradcam_summary.json").read_text(encoding="utf-8"))
    by_category = {item["category"]: item for item in summary}

    config = [
        ("correct_normal", "Correct Normal", "Pred: Normal"),
        ("correct_abnormal", "Correct Abnormal", "Pred: Abnormal"),
        ("false_positive", "False Positive", "Pred: Abnormal"),
        ("false_negative", "False Negative", "Pred: Normal"),
    ]

    paths = {
        "correct_normal": GRADCAM_DIR / "correct_normal_00003468_005.png",
        "correct_abnormal": GRADCAM_DIR / "correct_abnormal_00015799_013.png",
        "false_positive": GRADCAM_DIR / "false_positive_00021772_015.png",
        "false_negative": GRADCAM_DIR / "false_negative_00004482_001.png",
    }

    canvas = Image.new("RGB", (2600, 2200), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(60, bold=True)
    label_font = load_font(34, bold=True)
    meta_font = load_font(26, bold=False)

    draw_centered(draw, (80, 40, 2520, 170), "Grad-CAM Examples", title_font, (0, 0, 0))
    draw_centered(
        draw,
        (120, 165, 2480, 270),
        "Representative correct and error cases from ResNet50 with transfer learning",
        load_font(30, bold=False),
        (60, 60, 60),
    )

    cell_w, cell_h = 1120, 760
    positions = [
        (120, 320),
        (1360, 320),
        (120, 1210),
        (1360, 1210),
    ]

    for (category, title, pred_text), (x, y) in zip(config, positions):
        frame_x0, frame_y0 = x, y
        frame_x1, frame_y1 = x + cell_w, y + cell_h
        draw.rounded_rectangle((frame_x0, frame_y0, frame_x1, frame_y1), radius=18, outline=(70, 70, 70), width=3)

        draw_centered(draw, (x + 20, y + 16, x + cell_w - 20, y + 72), title, label_font, (0, 0, 0))

        img = Image.open(paths[category]).convert("RGB")
        fitted = fit_image(img, cell_w - 60, 520)
        canvas.paste(fitted, (x + (cell_w - fitted.width) // 2, y + 90))

        meta = by_category.get(category, {})
        prob = meta.get("probability_abnormal", None)
        label_line = f"True label: {'Abnormal' if meta.get('true_label', 0) == 1 else 'Normal'}   |   {pred_text}"
        prob_line = f"P(abnormal) = {prob:.3f}" if isinstance(prob, float) else ""
        draw_centered(draw, (x + 20, y + 630, x + cell_w - 20, y + 690), label_line, meta_font, (0, 0, 0))
        if prob_line:
            draw_centered(draw, (x + 20, y + 680, x + cell_w - 20, y + 735), prob_line, meta_font, (80, 80, 80))

    footer = (
        "These examples illustrate that Grad-CAM often highlights plausible thoracic regions in correct cases, "
        "but explanation maps do not guarantee correct decisions in false positive and false negative cases."
    )
    draw.multiline_text((120, 2050), footer, font=load_font(28), fill=(0, 0, 0), spacing=8)

    canvas.save(OUT_DIR / "gradcam_panel_poster.png", dpi=(300, 300))


def generate_dataset_example_poster():
    OUT_DIR.mkdir(exist_ok=True)
    samples = [
        ("Normal Example", "No Finding", find_image("00010002_000.png")),
        ("Abnormal Example", "Effusion", find_image("00010010_000.png")),
    ]

    canvas = Image.new("RGB", (2400, 1500), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(64, bold=True)
    subtitle_font = load_font(30, bold=False)
    label_font = load_font(36, bold=True)
    meta_font = load_font(28, bold=False)

    draw_centered(draw, (80, 40, 2320, 150), "NIH ChestXray14 Dataset Examples", title_font, (0, 0, 0))
    draw_centered(
        draw,
        (120, 145, 2280, 225),
        "Representative normal and abnormal chest X-ray inputs used in the binary classification task",
        subtitle_font,
        (60, 60, 60),
    )

    positions = [(120, 280), (1240, 280)]
    panel_w, panel_h = 1040, 1080

    for (title, diagnosis, path), (x, y) in zip(samples, positions):
        draw.rounded_rectangle((x, y, x + panel_w, y + panel_h), radius=18, outline=(70, 70, 70), width=3)
        draw_centered(draw, (x + 20, y + 20, x + panel_w - 20, y + 90), title, label_font, (0, 0, 0))
        draw_centered(draw, (x + 20, y + 86, x + panel_w - 20, y + 145), diagnosis, meta_font, (70, 70, 70))

        img = Image.open(path).convert("L").convert("RGB")
        fitted = fit_image(img, panel_w - 120, 820)
        img_x = x + (panel_w - fitted.width) // 2
        img_y = y + 165
        canvas.paste(fitted, (img_x, img_y))
        draw.rectangle((img_x - 4, img_y - 4, img_x + fitted.width + 4, img_y + fitted.height + 4), outline=(130, 130, 130), width=2)

        footer = "Binary label: Normal (0)" if "Normal" in title else "Binary label: Abnormal (1)"
        draw_centered(draw, (x + 20, y + 1010, x + panel_w - 20, y + 1060), footer, meta_font, (0, 0, 0))

    canvas.save(OUT_DIR / "dataset_example_poster.png", dpi=(300, 300))


def main():
    generate_roc_poster()
    generate_gradcam_poster()
    generate_dataset_example_poster()
    print(f"saved images to: {OUT_DIR}")


if __name__ == "__main__":
    main()
