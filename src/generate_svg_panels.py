from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
POSTER_DIR = ROOT / "poster-img"
RAW_DIR = ROOT / "data" / "raw"


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def png_bytes_data_uri(data: bytes) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def find_image(name: str) -> Path:
    matches = list(RAW_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find image: {name}")
    return matches[0]


def crop_case_panels(path: Path) -> tuple[str, str, str]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    top = 95
    bottom = h - 22
    panel_w = w // 3
    panels = [
        img.crop((0, top, panel_w, bottom)),
        img.crop((panel_w, top, 2 * panel_w, bottom)),
        img.crop((2 * panel_w, top, w, bottom)),
    ]
    uris = []
    for panel in panels:
        buf = io.BytesIO()
        panel.save(buf, format="PNG")
        uris.append(png_bytes_data_uri(buf.getvalue()))
    return tuple(uris)


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    svg = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
  {body}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def roc_svg() -> None:
    png = POSTER_DIR / "roc_curve_poster_transparent.png"
    if not png.exists():
        png = POSTER_DIR / "roc_curve_poster.png"
    href = image_data_uri(png)
    body = f"""
  <text x="1200" y="95" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="64" font-weight="700" fill="#000000">
    ROC Curve: ResNet50 with Transfer Learning
  </text>
  <text x="1200" y="150" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="34" fill="#444444">
    Selected best-balance model for poster presentation
  </text>
  <rect x="285" y="215" width="1830" height="1180" rx="12" ry="12" fill="none" stroke="#555555" stroke-width="3"/>
  <image x="300" y="230" width="1800" height="1150" xlink:href="{href}"/>
  <text x="1200" y="1605" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="30" fill="#111111">
    Test AUC = 0.7454, Accuracy = 0.6955, F1 = 0.6547
  </text>
"""
    write_svg(POSTER_DIR / "roc_curve_poster.svg", 2400, 1800, body)


def gradcam_svg() -> None:
    case_paths = [
        ("Correct Normal", ROOT / "outputs/figures/resnet50_transfer_full_v1/gradcam/correct_normal_00003468_005.png"),
        ("Correct Abnormal", ROOT / "outputs/figures/resnet50_transfer_full_v1/gradcam/correct_abnormal_00015799_013.png"),
        ("False Positive", ROOT / "outputs/figures/resnet50_transfer_full_v1/gradcam/false_positive_00021772_015.png"),
        ("False Negative", ROOT / "outputs/figures/resnet50_transfer_full_v1/gradcam/false_negative_00004482_001.png"),
    ]
    positions = [(90, 220), (1650, 220), (90, 1050), (1650, 1050)]
    card_w, card_h = 1460, 760
    panel_w, panel_h = 430, 500
    gap = 35
    body_parts = [
        """
  <text x="1600" y="90" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="62" font-weight="700" fill="#000000">
    Representative Grad-CAM Cases
  </text>
  <text x="1600" y="145" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="32" fill="#444444">
    Four example chest X-ray cases with editable labels for poster use
  </text>
"""
    ]

    for (title, case_path), (x0, y0) in zip(case_paths, positions):
        orig_href, cam_href, overlay_href = crop_case_panels(case_path)
        lefts = [x0 + 40, x0 + 40 + panel_w + gap, x0 + 40 + 2 * (panel_w + gap)]
        body_parts.append(
            f"""
  <rect x="{x0}" y="{y0}" width="{card_w}" height="{card_h}" rx="16" ry="16" fill="none" stroke="#555555" stroke-width="3"/>
  <text x="{x0 + card_w/2}" y="{y0 + 55}" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="34" font-weight="700" fill="#000000">
    {title}
  </text>
  <text x="{lefts[0] + panel_w/2}" y="{y0 + 112}" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="24" font-weight="700" fill="#000000">
    Original
  </text>
  <text x="{lefts[1] + panel_w/2}" y="{y0 + 112}" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="24" font-weight="700" fill="#000000">
    Grad-CAM
  </text>
  <text x="{lefts[2] + panel_w/2}" y="{y0 + 112}" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="24" font-weight="700" fill="#000000">
    Overlay
  </text>
  <image x="{lefts[0]}" y="{y0 + 130}" width="{panel_w}" height="{panel_h}" xlink:href="{orig_href}"/>
  <image x="{lefts[1]}" y="{y0 + 130}" width="{panel_w}" height="{panel_h}" xlink:href="{cam_href}"/>
  <image x="{lefts[2]}" y="{y0 + 130}" width="{panel_w}" height="{panel_h}" xlink:href="{overlay_href}"/>
"""
        )

    body_parts.append(
        """
  <text x="1600" y="1945" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="24" fill="#333333">
    Text is editable; image regions remain high-resolution embedded PNG panels.
  </text>
"""
    )
    write_svg(POSTER_DIR / "gradcam_four_case_large_text.svg", 3200, 2000, "".join(body_parts))


def composite_svg() -> None:
    roc_png = POSTER_DIR / "roc_curve_poster_transparent.png"
    if not roc_png.exists():
        roc_png = POSTER_DIR / "roc_curve_poster.png"
    grad_png = POSTER_DIR / "gradcam_panel_poster_transparent.png"
    if not grad_png.exists():
        grad_png = POSTER_DIR / "gradcam_panel_poster.png"

    roc_href = image_data_uri(roc_png)
    grad_href = image_data_uri(grad_png)
    body = f"""
  <text x="1600" y="85" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="54" font-weight="700" fill="#000000">
    Results Summary: Quantitative Performance and Explainability
  </text>
  <text x="1600" y="135" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="28" fill="#444444">
    Left: ROC of selected best-balance model    |    Right: Representative Grad-CAM cases
  </text>
  <rect x="72" y="262" width="1758" height="1388" rx="12" ry="12" fill="none" stroke="#555555" stroke-width="3"/>
  <image x="80" y="270" width="1742" height="1372" xlink:href="{roc_href}"/>
  <rect x="1972" y="262" width="1158" height="1388" rx="12" ry="12" fill="none" stroke="#555555" stroke-width="3"/>
  <image x="1980" y="270" width="1142" height="1372" xlink:href="{grad_href}"/>
  <text x="1600" y="1710" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="28" fill="#111111">
    DenseNet121 scratch achieved the highest AUC (0.7457), while ResNet50 transfer delivered the strongest balanced performance.
  </text>
"""
    write_svg(POSTER_DIR / "results_composite_poster.svg", 3200, 1800, body)


def dataset_example_svg() -> None:
    normal_href = image_data_uri(find_image("00010002_000.png"))
    abnormal_href = image_data_uri(find_image("00010010_000.png"))
    body = f"""
  <text x="1200" y="90" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="64" font-weight="700" fill="#000000">
    NIH ChestXray14 Dataset Examples
  </text>
  <text x="1200" y="145" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="30" fill="#444444">
    Representative normal and abnormal chest X-ray inputs used in the binary classification task
  </text>

  <rect x="120" y="280" width="1040" height="1080" rx="18" ry="18" fill="none" stroke="#555555" stroke-width="3"/>
  <text x="640" y="340" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="36" font-weight="700" fill="#000000">
    Normal Example
  </text>
  <text x="640" y="392" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="28" fill="#555555">
    No Finding
  </text>
  <rect x="250" y="445" width="780" height="820" fill="none" stroke="#888888" stroke-width="2"/>
  <image x="250" y="445" width="780" height="820" preserveAspectRatio="xMidYMid meet" xlink:href="{normal_href}"/>
  <text x="640" y="1325" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="28" fill="#000000">
    Binary label: Normal (0)
  </text>

  <rect x="1240" y="280" width="1040" height="1080" rx="18" ry="18" fill="none" stroke="#555555" stroke-width="3"/>
  <text x="1760" y="340" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="36" font-weight="700" fill="#000000">
    Abnormal Example
  </text>
  <text x="1760" y="392" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="28" fill="#555555">
    Effusion
  </text>
  <rect x="1370" y="445" width="780" height="820" fill="none" stroke="#888888" stroke-width="2"/>
  <image x="1370" y="445" width="780" height="820" preserveAspectRatio="xMidYMid meet" xlink:href="{abnormal_href}"/>
  <text x="1760" y="1325" text-anchor="middle" font-family="Arial, Calibri, sans-serif" font-size="28" fill="#000000">
    Binary label: Abnormal (1)
  </text>
"""
    write_svg(POSTER_DIR / "dataset_example_poster.svg", 2400, 1500, body)


def main() -> None:
    roc_svg()
    gradcam_svg()
    composite_svg()
    dataset_example_svg()
    print(f"saved SVG files to: {POSTER_DIR}")


if __name__ == "__main__":
    main()
