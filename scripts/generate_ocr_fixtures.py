"""Generate test fixture images for OCR E2E tests.

Creates small JPEG and PNG images with clear black text on white
background using PyMuPDF (fitz). Run once to produce the static
fixture files consumed by tests/test_ocr_e2e.py.

Usage:
    uv run python scripts/generate_ocr_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import fitz

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ocr"


def _render_text_image(text: str, width: int = 400, height: int = 100) -> fitz.Pixmap:
    """Render *text* centered on a white background and return a Pixmap."""
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)

    # White background is default; insert text in black
    font_size = 24
    text_point = fitz.Point(20, height / 2 + font_size / 3)
    page.insert_text(text_point, text, fontsize=font_size, color=(0, 0, 0))

    # Render to pixmap at 150 DPI for clear OCR
    pix = page.get_pixmap(dpi=150)
    doc.close()
    return pix


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # JPEG fixture
    pix = _render_text_image("Hello from JPEG")
    jpeg_path = FIXTURES_DIR / "hello_jpeg.jpg"
    pix.save(str(jpeg_path))
    print(f"Created {jpeg_path} ({jpeg_path.stat().st_size} bytes)")

    # PNG fixture
    pix = _render_text_image("Hello from PNG")
    png_path = FIXTURES_DIR / "hello_png.png"
    pix.save(str(png_path))
    print(f"Created {png_path} ({png_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
