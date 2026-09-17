"""
Shared font utilities for all image renderers.

Centralises script detection, font loading, text measurement, and wrap-width
estimation so that every renderer handles CJK, Devanagari, and Latin text
correctly without duplicating logic.
"""
import os
import textwrap
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_FONTS_DIR = Path(__file__).resolve().parents[3] / "fonts"

SCRIPT_FONTS: dict[str, list[str]] = {
    "cjk": [
        str(PROJECT_FONTS_DIR / "NotoSansCJKsc-Regular.otf"),
        "NotoSansCJK-Regular.ttc",
        "NotoSansSC-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ],
    "devanagari": [
        str(PROJECT_FONTS_DIR / "NotoSansDevanagari-Regular.ttf"),
        "NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    ],
}

LATIN_FONTS = [
    str(PROJECT_FONTS_DIR / "DejaVuSans.ttf"),
    "DejaVuSans.ttf",
]

MONOSPACE_FONTS = [
    str(PROJECT_FONTS_DIR / "DejaVuSansMono-Bold.ttf"),
    "DejaVuSansMono-Bold.ttf",
]

GRAPHVIZ_FONT_FOR_SCRIPT = {
    "cjk": "Noto Sans CJK SC",
    "devanagari": "Noto Sans Devanagari",
    "latin": "Times-Roman",
}


def detect_script(text: str) -> str:
    """Detect the dominant non-Latin script in *text*.

    Scans the first 500 non-ASCII characters (from the first 2 000 chars)
    and returns ``"cjk"``, ``"devanagari"``, or ``"latin"``.
    """
    cjk = deva = 0
    sample = [ch for ch in text[:2000] if ord(ch) > 127][:500]
    for ch in sample:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
            cjk += 1
        elif 0x0900 <= cp <= 0x097F:
            deva += 1
    if cjk > deva and cjk > 0:
        return "cjk"
    if deva > 0:
        return "devanagari"
    return "latin"


def load_font(
    candidates: list[str],
    size: int,
    explicit_path: Optional[str] = None,
) -> ImageFont.FreeTypeFont:
    """Try *candidates* in order and return the first that loads.

    Raises ``RuntimeError`` if none of the candidates can be loaded.
    No silent fallback — missing fonts must be fixed, not papered over.
    """
    if explicit_path and os.path.exists(explicit_path):
        return ImageFont.truetype(explicit_path, size)
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    raise RuntimeError(
        f"No font found from candidates: {candidates}. "
        f"Install the required fonts in {PROJECT_FONTS_DIR}/"
    )


def get_font_for_text(
    text: str,
    size: int,
    preferred_fonts: list[str],
    explicit_path: Optional[str] = None,
    cache: Optional[dict] = None,
) -> ImageFont.FreeTypeFont:
    """Return a font that can render *text*.

    Uses :func:`detect_script` to choose CJK / Devanagari fonts when the
    text requires them, falling back to *preferred_fonts* for Latin text.
    An optional *cache* dict avoids repeated filesystem probes.
    """
    script = detect_script(text)
    cache_key = f"{script}_{size}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    if script != "latin":
        candidates = SCRIPT_FONTS.get(script, [])
        font = load_font(candidates, size, explicit_path)
    else:
        font = load_font(preferred_fonts, size, explicit_path)

    if cache is not None:
        cache[cache_key] = font
    return font


def estimate_wrap_width(
    font: ImageFont.FreeTypeFont,
    available_width: int,
    text: str = "",
) -> int:
    """Estimate a character-count wrap width for *font* inside *available_width* px."""
    script = detect_script(text) if text else "latin"
    try:
        if script == "cjk":
            avg_w = font.getlength("\u4e00")
        elif script == "devanagari":
            avg_w = font.getlength("\u0905")
        else:
            avg_w = font.getlength("x")
    except (AttributeError, OSError):
        avg_w = getattr(font, "size", 14) * 0.6
    return max(10, int(available_width / avg_w)) if avg_w > 0 else 30


def measure_text_height(
    wrapped_text: str,
    font: ImageFont.FreeTypeFont,
    spacing: int = 4,
) -> int:
    """Return the pixel height of *wrapped_text* when rendered with *font*."""
    tmp = Image.new("RGB", (1, 1))
    dr = ImageDraw.Draw(tmp)
    bbox = dr.textbbox((0, 0), wrapped_text, font=font, spacing=spacing)
    return bbox[3] - bbox[1]


def auto_fit_font(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_height: int,
    spacing: int,
    padding: int,
    min_font_size: int = 10,
    max_iterations: int = 6,
) -> tuple[ImageFont.FreeTypeFont, str, int]:
    """Iteratively shrink *font* until *text* fits within the given box.

    Returns ``(fitted_font, wrapped_text, final_font_size)``.
    """
    font_size = font.size if hasattr(font, "size") else 28
    available_w = max_width - 2 * padding
    available_h = max_height - 2 * padding

    wrap_width = estimate_wrap_width(font, available_w, text)
    wrapped = textwrap.fill(text, width=wrap_width)
    text_h = measure_text_height(wrapped, font, spacing)

    if text_h <= available_h:
        return font, wrapped, font_size

    original_size = font_size
    for _ in range(max_iterations):
        scale = available_h / text_h * 0.90
        font_size = max(min_font_size, int(font_size * scale))
        font = ImageFont.truetype(font.path, font_size)
        wrap_width = estimate_wrap_width(font, available_w, text)
        wrapped = textwrap.fill(text, width=wrap_width)
        text_h = measure_text_height(wrapped, font, spacing)
        if text_h <= available_h or font_size <= min_font_size:
            break

    logger.info(
        f"Auto-fit: {original_size}pt → {font_size}pt "
        f"(text_h={text_h}px, limit={available_h}px)")
    return font, wrapped, font_size
