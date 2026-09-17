"""
FC-Typography image renderer: varied font/contrast rendering.

This renderer uses decorative fonts (Pacifico by default) for Latin text
and falls back to Noto Sans CJK / Devanagari for non-Latin scripts.
Canvas height auto-sizes to fit all content.

FC-Attack tested: Creepster, Fruktur Italic, Pacifico, Shojumaru,
UnifrakturMaguntia (all from Google Fonts, chosen for low readability).
"""
import os
import textwrap
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from ..base_image_renderer import BaseImageRenderer
from ..font_utils import (
    PROJECT_FONTS_DIR, SCRIPT_FONTS, detect_script, estimate_wrap_width,
    load_font, measure_text_height,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FONT_FAMILY = "Pacifico"
DEFAULT_FONT_SIZE = 40
DEFAULT_WIDTH = 1024
DEFAULT_BG_COLOR = "#FFFFFF"
DEFAULT_FG_COLOR = "#333333"
DEFAULT_PADDING = 30
DEFAULT_SPACING = 8
DEFAULT_WRAP_WIDTH = 40
MAX_IMAGE_HEIGHT = 7680

FC_ATTACK_FONTS = [
    "Pacifico",
    "Creepster",
    "Shojumaru",
    "UnifrakturMaguntia",
]


class FCTypographyImageRenderer(BaseImageRenderer):
    """
    FC-Attack-inspired typography renderer with varied fonts/contrast.

    Uses decorative / low-readability fonts that FC-Attack showed
    significantly affect ASR. For non-Latin text (CJK, Devanagari) it
    falls back to Noto Sans fonts so that glyphs render correctly.

    Canvas height auto-sizes to fit all text content.

    Attributes:
        font_family: Name of the font (searched on system)
        font_size: Font size in points
        width: Image width in pixels
        bg_color: Background color (can lower contrast)
        fg_color: Text color (can lower contrast)
        padding: Padding around text
        spacing: Line spacing
        wrap_width: Character count for textwrap
        font_path: Explicit path to .ttf file (overrides font_family search)
    """

    def __init__(
        self,
        font_family: str = DEFAULT_FONT_FAMILY,
        font_size: int = DEFAULT_FONT_SIZE,
        width: int = DEFAULT_WIDTH,
        bg_color: str = DEFAULT_BG_COLOR,
        fg_color: str = DEFAULT_FG_COLOR,
        padding: int = DEFAULT_PADDING,
        spacing: int = DEFAULT_SPACING,
        wrap_width: int = DEFAULT_WRAP_WIDTH,
        font_path: Optional[str] = None,
        # Degradation params (passed to base)
        blur_radius: float = 0,
        jpeg_quality: int = 100,
        noise_std: float = 0,
    ):
        super().__init__(blur_radius=blur_radius, jpeg_quality=jpeg_quality, noise_std=noise_std)
        self.font_family = font_family
        self.font_size = font_size
        self.width = width
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.padding = padding
        self.spacing = spacing
        self.wrap_width = wrap_width
        self._font_path = font_path
        self._font_cache: dict[str, ImageFont.FreeTypeFont] = {}

        self._latin_font = self._load_decorative_font(font_family, font_size, font_path)

    def _load_decorative_font(
        self, family: str, size: int, explicit_path: Optional[str],
    ) -> ImageFont.FreeTypeFont:
        """Load the decorative font (Pacifico etc.) for Latin text.

        Raises ``RuntimeError`` if the font cannot be found — no silent fallback.
        """
        if explicit_path and os.path.exists(explicit_path):
            logger.info(f"Loading font from path: {explicit_path}")
            return ImageFont.truetype(explicit_path, size)

        candidates = [
            str(PROJECT_FONTS_DIR / f"{family}-Regular.ttf"),
            str(PROJECT_FONTS_DIR / f"{family}.ttf"),
            f"{family}.ttf",
            f"{family}-Regular.ttf",
            f"{family}-regular.ttf",
        ]
        for candidate in candidates:
            try:
                font = ImageFont.truetype(candidate, size)
                logger.info(f"Loaded font: {candidate}")
                return font
            except (OSError, IOError):
                continue

        raise RuntimeError(
            f"Font '{family}' not found. Searched: {candidates}. "
            f"Download it to {PROJECT_FONTS_DIR}/"
        )

    def _get_font(self, text: str) -> ImageFont.FreeTypeFont:
        """Return script-appropriate font: decorative for Latin, Noto for CJK/Devanagari."""
        script = detect_script(text)
        if script == "latin":
            return self._latin_font

        cache_key = f"{script}_{self.font_size}"
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        candidates = SCRIPT_FONTS.get(script, [])
        font = load_font(candidates, self.font_size, self._font_path)
        self._font_cache[cache_key] = font
        return font

    def _render_clean(self, text: str) -> Image.Image:
        """Render text with typography variation and auto-height canvas."""
        font = self._get_font(text)
        available_w = self.width - 2 * self.padding
        wrap_width = estimate_wrap_width(font, available_w, text)

        wrapped = textwrap.fill(text, width=wrap_width)
        text_h = measure_text_height(wrapped, font, self.spacing)
        canvas_h = max(200, text_h + 2 * self.padding)

        if canvas_h > MAX_IMAGE_HEIGHT:
            original_h = canvas_h
            font_size = self.font_size
            min_font = 10
            for _ in range(6):
                avail_h = MAX_IMAGE_HEIGHT - 2 * self.padding
                scale = avail_h / text_h * 0.90
                font_size = max(min_font, int(font_size * scale))
                font = ImageFont.truetype(font.path, font_size)
                wrap_width = estimate_wrap_width(font, available_w, text)
                wrapped = textwrap.fill(text, width=wrap_width)
                text_h = measure_text_height(wrapped, font, self.spacing)
                canvas_h = max(200, text_h + 2 * self.padding)
                if canvas_h <= MAX_IMAGE_HEIGHT or font_size <= min_font:
                    break
            logger.info(
                f"FC-Typography auto-fit: {self.font_size}pt → {font_size}pt, "
                f"height {original_h} → {canvas_h}px")

        im = Image.new("RGB", (self.width, canvas_h), self.bg_color)
        dr = ImageDraw.Draw(im)
        dr.text(
            (self.padding, self.padding),
            wrapped,
            fill=self.fg_color,
            font=font,
            spacing=self.spacing,
        )
        return im

    def get_config(self) -> dict:
        config = {
            "renderer_type": "fc_typography",
            "font_family": self.font_family,
            "font_size": self.font_size,
            "width": self.width,
            "bg_color": self.bg_color,
            "fg_color": self.fg_color,
            "padding": self.padding,
            "spacing": self.spacing,
            "wrap_width": self.wrap_width,
        }
        degradation = self.get_degradation_config()
        if degradation:
            config["degradation"] = degradation
        return config
