"""
Plain image renderer: renders text directly onto a clean image.

The simplest possible text-to-image renderer with no prompt engineering,
no numbered steps, no special formatting. This is the purest baseline
for isolating the modality effect (text vs. image-rendered text).

Uses sensible typographic defaults for maximum readability:
  - Sans-serif font (Arial/DejaVuSans) at readable size
  - Script-aware font selection: CJK → Noto Sans CJK, Devanagari → Noto Sans Devanagari
  - Black text on white background
  - Automatic word wrapping within padded area
"""
import textwrap
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from ..base_image_renderer import BaseImageRenderer
from ..font_utils import (
    LATIN_FONTS, detect_script, estimate_wrap_width, get_font_for_text,
    measure_text_height,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FONT_SIZE = 28
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 768
DEFAULT_BG_COLOR = "#FFFFFF"
DEFAULT_FG_COLOR = "#000000"
DEFAULT_PADDING = 40
DEFAULT_LINE_SPACING = 8
MAX_IMAGE_HEIGHT = 7680  # Stay under Claude's 8000px API limit
# Per-page height budget for the paginating renderer. Fixed font + split
# overflow across multiple images (kept well under MAX_IMAGE_HEIGHT). Taller
# pages → fewer images per prompt → less max-len pressure (a 13k-char
# code_attack lands at ~2–3 images here instead of ~6 at 1536px).
DEFAULT_PAGE_HEIGHT = 4096


class PlainImageRenderer(BaseImageRenderer):
    """
    Plain text-on-image renderer — the simplest baseline.

    Renders the full prompt text onto a white image with automatic
    word wrapping. No numbered steps, no prompt tricks, no special
    formatting. Used as the clean baseline for encoding × modality
    experiments where we want to measure the pure modality gap.

    Font selection is script-aware: if text contains CJK or Devanagari
    characters, a Noto Sans font with those glyphs is used instead of
    the default Latin font. Font files are resolved from the project
    ``fonts/`` directory first, then system paths.
    """

    def __init__(
        self,
        font_size: int = DEFAULT_FONT_SIZE,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        bg_color: str = DEFAULT_BG_COLOR,
        fg_color: str = DEFAULT_FG_COLOR,
        padding: int = DEFAULT_PADDING,
        line_spacing: int = DEFAULT_LINE_SPACING,
        page_height: int = DEFAULT_PAGE_HEIGHT,
        font_path: Optional[str] = None,
        # Degradation params (passed to base)
        blur_radius: float = 0,
        jpeg_quality: int = 100,
        noise_std: float = 0,
    ):
        super().__init__(blur_radius=blur_radius, jpeg_quality=jpeg_quality, noise_std=noise_std)
        self.font_size = font_size
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.padding = padding
        self.line_spacing = line_spacing
        self.page_height = page_height
        self._explicit_font_path = font_path
        self._font_cache: dict[str, ImageFont.FreeTypeFont] = {}
        self._latin_font = get_font_for_text(
            "x", font_size, LATIN_FONTS, font_path, self._font_cache)
        self._wrap_width = estimate_wrap_width(
            self._latin_font, self.width - 2 * self.padding)

    def _get_font(self, text: str) -> ImageFont.FreeTypeFont:
        return get_font_for_text(
            text, self.font_size, LATIN_FONTS,
            self._explicit_font_path, self._font_cache)

    def _render_clean(self, text: str) -> Image.Image:
        """Render text onto an auto-sized image (fixed width, height fits text).

        If the rendered image would exceed MAX_IMAGE_HEIGHT, font size is
        reduced proportionally so all content fits within the limit.
        """
        font = self._get_font(text)
        font_size = self.font_size
        wrap_width = estimate_wrap_width(
            font, self.width - 2 * self.padding, text)

        wrapped = textwrap.fill(text, width=wrap_width)
        text_h = measure_text_height(wrapped, font, self.line_spacing)
        auto_h = max(100, text_h + 2 * self.padding)

        if auto_h > MAX_IMAGE_HEIGHT:
            original_h = auto_h
            min_font = 10
            for _ in range(3):
                scale = (MAX_IMAGE_HEIGHT - 2 * self.padding) / text_h * 0.92
                font_size = max(min_font, int(font_size * scale))
                font = ImageFont.truetype(font.path, font_size)
                wrap_width = estimate_wrap_width(
                    font, self.width - 2 * self.padding, text)
                wrapped = textwrap.fill(text, width=wrap_width)
                text_h = measure_text_height(wrapped, font, self.line_spacing)
                auto_h = max(100, text_h + 2 * self.padding)
                if auto_h <= MAX_IMAGE_HEIGHT or font_size <= min_font:
                    break
            logger.info(
                f"Image would be {original_h}px tall (limit {MAX_IMAGE_HEIGHT}px); "
                f"reduced font {self.font_size}pt→{font_size}pt, final height {auto_h}px")

        height = auto_h if self.height == 0 else min(self.height, auto_h) \
            if auto_h < self.height else auto_h

        im = Image.new("RGB", (self.width, height), self.bg_color)
        dr = ImageDraw.Draw(im)
        dr.text(
            (self.padding, self.padding),
            wrapped,
            fill=self.fg_color,
            font=font,
            spacing=self.line_spacing,
        )
        return im

    def _render_pages(self, text: str):
        """Render text across one OR MORE images at a FIXED font size.

        Unlike `_render_clean` (which shrinks the font to cram everything into
        one image, destroying OCR fidelity for long content), this keeps
        `self.font_size` constant and paginates: wrap once, then pack whole
        lines into successive pages, each auto-fit to its content but bounded
        by `self.page_height`. Short prompts yield a single page.
        """
        font = self._get_font(text)
        wrap_width = estimate_wrap_width(
            font, self.width - 2 * self.padding, text)
        wrapped = textwrap.fill(text, width=wrap_width)
        lines = wrapped.split("\n")

        # Height of one line (incl. inter-line spacing) at the fixed font.
        line_h = max(1, measure_text_height("Ag", font, self.line_spacing)
                     + self.line_spacing)
        content_h = max(line_h, self.page_height - 2 * self.padding)
        lines_per_page = max(1, content_h // line_h)

        pages = []
        for start in range(0, len(lines), lines_per_page):
            chunk = "\n".join(lines[start:start + lines_per_page])
            text_h = measure_text_height(chunk, font, self.line_spacing)
            page_h = max(100, text_h + 2 * self.padding)
            im = Image.new("RGB", (self.width, page_h), self.bg_color)
            dr = ImageDraw.Draw(im)
            dr.text(
                (self.padding, self.padding), chunk,
                fill=self.fg_color, font=font, spacing=self.line_spacing)
            pages.append(im)

        if not pages:  # empty/whitespace text → fall back to a single blank page
            pages = [self._render_clean(text)]
        logger.info(
            f"Paginated {len(text)} chars into {len(pages)} image(s) "
            f"at fixed font {self.font_size}pt (~{lines_per_page} lines/page)")
        return pages

    def get_config(self) -> dict:
        config = {
            "renderer_type": "plain",
            "font_size": self.font_size,
            "width": self.width,
            "height": self.height,
            "page_height": self.page_height,
            "max_image_height": MAX_IMAGE_HEIGHT,
            "bg_color": self.bg_color,
            "fg_color": self.fg_color,
            "padding": self.padding,
            "line_spacing": self.line_spacing,
            "wrap_width": self._wrap_width,
        }
        degradation = self.get_degradation_config()
        if degradation:
            config["degradation"] = degradation
        return config
