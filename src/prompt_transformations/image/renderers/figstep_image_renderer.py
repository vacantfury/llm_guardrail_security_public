"""
FigStep image renderer: typographic text-in-image rendering.

Canvas height auto-sizes to fit the full text content. The original
FigStep paper used short prompts on a fixed 760×760 canvas; our encoded
prompts are much longer, so auto-height is necessary to avoid truncation.

Reference: other_repos/FigStep/src/generate_prompts.py
"""
import textwrap
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from ..base_image_renderer import BaseImageRenderer
from ..font_utils import (
    MONOSPACE_FONTS, estimate_wrap_width, get_font_for_text,
    measure_text_height,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FONT_SIZE = 80
DEFAULT_WIDTH = 760
DEFAULT_BG_COLOR = "#FFFFFF"
DEFAULT_FG_COLOR = "#000000"
DEFAULT_X = 20
DEFAULT_Y = 10
DEFAULT_SPACING = 11
DEFAULT_WRAP_WIDTH = 15
DEFAULT_NUM_STEPS = 3
MAX_IMAGE_HEIGHT = 7680


class FigstepImageRenderer(BaseImageRenderer):
    """
    FigStep-style renderer: large monospace text on white background.

    1. **Auto-height canvas**: height grows to fit all text content,
       preventing the truncation that made long encoded prompts invisible.
    2. **Script-aware fonts**: CJK and Devanagari text uses Noto Sans
       fonts instead of FreeMonoBold (which lacks those glyphs).

    Attributes:
        font_size: Font size in points (default: 80, matching original)
        width: Image width in pixels (default: 760)
        bg_color: Background color (default: white)
        fg_color: Text color (default: black)
        text_x: X offset for text starting position
        text_y: Y offset for text starting position
        spacing: Line spacing in pixels
        wrap_width: Character count for textwrap (auto-computed for non-Latin)
        num_steps: Number of empty numbered steps to append (0 to disable)
        font_path: Optional path to .ttf font file
    """

    def __init__(
        self,
        font_size: int = DEFAULT_FONT_SIZE,
        width: int = DEFAULT_WIDTH,
        bg_color: str = DEFAULT_BG_COLOR,
        fg_color: str = DEFAULT_FG_COLOR,
        text_x: int = DEFAULT_X,
        text_y: int = DEFAULT_Y,
        spacing: int = DEFAULT_SPACING,
        wrap_width: int = DEFAULT_WRAP_WIDTH,
        num_steps: int = DEFAULT_NUM_STEPS,
        font_path: Optional[str] = None,
        # Dynamic adaptation params
        max_width: int = 2400,
        min_font_size: int = 40,
        # Degradation params (passed to base)
        blur_radius: float = 0,
        jpeg_quality: int = 100,
        noise_std: float = 0,
        max_aspect_ratio: float = 3.0,
    ):
        super().__init__(
            blur_radius=blur_radius, jpeg_quality=jpeg_quality,
            noise_std=noise_std, max_aspect_ratio=max_aspect_ratio,
        )
        self.font_size = font_size
        self.width = width
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.text_x = text_x
        self.text_y = text_y
        self.spacing = spacing
        self.wrap_width = wrap_width
        self.num_steps = num_steps
        self._font_path = font_path
        self._max_width = max_width
        self._min_font_size = min_font_size
        self._font_cache: dict[str, ImageFont.FreeTypeFont] = {}

    def _get_font(self, text: str) -> ImageFont.FreeTypeFont:
        return get_font_for_text(
            text, self.font_size, MONOSPACE_FONTS,
            self._font_path, self._font_cache)

    def _format_text(self, text: str, wrap_width: int) -> str:
        """Wrap text and append numbered empty steps."""
        text = text.removesuffix("\n")
        text = textwrap.fill(text, width=wrap_width)
        if self.num_steps > 0:
            for idx in range(1, self.num_steps + 1):
                text += f"\n{idx}. "
        return text

    def _render_clean(self, text: str) -> Image.Image:
        """Render text with dynamic canvas adaptation for aspect ratio.

        Strategy:
        1. Start with configured width and font_size.
        2. Measure height — if aspect ratio exceeds max_aspect_ratio, widen
           canvas (up to max_width) to reflow text into fewer lines.
        3. If still too tall after max width, shrink font (down to min_font_size).
        4. Clamp height to MAX_IMAGE_HEIGHT as a final safety net.
        """
        font_size = self.font_size
        canvas_w = self.width
        font = self._get_font(text)

        for iteration in range(8):
            available_w = canvas_w - 2 * self.text_x
            wrap_width = estimate_wrap_width(font, available_w, text)
            formatted = self._format_text(text, wrap_width)
            text_h = measure_text_height(formatted, font, self.spacing)
            canvas_h = max(200, text_h + self.text_y + self.text_x)

            aspect = canvas_h / canvas_w if canvas_w > 0 else 999
            if aspect <= self.max_aspect_ratio:
                break

            min_w = self._width_for_aspect(canvas_h)
            if min_w <= self._max_width and canvas_w < min_w:
                canvas_w = min(min_w, self._max_width)
                continue

            canvas_w = self._max_width
            if font_size > self._min_font_size:
                font_size = max(self._min_font_size, int(font_size * 0.75))
                font = ImageFont.truetype(font.path, font_size)
                continue

            break

        if canvas_h > MAX_IMAGE_HEIGHT:
            canvas_h = MAX_IMAGE_HEIGHT

        if canvas_w != self.width or font_size != self.font_size:
            logger.debug(
                f"FigStep adaptive: width {self.width}→{canvas_w}px, "
                f"font {self.font_size}→{font_size}pt, "
                f"size {canvas_w}x{canvas_h}px (aspect {canvas_h/canvas_w:.1f}:1)")

        im = Image.new("RGB", (canvas_w, canvas_h), self.bg_color)
        dr = ImageDraw.Draw(im)
        dr.text(
            (self.text_x, self.text_y),
            formatted,
            fill=self.fg_color,
            font=font,
            spacing=self.spacing,
        )
        return im

    def get_config(self) -> dict:
        config = {
            "renderer_type": "figstep",
            "font_size": self.font_size,
            "width": self.width,
            "bg_color": self.bg_color,
            "fg_color": self.fg_color,
            "text_x": self.text_x,
            "text_y": self.text_y,
            "spacing": self.spacing,
            "wrap_width": self.wrap_width,
            "num_steps": self.num_steps,
        }
        degradation = self.get_degradation_config()
        if degradation:
            config["degradation"] = degradation
        return config
