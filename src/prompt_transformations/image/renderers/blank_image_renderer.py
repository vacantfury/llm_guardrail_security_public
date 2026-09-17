"""
Blank image renderer — outputs a pure blank canvas regardless of input text.

Used by the `ir_blank` transformation for the "text+blank" destroyer-mode
ablation: encoded text remains in the text channel, image channel carries
only a blank PNG. Tests whether the destroyer effect depends on the rendered
text image, or just on the *presence* of an image companion.
"""
from typing import Optional

from PIL import Image

from ..base_image_renderer import BaseImageRenderer

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 768
DEFAULT_BG_COLOR = "#FFFFFF"


class BlankImageRenderer(BaseImageRenderer):
    """Returns a fixed-size blank canvas. Ignores input text."""

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        bg_color: str = DEFAULT_BG_COLOR,
        blur_radius: float = 0,
        jpeg_quality: int = 100,
        noise_std: float = 0,
    ):
        super().__init__(
            blur_radius=blur_radius, jpeg_quality=jpeg_quality, noise_std=noise_std)
        self.width = width
        self.height = height
        self.bg_color = bg_color

    def _render_clean(self, text: str) -> Image.Image:
        # text is intentionally ignored — this renderer always returns blank.
        return Image.new("RGB", (self.width, self.height), self.bg_color)

    def get_config(self) -> dict:
        config = {
            "renderer_type": "blank",
            "width": self.width,
            "height": self.height,
            "bg_color": self.bg_color,
        }
        degradation = self.get_degradation_config()
        if degradation:
            config["degradation"] = degradation
        return config
