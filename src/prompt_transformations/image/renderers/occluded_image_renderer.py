"""
Occluded-text renderer — Adversarial Smuggling "Perceptual Blindness" pathway.

Renders the payload text (via the plain renderer) then overlays regular
occluding bars ("venetian-blind" stripes) that cover part of every glyph. A
human reads through the gaps by pattern completion; an OCR/decode safety check
loses characters where the bars fall — the payload is fully present yet
filter-hostile, which is exactly the recover/decode gap our defense targets.

Reference: other_repos/smugglebench (Li et al., ACL 2026), Occluded-Text
technique. This is a synthetic approximation — the released benchmark ships
curated real-world photos, not construction code, so we reproduce the *effect*
(regular partial occlusion) with a deterministic PIL overlay.
"""
from PIL import Image, ImageDraw

from .plain_image_renderer import PlainImageRenderer


class OccludedImageRenderer(PlainImageRenderer):
    """Plain text-on-canvas with a regular occluding-bar overlay.

    Extra params (on top of the plain renderer's):
        occlusion_orientation: "horizontal" | "vertical" bar direction.
        occlusion_color: bar color (default = background, so bars "erase"
            bands of text rather than adding foreign ink).
        occlusion_opacity: 0..1 bar alpha (1.0 = fully hides the covered band).
        bar_thickness: bar width in px (the occluded band).
        bar_gap: clear gap in px between consecutive bars (the readable band).
    """

    def __init__(
        self,
        occlusion_orientation: str = "horizontal",
        occlusion_color: str = None,
        occlusion_opacity: float = 0.85,
        bar_thickness: int = 12,
        bar_gap: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.occlusion_orientation = occlusion_orientation
        # Default to the background color so bars subtract text rather than
        # adding contrasting ink (a cleaner "blinds" occlusion).
        self.occlusion_color = occlusion_color or self.bg_color
        self.occlusion_opacity = float(occlusion_opacity)
        self.bar_thickness = int(bar_thickness)
        self.bar_gap = int(bar_gap)

    def _occlude(self, img: Image.Image) -> Image.Image:
        base = img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        alpha = max(0, min(255, int(round(self.occlusion_opacity * 255))))
        rgb = Image.new("RGB", (1, 1), self.occlusion_color).getpixel((0, 0))
        fill = (rgb[0], rgb[1], rgb[2], alpha)
        step = max(1, self.bar_thickness + self.bar_gap)
        w, h = base.size
        if self.occlusion_orientation == "vertical":
            for x in range(0, w, step):
                draw.rectangle([x, 0, x + self.bar_thickness, h], fill=fill)
        else:
            for y in range(0, h, step):
                draw.rectangle([0, y, w, y + self.bar_thickness], fill=fill)
        return Image.alpha_composite(base, overlay).convert("RGB")

    def _render_clean(self, text: str) -> Image.Image:
        return self._occlude(super()._render_clean(text))

    def _render_pages(self, text: str):
        return [self._occlude(p) for p in super()._render_pages(text)]

    def get_config(self) -> dict:
        config = super().get_config()
        config["renderer_type"] = "occluded"
        config["occlusion"] = {
            "orientation": self.occlusion_orientation,
            "color": self.occlusion_color,
            "opacity": self.occlusion_opacity,
            "bar_thickness": self.bar_thickness,
            "bar_gap": self.bar_gap,
        }
        return config
