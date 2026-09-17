"""
Constant image renderer — loads a fixed image file regardless of input text.

Used by the `ir_constant` transformation for the "text+constant-image"
destroyer-mode ablation: encoded text remains in the text channel, image
channel carries a fixed irrelevant image (e.g. a rabbit photo). Tests
whether the destroyer effect depends on the rendered text image, or just
on the presence of *any* image companion.

The image is configured by a single `image_path` parameter. To swap from
the default rabit.jpeg to a different decoy image (mountain, landscape, etc.)
just override `image_path` in the task config — no renderer code change.
"""
from pathlib import Path

from PIL import Image

from ..base_image_renderer import BaseImageRenderer


# Default bundled image (resolved relative to repo root).
_DEFAULT_IMAGE_REL = "src/prompt_transformations/image/images/rabit.jpeg"


def _resolve_image_path(image_path: str) -> Path:
    """Absolute → as-is; relative → relative to repo root."""
    p = Path(image_path)
    if p.is_absolute():
        return p
    # This file: src/prompt_transformations/image/renderers/constant_image_renderer.py
    # parents[0]=renderers, [1]=image, [2]=prompt_transformations, [3]=src, [4]=repo root.
    repo_root = Path(__file__).resolve().parents[4]
    return (repo_root / image_path).resolve()


class ConstantImageRenderer(BaseImageRenderer):
    """Returns a fixed image loaded from disk. Ignores input text."""

    def __init__(
        self,
        image_path: str = _DEFAULT_IMAGE_REL,
        blur_radius: float = 0,
        jpeg_quality: int = 100,
        noise_std: float = 0,
    ):
        super().__init__(
            blur_radius=blur_radius, jpeg_quality=jpeg_quality, noise_std=noise_std)
        resolved = _resolve_image_path(image_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"constant_image_renderer: image_path does not resolve to a file: "
                f"{image_path!r} → {resolved}")
        self._image_path = image_path
        self._resolved_path = resolved
        self._image = Image.open(resolved).convert("RGB")

    def _render_clean(self, text: str) -> Image.Image:
        # text is intentionally ignored — this renderer always returns the same image.
        return self._image.copy()

    def get_config(self) -> dict:
        config = {
            "renderer_type": "constant",
            "image_path": self._image_path,
            "resolved_path": str(self._resolved_path),
        }
        degradation = self.get_degradation_config()
        if degradation:
            config["degradation"] = degradation
        return config
