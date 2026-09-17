"""
Base image renderer — abstract interface for all rendering strategies.

All renderers must implement render(), render_to_file(), render_to_base64(),
and get_config(). This mirrors the BaseEncoder pattern from text_encoding.

Degradation (blur, JPEG compression, noise) is handled at the base class level
and applied as post-processing after any renderer's render() call.
"""
import base64
import os
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseImageRenderer(ABC):
    """
    Abstract base class for all image renderers.
    
    Subclasses implement _render_clean() to produce a PIL Image from text.
    The base class wraps it with optional degradation (blur, JPEG, noise).
    render_to_file() and render_to_base64() are provided for free.
    
    Degradation parameters (set via __init__ or YAML config):
        blur_radius: Gaussian blur radius (0 = no blur)
        jpeg_quality: JPEG re-compression quality (100 = no compression, lower = worse)
        noise_std: Gaussian noise standard deviation (0 = no noise)
    """
    
    def __init__(
        self,
        blur_radius: float = 0,
        jpeg_quality: int = 100,
        noise_std: float = 0,
        max_aspect_ratio: float = 3.0,
    ):
        self.blur_radius = blur_radius
        self.jpeg_quality = jpeg_quality
        self.noise_std = noise_std
        self.max_aspect_ratio = max_aspect_ratio

    def _width_for_aspect(self, height: int) -> int:
        """Minimum width to satisfy max_aspect_ratio given a height."""
        return max(1, int(height / self.max_aspect_ratio))
    
    @abstractmethod
    def _render_clean(self, text: str) -> Image.Image:
        """
        Render text onto a clean image (no degradation).
        
        Subclasses implement this instead of render().
        
        Args:
            text: The text to render
            
        Returns:
            PIL Image object
        """
        pass
    
    def _apply_degradation(self, img: Image.Image) -> Image.Image:
        """
        Apply post-processing degradation to a rendered image.
        
        Applied in order: noise → blur → JPEG compression.
        Only applies steps where parameters are non-default.
        """
        # Gaussian noise
        if self.noise_std > 0:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, self.noise_std, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        
        # Gaussian blur
        if self.blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        
        # JPEG compression (re-encode and decode to introduce artifacts)
        if self.jpeg_quality < 100:
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=self.jpeg_quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")
        
        return img
    
    def render(self, text: str) -> Image.Image:
        """
        Render text onto an image, with optional degradation.
        
        Calls _render_clean() then applies degradation if configured.
        
        Args:
            text: The text to render
            
        Returns:
            PIL Image object (possibly degraded)
        """
        img = self._render_clean(text)

        if self.blur_radius > 0 or self.jpeg_quality < 100 or self.noise_std > 0:
            img = self._apply_degradation(img)

        return img

    def _render_pages(self, text: str) -> list[Image.Image]:
        """Render text onto one OR MORE clean images (no degradation).

        Default implementation is single-page (wraps _render_clean), so every
        existing renderer keeps its single-image behavior unchanged. Paginating
        renderers (plain) override this to keep a fixed font and split
        overflowing content across multiple images instead of shrinking the
        font to illegibility.
        """
        return [self._render_clean(text)]

    def render_pages(self, text: str) -> list[Image.Image]:
        """Render text to a list of images, applying degradation per page."""
        pages = self._render_pages(text)
        if self.blur_radius > 0 or self.jpeg_quality < 100 or self.noise_std > 0:
            pages = [self._apply_degradation(p) for p in pages]
        return pages

    def render_to_files(self, text: str, output_path: str) -> list[str]:
        """Render text and save to one or more PNG files.

        For a single page, saves exactly to ``output_path`` (back-compat with
        callers/filenames that assume one image). For N>1 pages, saves
        ``<stem>_p0<suffix>``, ``<stem>_p1<suffix>``, … next to ``output_path``.

        Returns:
            List of absolute paths to the saved PNG(s), in page order.
        """
        pages = self.render_pages(text)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if len(pages) == 1:
            pages[0].save(out, "PNG")
            logger.debug(f"Saved image: {out}")
            return [os.path.abspath(out)]
        paths: list[str] = []
        for i, page in enumerate(pages):
            page_path = out.with_name(f"{out.stem}_p{i}{out.suffix}")
            page.save(page_path, "PNG")
            paths.append(os.path.abspath(page_path))
        logger.debug(f"Saved {len(pages)} paginated images for {out.name}")
        return paths

    def render_to_file(self, text: str, output_path: str) -> str:
        """
        Render text and save to PNG file.
        
        Args:
            text: The text to render
            output_path: Path to save the PNG
            
        Returns:
            Absolute path to saved file
        """
        img = self.render(text)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG")
        logger.debug(f"Saved image: {output_path}")
        return os.path.abspath(output_path)
    
    def render_to_base64(self, text: str) -> str:
        """
        Render text and return as base64-encoded PNG string.
        
        Args:
            text: The text to render
            
        Returns:
            Base64-encoded PNG string
        """
        img = self.render(text)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def get_degradation_config(self) -> dict:
        """Return degradation parameters (included in all renderer configs)."""
        config = {}
        if self.blur_radius > 0:
            config["blur_radius"] = self.blur_radius
        if self.jpeg_quality < 100:
            config["jpeg_quality"] = self.jpeg_quality
        if self.noise_std > 0:
            config["noise_std"] = self.noise_std
        return config
    
    @abstractmethod
    def get_config(self) -> dict:
        """Return renderer config for saving alongside output."""
        pass
