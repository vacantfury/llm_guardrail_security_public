"""
FC-Flowchart image renderer: Graphviz-based flowchart rendering.

FC-Attack uses a fine-tuned step-description generator to split harmful queries
into numbered steps. Since our experiment tests rendering (not decomposition),
we render the FULL prompt text in the goal node. This isolates the rendering
variable from the decomposition confound.

Requires: graphviz Python package and Graphviz system binary.
  pip install graphviz
  brew install graphviz  (macOS) / apt install graphviz (Linux)
"""
import textwrap
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from io import BytesIO

from PIL import Image

from ..base_image_renderer import BaseImageRenderer
from ..font_utils import GRAPHVIZ_FONT_FOR_SCRIPT, detect_script
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DPI = 600
DEFAULT_LAYOUT = "vertical"  # vertical | horizontal | tortuous


class FCFlowchartImageRenderer(BaseImageRenderer):
    """
    FC-Attack-style flowchart renderer using Graphviz.
    
    Renders text inside a Graphviz directed graph with:
    - Goal node (ellipse) containing the full prompt
    - Optional numbered step nodes (box) if steps are provided
    - Configurable layout direction
    
    Attributes:
        layout: Flowchart layout — "vertical" (TB), "horizontal" (LR), "tortuous" (S-shaped)
        dpi: Dots per inch for rendering (default: 600, matching FC-Attack)
        wrap_width: Character count for text wrapping inside nodes
        font_name: Graphviz font name
        font_size: Graphviz font size string
    """
    
    def __init__(
        self,
        layout: str = DEFAULT_LAYOUT,
        dpi: int = DEFAULT_DPI,
        wrap_width: int = 30,
        font_name: str = "Times-Roman",
        font_size: str = "14",
        # Dynamic adaptation params
        long_text_threshold: int = 500,
        long_text_dpi: int = 300,
        max_wrap_width: int = 100,
        auto_layout_for_long: bool = True,
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
        self.layout = layout
        self.dpi = dpi
        self.wrap_width = wrap_width
        self.font_name = font_name
        self.font_size = font_size
        self._long_text_threshold = long_text_threshold
        self._long_text_dpi = long_text_dpi
        self._max_wrap_width = max_wrap_width
        self._auto_layout_for_long = auto_layout_for_long
        
        # Verify graphviz is available
        try:
            import graphviz
            self._graphviz = graphviz
        except ImportError:
            raise ImportError(
                "FC-Flowchart renderer requires 'graphviz' package. "
                "Install with: pip install graphviz && brew install graphviz"
            )
    
    def _compute_adaptive_params(self, text: str) -> tuple[int, int, str]:
        """Return (effective_wrap_width, effective_dpi, effective_layout) for *text*.

        For long text (above threshold), widens wrap_width and reduces DPI to
        produce wider, shorter images that stay within max_aspect_ratio. Also
        switches layout to horizontal if auto_layout_for_long is enabled.
        """
        text_len = len(text)
        if text_len <= self._long_text_threshold:
            return self.wrap_width, self.dpi, self.layout

        scale = min(text_len / self._long_text_threshold, 4.0)
        effective_wrap = min(int(self.wrap_width * scale), self._max_wrap_width)
        effective_dpi = self._long_text_dpi
        effective_layout = "horizontal" if self._auto_layout_for_long else self.layout

        logger.debug(
            f"FC-Flowchart adaptive: len={text_len}, "
            f"wrap {self.wrap_width}→{effective_wrap}, "
            f"dpi {self.dpi}→{effective_dpi}, layout→{effective_layout}")
        return effective_wrap, effective_dpi, effective_layout

    def _wrap_node_text(self, text: str, wrap_width: int | None = None) -> str:
        """Wrap text for Graphviz node labels (newline-separated).

        Handles CJK text (no spaces) by inserting breaks every wrap_width chars.
        """
        ww = wrap_width if wrap_width is not None else self.wrap_width
        if self._is_cjk_dominant(text):
            lines = [text[i:i + ww] for i in range(0, len(text), ww)]
            return "\\n".join(lines)
        return textwrap.fill(text, width=ww).replace("\n", "\\n")

    @staticmethod
    def _is_cjk_dominant(text: str) -> bool:
        """Check if text is predominantly CJK (no spaces to split on)."""
        if not text:
            return False
        cjk_count = sum(1 for ch in text if unicodedata.category(ch).startswith("Lo"))
        return cjk_count > len(text) * 0.3
    
    def _build_vertical(self, text: str, dpi: int, wrap_width: int) -> "graphviz.Digraph":
        """Build vertical (top-to-bottom) flowchart — FC-Attack's default."""
        dot = self._graphviz.Digraph()
        dot.attr(dpi=str(dpi))
        
        wrapped = self._wrap_node_text(text, wrap_width)
        dot.node("goal", wrapped, shape="ellipse")
        
        return dot
    
    def _build_horizontal(self, text: str, dpi: int, wrap_width: int) -> "graphviz.Digraph":
        """Build horizontal (left-to-right) flowchart."""
        dot = self._graphviz.Digraph()
        dot.attr(dpi=str(dpi), rankdir="LR")
        
        wrapped = self._wrap_node_text(text, wrap_width)
        dot.node("goal", wrapped, shape="ellipse")
        
        return dot
    
    def _build_tortuous(self, text: str, dpi: int, wrap_width: int) -> "graphviz.Digraph":
        """Build S-shaped (tortuous) flowchart — FC-Attack's zigzag layout."""
        dot = self._graphviz.Digraph()
        dot.attr(dpi=str(dpi), rankdir="TB")

        # Split text into chunks for zigzag nodes
        if self._is_cjk_dominant(text):
            chunk_size = max(1, len(text) // 3)
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        else:
            words = text.split()
            chunk_size = max(1, len(words) // 3)
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i:i + chunk_size]))
        
        # Create nodes in rows of ~3, alternating direction
        prev_node = None
        for i, chunk in enumerate(chunks[:6]):  # Max 6 chunks
            node_id = f"step_{i}"
            wrapped = self._wrap_node_text(chunk, wrap_width)
            shape = "ellipse" if i == 0 else "box"
            dot.node(node_id, wrapped, shape=shape, style="filled", fillcolor="white")
            if prev_node:
                dot.edge(prev_node, node_id)
            prev_node = node_id
        
        return dot
    
    def _render_clean(self, text: str) -> Image.Image:
        """Render text as a Graphviz flowchart with adaptive parameters.

        For long text, widens wrap_width, reduces DPI, and optionally switches
        to horizontal layout to maintain readable aspect ratios.
        """
        effective_wrap, effective_dpi, effective_layout = self._compute_adaptive_params(text)

        builders = {
            "vertical": self._build_vertical,
            "horizontal": self._build_horizontal,
            "tortuous": self._build_tortuous,
        }

        if effective_layout not in builders:
            raise ValueError(f"Unknown layout '{effective_layout}'. Options: {list(builders.keys())}")

        dot = builders[effective_layout](text, effective_dpi, effective_wrap)

        script = detect_script(text)
        font_name = GRAPHVIZ_FONT_FOR_SCRIPT.get(script, self.font_name)
        dot.attr("node", fontname=font_name, fontsize=self.font_size)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(dot.pipe, format="png")
                png_bytes = future.result(timeout=30)
        except TimeoutError:
            logger.warning(
                f"Graphviz rendering timed out (30s) for text[:50]={text[:50]!r}. "
                f"Falling back to blank image.")
            return Image.new("RGB", (800, 600), "white")
        except Exception as e:
            logger.warning(
                f"Graphviz rendering failed: {e}. "
                f"Falling back to blank image for text[:50]={text[:50]!r}")
            return Image.new("RGB", (800, 600), "white")

        img = Image.open(BytesIO(png_bytes)).convert("RGB")

        w, h = img.size
        if h > 0 and h / w > self.max_aspect_ratio:
            target_w = self._width_for_aspect(h)
            img = img.resize((target_w, h), Image.LANCZOS)
            logger.debug(
                f"FC-Flowchart post-resize: {w}x{h} → {target_w}x{h} "
                f"(aspect {h/target_w:.1f}:1)")

        return img
    
    def get_config(self) -> dict:
        """Return renderer config for saving alongside output."""
        config = {
            "renderer_type": "fc_flowchart",
            "layout": self.layout,
            "dpi": self.dpi,
            "wrap_width": self.wrap_width,
            "font_name": self.font_name,
            "font_size": self.font_size,
        }
        degradation = self.get_degradation_config()
        if degradation:
            config["degradation"] = degradation
        return config
