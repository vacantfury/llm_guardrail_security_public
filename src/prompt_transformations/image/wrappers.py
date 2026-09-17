"""
Thin PromptTransformation wrappers around legacy BaseImageRenderer implementations.

Each class is a 3-line subclass that:
  - declares its canonical type_name (registry key, prefixed `ir_`)
  - points at its legacy renderer class

Naming convention: `ir_<short>` (image rendering, short style name) — matches
how experiment_results.md and proposal.md refer to renderer variants.

Configurable params (inherited from ImageRendererTransformation):
  - image_content: current_text | blank | unrelated
  - keep_text: True (text+image) | False (image-only with stock instruction)
  - text_replacement: stock text when keep_text=False
  - plus any renderer-specific kwargs (font_size, width, etc.)
"""
from src.prompt_transformations.transformation_factory import register_transformation
from .base_transformation import ImageRendererTransformation
from .renderers.plain_image_renderer import PlainImageRenderer
from .renderers.fc_typography_image_renderer import FCTypographyImageRenderer
from .renderers.fc_flowchart_image_renderer import FCFlowchartImageRenderer
from .renderers.blank_image_renderer import BlankImageRenderer
from .renderers.constant_image_renderer import ConstantImageRenderer
from .renderers.occluded_image_renderer import OccludedImageRenderer


@register_transformation
class IRPlainTransformation(ImageRendererTransformation):
    type_name = "ir_plain"
    renderer_class = PlainImageRenderer


@register_transformation
class IRFCTypoTransformation(ImageRendererTransformation):
    type_name = "ir_fc_typo"
    renderer_class = FCTypographyImageRenderer


# ir_figstep is NOT here: FigStep needs an aux-LLM declarative-paraphrase step, so
# it owns a module like the other LLM-using image attacks → image/figstep.py.


@register_transformation
class IRFCFlowchartTransformation(ImageRendererTransformation):
    """Flowchart-style RENDER — **ours, not FC-Attack.**"""
    type_name = "ir_fc_flowchart"
    renderer_class = FCFlowchartImageRenderer


@register_transformation
class IRBlankTransformation(ImageRendererTransformation):
    type_name = "ir_blank"
    renderer_class = BlankImageRenderer


@register_transformation
class IRConstantTransformation(ImageRendererTransformation):
    type_name = "ir_constant"
    renderer_class = ConstantImageRenderer


@register_transformation
class IRLowContrastTransformation(ImageRendererTransformation):
    # Adversarial Smuggling — Low-Contrast (perceptual-blindness) attack.
    # Reuses the plain renderer; low fg/bg contrast set in conf/imaging/low_contrast.yaml.
    type_name = "ir_low_contrast"
    renderer_class = PlainImageRenderer


@register_transformation
class IROccludedTransformation(ImageRendererTransformation):
    # Adversarial Smuggling — Occluded-Text (perceptual-blindness) attack.
    type_name = "ir_occluded"
    renderer_class = OccludedImageRenderer
