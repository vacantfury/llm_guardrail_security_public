"""
Concrete image renderers.

To add a new renderer:
1. Create a new file in this directory (e.g., visco_image_renderer.py)
2. Create a class inheriting from BaseImageRenderer
3. Implement _render_clean() and get_config()
4. Register it in image_renderer_factory.py
"""

from .figstep_image_renderer import FigstepImageRenderer
from .fc_typography_image_renderer import FCTypographyImageRenderer
from .fc_flowchart_image_renderer import FCFlowchartImageRenderer
from .plain_image_renderer import PlainImageRenderer
from .blank_image_renderer import BlankImageRenderer
from .constant_image_renderer import ConstantImageRenderer
from .occluded_image_renderer import OccludedImageRenderer

__all__ = [
    'PlainImageRenderer',
    'FigstepImageRenderer',
    'FCTypographyImageRenderer',
    'FCFlowchartImageRenderer',
    'BlankImageRenderer',
    'ConstantImageRenderer',
    'OccludedImageRenderer',
]
