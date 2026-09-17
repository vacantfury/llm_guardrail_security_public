"""Text-to-multimodal prompt transformations.

All variants share the same ImageRendererTransformation wrapper and live in
`wrappers.py`, each backed by a BaseImageRenderer subclass in `renderers/`:

  ir_plain         — renders Prompt.encoded as plain text-on-canvas
  ir_fc_typo       — typography-formatted rendering
  ir_figstep       — FigStep (own module figstep.py: aux-LLM declarative
                     paraphrase -> typographic numbered list -> fixed instruction)
  ir_fc_flowchart  — flowchart-style structural rendering
  ir_blank         — pure blank canvas (image channel carries nothing)
  ir_constant      — fixed image loaded from disk (e.g. rabit.jpeg);
                     swap via `image_path` param

Param convention shared by all:
  - keep_text:   True keeps Prompt.encoded; False replaces with text_replacement
  - text_replacement: stock text used when keep_text=False
                      (default: "Please check the image and answer accordingly.")
  - renderer-specific kwargs forwarded by the wrapper (font_size, image_path, etc.)
"""
from . import wrappers       # noqa: F401  — all ir_* transformations registered here
from . import cross_modal_split  # noqa: F401  — Paper C RQ-D split attack
from . import semantic_split
from . import mm_safetybench_typo  # noqa: F401  — ir_mm_typo (query-relevant typography)
from . import distraction_grid     # noqa: F401  — ir_distraction_grid (Text-DJ/CS-DJ)
from . import figstep            # noqa: F401  — ir_figstep (declarative paraphrase + typographic list)
from . import camo                  # noqa: F401  — ir_camo (Jiang et al. 2025 cross-modal obfuscation)
