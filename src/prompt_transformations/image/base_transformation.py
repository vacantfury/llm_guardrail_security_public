"""
Adapter base for BaseImageRenderer subclasses.

Each image transformation owns the multimodal message it produces: the image
channel (rendered by its BaseImageRenderer) and the text channel (whether to
keep the encoded text alongside, or replace it with a stock "check the image"
instruction).

Renderer constructor signatures vary, so this base does `inspect.signature`
filtering on kwargs before forwarding to renderer_class.

Two shared params on every image transformation:

  keep_text (bool): text-channel handling AFTER rendering
      False (default) → replace Prompt.encoded with `text_replacement`
                        ("Please check the image and answer accordingly.").
                        Image-only delivery mode.
      True            → leave Prompt.encoded unchanged. Text+image both
                        carry signal.

  text_replacement (str | None): the stock text used when keep_text=False.
      Defaults to "Please check the image and answer accordingly."

What gets rendered into the image is entirely the renderer's concern:
  ir_plain / ir_fc_*  → render the encoded text
  ir_blank            → ignore text, output blank canvas
  ir_constant         → ignore text, load a fixed image file
"""
import inspect
from pathlib import Path
from typing import ClassVar, Optional, Type

from src.experiment.schemas import Prompt
from src.prompt_transformations.base import PromptTransformation, Modality
from .base_image_renderer import BaseImageRenderer


# Meta kwargs our wrapper consumes that renderer constructors don't accept.
_OUR_META_KWARGS = frozenset({
    "type_name", "input_modality", "output_modality",
    "keep_text", "text_replacement",
})


# Stock text used to replace Prompt.encoded when keep_text=False (image-only delivery).
DEFAULT_TEXT_REPLACEMENT = (
    "Please check the image and answer accordingly."
)


class ImageRendererTransformation(PromptTransformation):
    """Adapter wrapping a BaseImageRenderer subclass.

    Subclasses set `type_name` and `renderer_class`. The chain executor sees
    the standard `apply(prompts, step_dir)` interface; per-prompt the wrapper
    writes one PNG into step_dir/images/ and updates Prompt.image_encoded
    plus optionally Prompt.encoded.

    The renderer receives Prompt.encoded as the `text` argument to
    `render_to_file()`; renderers that ignore text (blank, constant) simply
    don't use it.
    """

    input_modality = Modality.TEXT
    output_modality = Modality.MULTIMODAL

    renderer_class: ClassVar[Optional[Type[BaseImageRenderer]]] = None

    def __init__(
        self,
        keep_text: bool = True,
        text_replacement: Optional[str] = None,
        **renderer_kwargs,
    ):
        super().__init__(
            keep_text=keep_text, text_replacement=text_replacement,
            **renderer_kwargs,
        )
        if self.renderer_class is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set `renderer_class`")
        self._keep_text = keep_text
        self._text_replacement = text_replacement or DEFAULT_TEXT_REPLACEMENT
        forward = {
            k: v for k, v in renderer_kwargs.items() if k not in _OUR_META_KWARGS
        }
        # Strip subsystem-level fields that aren't renderer constructor kwargs.
        forward.pop("renderer_type", None)
        forward.pop("quality_check", None)
        # Drop any remaining kwargs the specific renderer constructor doesn't
        # accept — handles cross-renderer YAML fields (e.g. `max_aspect_ratio`
        # is in conf/imaging/default.yaml but only some renderers accept it).
        sig = inspect.signature(self.renderer_class.__init__)
        valid = set(sig.parameters) - {"self"}
        forward = {k: v for k, v in forward.items() if k in valid}
        self._renderer: BaseImageRenderer = self.renderer_class(**forward)
        self._image_count = 0

    def apply(self, prompts: list[Prompt], step_dir: Path) -> list[Prompt]:
        images_dir = step_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        out: list[Prompt] = []
        for p in prompts:
            img_name = f"{p.id}_encoded.png"
            text_to_render = p.encoded or ""
            # Paginating renderers may emit multiple images; single-image
            # renderers return exactly one path (saved as `img_name`).
            abs_paths = self._renderer.render_to_files(
                text_to_render, str(images_dir / img_name))
            img_rels = [f"images/{Path(pth).name}" for pth in abs_paths]
            self._image_count += len(abs_paths)

            updates: dict = {"image_encoded": img_rels, "encoding": self.type_name}
            if not self._keep_text:
                # Image-only delivery: replace the text channel with a stock
                # "check the image" instruction so the model has something to
                # latch onto besides the image.
                updates["encoded"] = self._text_replacement
            # else: keep Prompt.encoded as-is (text+image both go to model).
            out.append(p.model_copy(update=updates))
        return out

    def step_metrics(self) -> dict:
        return {"image_count": self._image_count, "images_dir": "images/"}
