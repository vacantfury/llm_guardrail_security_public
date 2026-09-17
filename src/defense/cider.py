"""CIDER — Cross-modality Information DEtectoR (Xu et al., Findings of EMNLP 2024).

MECHANISM, transcribed from the authors' released code (`other_repos/CIDER`,
`code/defender.py` + `code/utils.py`), not from the paper prose:

  1. Embed the text channel and the image with the target MLLM's OWN encoders.
  2. Denoise the image, recording the embedding at each denoise checkpoint.
  3. Let delta = cos(text, img_denoised) - cos(text, img_original) at each checkpoint.
     Flag the input as adversarial if min(delta) < tau, i.e. if denoising costs
     MORE cross-modal similarity than a clean image would lose.
  4. Flagged -> block (canned refusal). Passed -> the target sees the ORIGINAL input.

So CIDER is a GATE defense in this repo's taxonomy (like `guard_baseline`): it
classifies and blocks, it never transforms the prompt.

TAU IS CALIBRATED, NOT CHOSEN. `defender.py` sets it as a percentile of the
delta distribution measured on CLEAN images:
    self.threshold = np.percentile(decrease, (1 - ratio) * 100)
so tau is a false-positive-rate budget on benign traffic, exactly like our own
guard-threshold sweep. Run this defense once with `threshold: null` over the benign
channel to collect deltas (everything passes, which also gives CIDER's over-refusal
floor for free), then set `threshold` from `paper_c_cider_threshold.py`.

THE EMBEDDING SPACE IS THE MLLM'S, NOT CLIP'S — CORRECTED 2026-08-05. An earlier
version of this file embedded with `CLIPModel.text_embeds` / `.image_embeds`, i.e.
CLIP's *contrastive projection* space, justified on the grounds that LLaVA-1.5-7B's
vision tower is CLIP ViT-L/14-336. That justification was wrong and the numbers it
produced (the delta traces under `outputs/autoattack_defense/cider_deltas/` from
2026-07-30) are INVALID. What the authors actually do, in both `LlavaEncoder` and
`QwenEncoder` (`code/utils.py:200-281`):

  * image -> vision tower -> **multi_modal_projector** -> mean-pool patches,
    landing in the LLM's input-embedding space (4096-d for LLaVA-1.5-7B);
  * text  -> **model.get_input_embeddings()** -> mean-pool tokens, same space.

CLIP's text encoder is never called at all. The paper says the same in prose --
"the image and text encoder of the state-of-the-art MLLM LLaVA-v1.5-7B" (§4.1) --
and attributes the threshold's cross-model transferability to "their shared
transformer-based LLM backbones, which generate comparable representations" (§6),
which is a statement about the LLM space, not CLIP's. Cosine similarity between
mean-pooled LLM token embeddings and mean-pooled projected patches is a different
quantity from CLIP contrastive similarity, and neither the delta distribution nor
the calibrated tau transfers between them.

ONE DECLARED DEVIATION THAT WEAKENS THE DETECTOR — state it wherever a number from
this module is reported. The authors' headline denoiser is guided diffusion run for
350 iterations with an embedding taken every 50 (`generate_denoised_img(...,
model="diffusion", cps=8)`), so `predict()` gets SEVEN deltas per image and flags if
ANY of them falls below tau. Its ImageNet checkpoint (`256x256_diffusion_uncond.pt`,
~2GB) is not shipped in their repo. We fall back to the DnCNN path their `utils.py`
also implements, whose weights ARE shipped -- but that denoises once, so we get ONE
delta and take the min over a single value. That is strictly fewer chances to trip
the detector, so this configuration is a LOWER BOUND on CIDER's detection power. It
biases toward "CIDER does not flag our attacks", which is the direction of our own
pre-registered expectation -- so it must never be reported as CIDER's true strength.

SCOPE. CIDER is defined only where an image exists. Five of our eleven attacks are
text-only; on those the detector has no image to denoise and this implementation
passes the prompt through unchanged (logged, and reported as an inapplicability
rather than a pass rate).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from src.experiment.schemas import Prompt
from llm_utils.base_llm_service import BaseLLMService
from src.utils.logger import get_logger
from .base import Defense, build_conversation_message
from .defender_factory import register_defense
from .guard_utils import GUARD_REFUSAL_TEXT

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# DnCNN, vendored from the CIDER release (code/models/DnCNN/DnCNN.py), which
# itself notes "a modified implementation of the DnCNN from
# https://github.com/cszn/DnCNN". Copied rather than re-derived so the weights
# in their shipped checkpoint load into the architecture they were trained on.
# torch is imported lazily inside the builder: this module is imported by the
# defense factory on every run, including API-only runs with no torch present.
# --------------------------------------------------------------------------
def _build_dncnn(depth: int = 17, n_channels: int = 64, image_channels: int = 3):
    import torch.nn as nn

    class DnCNN(nn.Module):
        def __init__(self):
            super().__init__()
            padding = 1
            kernel_size = 3
            layers = [
                nn.Conv2d(image_channels, n_channels, kernel_size,
                          padding=padding, bias=True),
                nn.ReLU(inplace=True),
            ]
            for _ in range(depth - 2):
                layers += [
                    nn.Conv2d(n_channels, n_channels, kernel_size,
                              padding=padding, bias=False),
                    nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.Conv2d(n_channels, image_channels, kernel_size,
                                    padding=padding, bias=False))
            self.dncnn = nn.Sequential(*layers)

        def forward(self, x):
            return x - self.dncnn(x)   # residual: predicts the noise

    return DnCNN()


def _load_dncnn(ckpt_path: str, device):
    """Load the authors' shipped checkpoint, tolerating either layout.

    Their loader wraps the net in `torch.nn.DataParallel`, so the saved keys may
    carry a `module.` prefix, and the file may be either a bare state_dict or a
    training checkpoint dict with one under `state_dict`.
    """
    import torch

    net = _build_dncnn()
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "net"):
            if key in raw and isinstance(raw[key], dict):
                state = raw[key]
                break
    if hasattr(state, "state_dict"):        # a pickled nn.Module
        state = state.state_dict()
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning(
            f"CIDER DnCNN load: {len(missing)} missing / {len(unexpected)} "
            f"unexpected keys. First missing: {list(missing)[:3]}")
        if len(missing) > len(list(net.state_dict())) // 2:
            raise RuntimeError(
                f"CIDER DnCNN checkpoint at {ckpt_path} does not match the "
                f"architecture ({len(missing)} keys unfilled) — refusing to run "
                f"a randomly-initialised denoiser, which would silently make "
                f"the baseline meaningless.")
    return net.to(device).eval()


# The authors resize every image to 224x224 before denoising and before
# embedding (`utils.py:366`, `utils.py:487`). The DnCNN weights were trained at
# that operating point, so this is not a free parameter.
_CIDER_RESIZE = 224


def _resolve_checkpoint(path: str, what: str, how_to_get_it: str) -> str:
    """Expand `~`/env vars in a checkpoint path and fail with an actionable message.

    Both checkpoint paths in conf/defense/cider.yaml are home-relative
    (`~/models/...`) ON PURPOSE: the three clusters have three different home
    directories, and the weights are too large (2GB) / not ours to redistribute,
    so they are placed per machine rather than committed. But nothing in the
    config layer expands `~` — `load_conf` returns the raw string and `torch.load`
    passes it to `open()`, which treats a leading `~` as a literal directory name.
    Without this, a correctly-provisioned cluster still dies with a bare
    `FileNotFoundError: '~/models/...'` several minutes into a run, after the 7B
    encoder has already loaded. Expanding at the point of USE (not in __init__)
    keeps the portable form in the recorded provenance config, which is the
    honest record: the path IS machine-relative.
    """
    resolved = os.path.expanduser(os.path.expandvars(path))
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"CIDER: {what} not found at {resolved!r} (from {path!r}). {how_to_get_it}")
    return resolved


# --------------------------------------------------------------------------
# Guided-diffusion denoiser — the authors' HEADLINE path, reproducing
# `models/diffusion_denoiser/imagenet/DRM.py::DiffusionRobustModel`.
#
# The UNet/diffusion code is vendored at src/defense/vendor/guided_diffusion
# (OpenAI's guided-diffusion, MIT; the same copy CIDER itself vendors). It is
# vendored rather than pip-installed because upstream's setup.py declares
# `py_modules=["guided_diffusion"]` for what is actually a package, so
# `pip install git+...` does not reliably install it — and because other_repos/
# is gitignored, so the clusters would not receive it any other way.
#
# NOTE ON THE CLASSIFIER: DRM.py also loads a ViT classifier, but only its
# `forward()` uses it. CIDER calls `.denoise()` exclusively, so the classifier is
# omitted here — that removes a dependency without touching the denoise path.
# --------------------------------------------------------------------------
class _DiffusionArgs:
    """Verbatim from DRM.py's Args (the 256x256 unconditional ImageNet model)."""
    image_size = 256
    num_channels = 256
    num_res_blocks = 2
    num_heads = 4
    num_heads_upsample = -1
    num_head_channels = 64
    attention_resolutions = "32,16,8"
    channel_mult = ""
    dropout = 0.0
    class_cond = False
    use_checkpoint = False
    use_scale_shift_norm = True
    resblock_updown = True
    use_fp16 = False
    use_new_attention_order = False
    clip_denoised = True
    num_samples = 10000
    batch_size = 16
    use_ddim = False
    model_path = ""
    classifier_path = ""
    classifier_scale = 1.0
    learn_sigma = True
    diffusion_steps = 1000
    noise_schedule = "linear"
    timestep_respacing = None
    use_kl = False
    predict_xstart = False
    rescale_timesteps = False
    rescale_learned_sigmas = False


class _DiffusionDenoiser:
    """One-shot denoise at a given noise level, exactly as DRM.denoise does.

    For each checkpoint t, the image is noised to level t with `q_sample` and
    then a SINGLE `p_sample` step is taken, keeping `pred_xstart`. The
    checkpoints are therefore independent one-shot denoises at increasing noise
    levels, NOT a progressive denoising chain — a detail easy to get wrong when
    reading the paper's "350 denoising iterations" prose alone.
    """

    def __init__(self, checkpoint: str, device, seed: int = 0):
        import torch
        from .vendor.guided_diffusion.script_util import (
            args_to_dict, create_model_and_diffusion, model_and_diffusion_defaults,
        )

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(_DiffusionArgs(), model_and_diffusion_defaults().keys())
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        self.model = model.to(device).eval()
        self.diffusion = diffusion
        self.device = device
        # The authors do not seed, so their deltas carry sampling noise run to
        # run. We seed per call: same input -> same delta, which makes a
        # calibration pass and a scoring pass comparable. This makes the detector
        # deterministic, it does not change what it measures.
        self.seed = seed

    def denoise(self, x01, t: int):
        """x01: [B,3,H,W] float in [0,1] -> denoised tensor in [0,1]."""
        import torch

        x_start = x01 * 2 - 1                      # their projection to [-1,1]
        gen = torch.Generator(device=x_start.device).manual_seed(self.seed + t)
        noise = torch.randn(x_start.shape, generator=gen,
                            device=x_start.device, dtype=x_start.dtype)
        t_batch = torch.tensor([t] * len(x_start), device=x_start.device)
        with torch.no_grad():
            x_t = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)
            out = self.diffusion.p_sample(
                self.model, x_t, t_batch, clip_denoised=True)["pred_xstart"]
        return torch.clamp((out + 1) / 2, 0, 1)    # back to [0,1]


@register_defense
class Cider(Defense):
    """Published cross-modal consistency detector, as a gate defense."""

    type_name = "cider"

    def __init__(self,
                 encoder_model: str = "llava-hf/llava-1.5-7b-hf",
                 denoiser: str = "diffusion",
                 diffusion_checkpoint: str = "",
                 denoise_steps: int = 350,
                 denoise_every: int = 50,
                 dncnn_checkpoint: str = "",
                 threshold: Optional[float] = None,
                 device: str = "cuda",
                 batch_size: int = 16,
                 trace_dir: str = "outputs/autoattack_defense/cider_deltas",
                 **kwargs):
        # Fail loudly on the superseded key rather than swallowing it in kwargs
        # and silently running the wrong embedding space again.
        if "clip_model" in kwargs:
            raise ValueError(
                "CIDER: `clip_model` is no longer a valid option. CIDER embeds "
                "with the target MLLM's own encoders (vision tower -> "
                "multi_modal_projector, and the LLM input-embedding table for "
                "text), NOT with CLIP's contrastive projection space — see this "
                "module's docstring. Set `encoder_model` instead, and discard any "
                "delta traces produced before 2026-08-05.")
        if denoiser not in ("diffusion", "dncnn"):
            raise ValueError(
                f"CIDER: denoiser must be 'diffusion' (the authors' headline "
                f"path) or 'dncnn' (their shipped fallback), got {denoiser!r}.")
        super().__init__(encoder_model=encoder_model, denoiser=denoiser,
                         diffusion_checkpoint=diffusion_checkpoint,
                         denoise_steps=denoise_steps, denoise_every=denoise_every,
                         dncnn_checkpoint=dncnn_checkpoint,
                         threshold=threshold, device=device,
                         batch_size=batch_size, trace_dir=trace_dir, **kwargs)
        self._encoder_name = encoder_model
        self._denoiser_kind = denoiser
        self._diffusion_ckpt = diffusion_checkpoint
        # range(50, 400, 50) in their threshold_selection.py: seven checkpoints.
        self._checkpoints = list(range(denoise_every, denoise_steps + 1, denoise_every))
        self._ckpt = dncnn_checkpoint
        self._tau = threshold
        self._device_str = device
        self._batch_size = batch_size
        self._trace_dir = trace_dir
        self._vlm = None
        self._processor = None
        self._denoiser = None
        self._device = None

    # ---------------- lazy model construction ----------------
    def _load(self):
        if self._vlm is not None:
            return
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self._device = torch.device(
            self._device_str if torch.cuda.is_available() else "cpu")
        if self._device.type == "cpu":
            logger.warning("CIDER: no CUDA visible — running the 7B encoder + "
                           "DnCNN on CPU, which is slow but correct.")
        logger.info(f"CIDER: loading encoder {self._encoder_name} on {self._device}")
        self._vlm = LlavaForConditionalGeneration.from_pretrained(
            self._encoder_name,
            torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self._device).eval()
        self._processor = AutoProcessor.from_pretrained(self._encoder_name)

        if self._denoiser_kind == "diffusion":
            if not self._diffusion_ckpt:
                raise ValueError(
                    "CIDER needs diffusion_checkpoint — OpenAI's "
                    "256x256_diffusion_uncond.pt (~2GB), which the authors' repo "
                    "does not ship. Download it from "
                    "https://openaipublic.blob.core.windows.net/diffusion/"
                    "jul-2021/256x256_diffusion_uncond.pt and set the path in "
                    "conf/defense/cider.yaml, or set denoiser: dncnn to run the "
                    "weaker shipped fallback (a LOWER BOUND — see the docstring).")
            logger.info(
                f"CIDER: loading guided-diffusion denoiser (the authors' headline "
                f"path), {len(self._checkpoints)} checkpoints at t="
                f"{self._checkpoints}")
            self._denoiser = _DiffusionDenoiser(
                _resolve_checkpoint(
                    self._diffusion_ckpt, "diffusion checkpoint",
                    "Download it once per machine: curl -L -o "
                    "~/models/cider_256x256_diffusion_uncond.pt "
                    "https://openaipublic.blob.core.windows.net/diffusion/"
                    "jul-2021/256x256_diffusion_uncond.pt"),
                self._device)
        else:
            if not self._ckpt:
                raise ValueError(
                    "CIDER needs dncnn_checkpoint — the denoiser weights shipped "
                    "in the authors' repo at code/models/DnCNN/checkpoint.pth.tar. "
                    "Set it in conf/defense/cider.yaml (other_repos/ is gitignored, "
                    "so the file must be placed on each cluster).")
            logger.warning(
                "CIDER: using the DnCNN fallback, which denoises ONCE. This yields "
                "one delta instead of the headline path's seven, so the detector "
                "is strictly weaker and this run is a LOWER BOUND on CIDER's "
                "detection power. Do not report it as CIDER's strength.")
            self._denoiser = _load_dncnn(
                _resolve_checkpoint(
                    self._ckpt, "DnCNN checkpoint",
                    "Copy it from the authors' repo: "
                    "other_repos/CIDER/code/models/DnCNN/checkpoint.pth.tar "
                    "-> ~/models/cider_dncnn_checkpoint.pth.tar on each cluster."),
                self._device)

    # ---------------- the authors' encoder path ----------------
    def _submodule(self, name: str):
        """Resolve a LLaVA submodule across transformers layouts.

        transformers moved `vision_tower` / `multi_modal_projector` under
        `.model` in 4.52; the authors' code predates that. Try both rather than
        pinning a version, and fail loudly if neither exists — a silently missing
        projector would leave us embedding in the wrong space all over again.
        """
        for holder in (self._vlm, getattr(self._vlm, "model", None)):
            if holder is not None and hasattr(holder, name):
                return getattr(holder, name)
        raise AttributeError(
            f"CIDER: could not find `{name}` on {type(self._vlm).__name__} or its "
            f".model — the installed transformers layout is not one this encoder "
            f"path handles. Fix the resolver rather than falling back, since any "
            f"fallback would change the embedding space.")

    def _embed_text(self, texts: list[str]):
        """Mean-pooled LLM input embeddings, matching LlavaEncoder.embed_text.

        The authors embed one string at a time with no padding. We batch with a
        padded tokenizer but mask the pad positions out of the mean, which is
        arithmetically identical to their per-item mean and much faster.
        """
        import torch

        tok = self._processor.tokenizer
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(self._device)
        mask = enc["attention_mask"].to(self._device).unsqueeze(-1)
        emb = self._vlm.get_input_embeddings()(input_ids)
        summed = (emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return (summed / counts).float()

    def _embed_image(self, pixel_values):
        """Mean-pooled projected patches, matching LlavaEncoder.embed_img."""
        vision_tower = self._submodule("vision_tower")
        projector = self._submodule("multi_modal_projector")

        outputs = vision_tower(pixel_values, output_hidden_states=True)
        layer = self._vlm.config.vision_feature_layer
        if isinstance(layer, (list, tuple)):   # newer configs allow a list
            layer = layer[0]
        feat = outputs.hidden_states[layer]
        feat = feat[:, 1:]                      # drop CLS, "by default" in their code
        feat = projector(feat)
        return feat.mean(dim=1).float()

    # ---------------- the detector ----------------
    def _deltas(self, items: list[tuple[str, str, object]]) -> dict[str, float]:
        """min(delta) per prompt id; delta = cos(text,denoised) - cos(text,orig)."""
        import numpy as np
        import torch
        import torch.nn.functional as F

        self._load()
        deltas: dict[str, float] = {}
        for start in range(0, len(items), self._batch_size):
            chunk = items[start:start + self._batch_size]
            ids = [c[0] for c in chunk]
            texts = [c[1] or "" for c in chunk]
            images = [c[2] for c in chunk]
            # The paginating renderer can emit several pages; CIDER scores one
            # image, so take the first page and note it in the trace.
            images = [im[0] if isinstance(im, list) else im for im in images]
            # Match the authors: RGB, resized to 224x224 before anything else.
            images = [im.convert("RGB").resize((_CIDER_RESIZE, _CIDER_RESIZE))
                      for im in images]

            with torch.no_grad():
                t_emb = self._embed_text(texts)

                # Denoisers operate on [0,1] pixels at 224x224.
                raw = torch.stack([
                    torch.from_numpy(np.asarray(im).astype("float32") / 255.0)
                    .permute(2, 0, 1)
                    for im in images
                ]).to(self._device)

                if self._denoiser_kind == "diffusion":
                    denoised = [self._denoiser.denoise(raw, t)
                                for t in self._checkpoints]
                else:
                    dn_in = raw.to(next(self._denoiser.parameters()).dtype)
                    denoised = [torch.clamp(self._denoiser(dn_in), 0, 1)]

                # Both arms go through the SAME image processor, so any
                # normalisation/resize it applies cancels in the delta.
                proc = self._processor.image_processor
                dtype = next(self._vlm.parameters()).dtype

                def _embed(tensor_or_pils):
                    px = proc(images=tensor_or_pils, return_tensors="pt")["pixel_values"]
                    return self._embed_image(px.to(self._device, dtype))

                def _to_uint8(batch):
                    return [(d.permute(1, 2, 0).float().cpu().numpy() * 255)
                            .astype("uint8") for d in batch]

                cos_o = F.cosine_similarity(t_emb, _embed(images), dim=-1)

                # min over checkpoints: their predict() flags if ANY checkpoint's
                # delta dips below tau, which is exactly min(delta) < tau.
                per_ckpt = []
                for den in denoised:
                    cos_d = F.cosine_similarity(t_emb, _embed(_to_uint8(den)), dim=-1)
                    per_ckpt.append(cos_d - cos_o)
                diffs = (torch.stack(per_ckpt).min(dim=0).values
                         .detach().cpu().reshape(-1).tolist())
                if len(diffs) != len(ids):
                    raise RuntimeError(
                        f"CIDER: {len(diffs)} deltas for {len(ids)} inputs — the "
                        f"encoder batch and the id list disagree, which would "
                        f"mis-attribute every score in this chunk.")
                for pid, delta in zip(ids, diffs):
                    # Keys are forced to str: a prompt id that arrives as a tensor
                    # or numpy scalar silently breaks the JSON trace downstream.
                    deltas[str(pid)] = float(delta)
        return deltas

    def query(
        self,
        prompts: list[Prompt],
        target_service: BaseLLMService,
        is_multimodal: bool,
        source_dir: Optional[Path] = None,
        system_message: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        deltas: dict[str, float] = {}
        if is_multimodal:
            items = []
            for p in prompts:
                messages = build_conversation_message(p, is_multimodal, source_dir)
                text_side, image_side = messages[0]
                if image_side is None:
                    continue
                items.append((p.id, text_side or "", image_side))
            if items:
                deltas = self._deltas(items)
        else:
            logger.info(
                f"CIDER: text-only input ({len(prompts)} prompts) — the detector "
                f"has no image to denoise, so it is INAPPLICABLE here and every "
                f"prompt passes. Reported as inapplicability, not as coverage.")

        # tau=None is the calibration pass: measure deltas, block nothing.
        blocked: set[str] = set()
        if self._tau is not None:
            blocked = {pid for pid, d in deltas.items() if d < self._tau}
        logger.info(
            f"CIDER: tau={self._tau}, scored {len(deltas)} images, "
            f"blocked {len(blocked)}/{len(prompts)}"
            + (" (CALIBRATION PASS — nothing blocked by design)"
               if self._tau is None else ""))

        # Deltas depend ONLY on (text, image) — not on the guard, the condition, or
        # the target — so the trace is keyed by the upstream attack/channel and one
        # file per attack is both correct and collision-free across cells. It is
        # written to its OWN tree, never into the shared prompt_transform artifact
        # dir, which cells read concurrently.
        if source_dir is not None and deltas:
            try:
                out_dir = Path(self._trace_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                src = Path(source_dir)
                trace = out_dir / f"{src.parent.name}__{src.name}.jsonl"
                with open(trace, "w") as fh:
                    for pid, d in deltas.items():
                        fh.write(json.dumps({"id": pid, "delta": d,
                                             "blocked": pid in blocked}) + "\n")
            except OSError as exc:
                logger.warning(f"CIDER: could not write delta trace: {exc}")

        target_convs = [
            (p.id, build_conversation_message(p, is_multimodal, source_dir))
            for p in prompts if p.id not in blocked
        ]
        results: dict[str, str] = {}
        if target_convs:
            results = dict(target_service.batch_chat(
                conversations=target_convs,
                system_message=system_message,
                is_test=True,
            ))
        return [(p.id, results.get(p.id, GUARD_REFUSAL_TEXT)) for p in prompts]
