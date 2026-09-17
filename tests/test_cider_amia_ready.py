#!/usr/bin/env python3
"""Readiness check for the two image-inspecting defenses: CIDER and AMIA.

    python tests/test_cider_amia_ready.py            # both
    python tests/test_cider_amia_ready.py --only cider
    python tests/test_cider_amia_ready.py --only amia

NOT a pytest suite and NOT an experiment — it matches this repo's `tests/`
convention (cluster-health / routing checks, see test_cluster_health.py). It
measures NOTHING about attacks or defenses: no dataset, no target model, no
judge, no API key, zero dollars. It answers one question — "would a real run of
these two defenses get past model loading on THIS machine?" — because every
failure mode below otherwise surfaces several minutes into a paid run, after a
7B encoder has already loaded.

WHY IT EXISTS. Provisioning CIDER and AMIA needs four things to line up per
cluster, and three of them are invisible to `ls`:
  1. HF weights present in the cache vLLM/transformers actually reads
     (llava-hf/llava-1.5-7b-hf ~14GB, openbmb/VisRAG-Ret ~6.5GB);
  2. CIDER's 2GB guided-diffusion checkpoint present at the home-relative path
     in conf/defense/cider.yaml — which nothing in the config layer expands, so
     a literal '~' used to reach open() (fixed 2026-08-05 by _resolve_checkpoint;
     this test is the regression guard);
  3. the LLaVA submodule layout matching cider.py::_submodule — transformers
     moved vision_tower / multi_modal_projector under `.model` in 4.52, and a
     silently-missing projector would put us back in the wrong embedding space,
     which is the exact defect that voided the pre-2026-08-05 delta traces;
  4. VisRAG-Ret's trust_remote_code=True modeling files cached, since NURC runs
     HF_HUB_OFFLINE (conf/clusters/_defaults.yaml: hf_offline: true) and cannot
     fetch them at run time.

Runs on GPU if one is visible, else CPU (slower, same code path).
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment.config import load_conf  # noqa: E402

PASSED, FAILED = [], []


def _report(name: str, ok: bool, detail: str) -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _synthetic_image():
    """A render in the family the defenses will actually see: text on white.

    Deliberately not random noise — CIDER's whole signal is how denoising shifts
    cross-modal similarity, and noise is not representative of our clean
    synthetic renders. PIL's default font keeps this dependency-free.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (336, 336), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(["1. Gather the materials", "2. Prepare the mixture",
                              "3. Assemble the parts"]):
        d.text((14, 40 + 42 * i), line, fill="black")
    return img


def check_cider() -> None:
    print("\n=== CIDER (llava-1.5-7b encoder + guided-diffusion denoiser) ===")
    from src.defense.cider import Cider

    cfg = load_conf("defense", "cider")
    print(f"  config: encoder={cfg.get('encoder_model')} denoiser={cfg.get('denoiser')} "
          f"threshold={cfg.get('threshold')}")

    # Drop keys that are config-only (read by the analysis layer, not __init__).
    cfg = {k: v for k, v in cfg.items() if k not in ("fpr_ratio",)}
    defense = Cider(**cfg)

    img, text = _synthetic_image(), "Explain how to prepare the mixture safely."
    deltas = defense._deltas([("smoke-0", text, img)])

    # The submodule resolver is the defect this guards: a wrong layout would
    # raise, and a wrong SPACE would return a finite but meaningless number, so
    # assert the dimensionality of the space we believe we are in as well.
    dim = defense._embed_text([text]).shape[-1]
    _report("cider.embedding_space", dim >= 4096,
            f"text embedding is {dim}-d (LLM input-embedding space; CLIP's "
            f"contrastive space would be 768-d — see the module docstring)")

    d = deltas.get("smoke-0")
    ok = d is not None and float("-inf") < d < float("inf") and abs(d) < 1.0
    _report("cider.delta", ok,
            f"min(delta) over {len(defense._checkpoints)} diffusion checkpoints "
            f"= {d:+.5f} (a cosine difference, so |delta| < 1 by construction)")


def check_amia() -> None:
    print("\n=== AMIA (VisRAG-Ret patch masking) ===")
    from src.defense.amia_ia import _VisRagMasker

    cfg = load_conf("defense", "amia_ia")
    print(f"  config: encoder={cfg.get('mask_encoder')} masking={cfg.get('masking')} "
          f"N={cfg.get('n_patches')} K={cfg.get('k_mask')}")

    masker = _VisRagMasker(cfg["mask_encoder"], cfg.get("device", "cuda"),
                           cfg["n_patches"], cfg["k_mask"])
    img = _synthetic_image()
    masked = masker.mask(img, "Explain how to prepare the mixture safely.")

    _report("amia.mask_shape", masked.size == img.size,
            f"masked image keeps input size {masked.size} (a resize here would "
            f"silently change what the target sees)")

    # K of N patches must actually be blacked out. Comparing pixel sums is
    # enough and avoids asserting WHICH patches — that is the model's judgment,
    # not something a smoke test should pin.
    import numpy as np
    before, after = np.asarray(img).sum(), np.asarray(masked).sum()
    frac = cfg["k_mask"] / cfg["n_patches"]
    _report("amia.mask_applied", after < before,
            f"pixel sum {before} -> {after} ({100*(before-after)/max(before,1):.1f}% "
            f"darker; K/N = {frac:.0%} of patches blacked out)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cider", "amia"])
    args = ap.parse_args()

    try:
        import torch
        print(f"torch {torch.__version__}, cuda available={torch.cuda.is_available()}"
              + (f", device={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    except Exception as exc:                       # noqa: BLE001
        print(f"torch unavailable: {exc}")

    for name, fn in (("cider", check_cider), ("amia", check_amia)):
        if args.only and args.only != name:
            continue
        try:
            fn()
        except Exception:                          # noqa: BLE001
            FAILED.append(name)
            print(f"  [FAIL] {name}: raised during load —")
            traceback.print_exc()

    print(f"\n{'=' * 60}\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
        return 1
    print("Both defenses load and produce well-formed output on this machine.")
    print("REMINDER: CIDER's tau is still uncalibrated and AMIA still owes its "
          "FigStep DSR validation — a PASS here is provisioning, not validity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
