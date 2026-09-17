# Vendored: OpenAI `guided-diffusion`

Source: <https://github.com/openai/guided-diffusion> (MIT License, © 2021 OpenAI).
Accompanying paper: Dhariwal & Nichol, *Diffusion Models Beat GANs on Image
Synthesis*, NeurIPS 2021.

These files are copied **unmodified** from that project. They are vendored, not
imported as a dependency, for two reasons:

1. Upstream's `setup.py` declares `py_modules=["guided_diffusion"]` for what is
   actually a package, so `pip install git+https://github.com/openai/guided-diffusion`
   does not reliably install an importable module.
2. Our reference copy lives under `other_repos/`, which is gitignored — so the
   compute clusters would never receive it. Vendoring puts it on the git path.

## Why this repo needs it

It is the denoiser behind the **CIDER** baseline (`src/defense/cider.py`). CIDER
(Xu et al., Findings of EMNLP 2024) detects adversarial images by measuring how
much cross-modal similarity an image loses under denoising, and its headline
configuration uses this guided-diffusion model. The CIDER authors vendor the same
code in their own release; we reproduce their setup rather than substituting a
different denoiser, because substituting the weaker DnCNN fallback measurably
weakens the detector and would bias our reported result.

## Pruning

Only the modules on the `create_model_and_diffusion` → `q_sample`/`p_sample` path
are included: `script_util`, `unet`, `gaussian_diffusion`, `nn`, `respace`,
`fp16_util`, `losses`, `logger`. The training/distributed/data-loading modules
(`train_util`, `image_datasets`, `dist_util`, `resample`) are omitted — nothing
retained imports them.

## Weights

The checkpoint (`256x256_diffusion_uncond.pt`, ~2 GB) is **not** in this repo. It
is downloaded per machine; see `diffusion_checkpoint` in `conf/defense/cider.yaml`.
