# LLM Guardrail Security

Research on the security of the **guardrail layer**: the safety mechanisms deployed around large language models and vision-language models, such as guard classifiers, input and output filters, and caption or decode pipelines. The work studies how encoded and image-rendered jailbreak prompts pass these guards, and how black-box defenses can close the gap.

This repository is the shared experiment harness behind the papers below. One codebase serves all of them: the attacks, defenses, judges and the experiment runner are common, and each paper owns a namespace of experiment presets.

## Papers

| Paper | Where | Experiment presets |
|---|---|---|
| *Exposing LLM Safety Gaps Through Mathematical Encoding: New Attacks and Systematic Analysis* | Canadian AI 2026, PMLR v318, pp. 662–677. [Proceedings](https://proceedings.mlr.press/v318/zhang26a.html) | text encoders in `src/prompt_transformations/text/`; original code: [math_encoding_llm_jailbreaking](https://github.com/vacantfury/math_encoding_llm_jailbreaking) |
| *Attack Ensembles Expose a Safety-Utility Trade-off in Black-Box Guard Defenses Against Encoded VLM Jailbreaks* | [arXiv:2607.26574](https://arxiv.org/abs/2607.26574) | `conf/experiment/autoattack_defense/` |
| *Borrowed Strength: Best-of-N Search over a Code Encoding Breaks Self-Check Jailbreak Defenses* | [arXiv:2607.26639](https://arxiv.org/abs/2607.26639) | `conf/experiment/bestofn_attack/`, `conf/experiment/bestofn_defense/` |
| *Decoy Images Amplify Caption-Mediated Defenses Against Encoded Jailbreaks* | [arXiv:2608.01043](https://arxiv.org/abs/2608.01043) | `conf/experiment/image_presence_threshold/` |
| *Whose Refusal Is It? The Unmeasured Contribution of Black-Box Multimodal Guardrails* | [arXiv:2608.08641](https://arxiv.org/abs/2608.08641) | `conf/experiment/defense_read_access/` (earlier rounds share `image_presence_threshold/`) |

## Threat model

**Attacker.** Black-box. The attacker sends inputs to a deployed system and reads its outputs. There is no access to weights, gradients, system prompts or guard internals. The attacker controls the full user input: text, an attached image, or both. The goal is a harmful completion for a given harmful request (HarmBench and JailbreakBench behaviors).

**Attack surface.** The attacker re-represents the request instead of optimizing tokens:

- *Text encodings* rewrite the request into another representation: set theory, formal logic, quantum-mechanics notation, classical languages, ciphers, code, semantic camouflage, nested scenes.
- *Image renderings* move the text into an image: plain typography, FigStep-style lists, flowcharts, low-contrast and occluded renders, distraction grids.
- *Chains* combine the two, so any encoding can be rendered into an image.

Three attacker strengths are studied. A *suite* attacker tries every attack in a fixed suite and succeeds if any one succeeds. A *sampling* attacker draws N variants of one attack (Best-of-N). An *adaptive* attacker knows the defense design and aims at it (decode evasion, harm split across the text and image channels).

**Defender.** Also black-box with respect to the target: the target model is frozen and only queried. The defender adds components around it at inference time, with no retraining: guard classifiers that block or pass an input (gates), prompt-level transform defenses, and recover and decode steps that turn an obfuscated input back into plain text before a guard reads it.

**What is measured.** Attack success rate (ASR): the share of harmful requests that yield a harmful completion, scored by an external judge with the HarmBench rubric. The cost on benign traffic: over-refusal, scored with the OR-Bench three-class rubric. The judge is never the target and never a defense component.

**Out of scope.** White-box and gradient-based attacks, training-time defenses, multi-turn conversations and agent or tool-use settings.

## Pipeline

```
        ATTACK                    DEFENSE                        MEASURE
  ┌───────────────┐    ┌──────────────────────────┐    ┌──────────────────┐
  │ text encoder  │    │ amplifier: recover+decode│    │ target VLM answer│
  │  ± image      │───▶│         ↓                │───▶│        ↓         │
  │  renderer     │    │ defense: transform OR gate│   │ judge → ASR / OR │
  └───────────────┘    └──────────────────────────┘    └──────────────────┘
```

1. **Attack** (`prompt_transform` mode). A prompt, harmful or benign, is encoded into another representation, rendered into an image, or both.
2. **Amplifier.** *Recover*: the target VLM transcribes the image back to text. *Decode*: the target VLM restates the encoding in plain English. This is a transform, not a safety mechanism. It exists so that a defense can see the hidden payload, and it can be placed in front of any guard.
3. **Defense** (`defense+evaluate` mode). The safety mechanism acts on the prompt, in one of two kinds:
   - A **transform defense** (for example SAGE) rewrites the text into a safety-check prompt that the target answers.
   - A **gate defense** (for example WildGuard, LlamaGuard-3, Qwen3Guard) is a classifier. It blocks with a fixed refusal or passes the original prompt through unchanged. A gate never rewrites.
4. **Judge.** An external model scores the answer. Harm is scored with the HarmBench rubric and gives ASR. Refusal is scored with the OR-Bench rubric and gives over-refusal.

The amplifier's effect is measured as `modality_complete` against `guard_baseline` with the defense held fixed. Over-refusal is a property of the chosen defense's own calibration, not of the amplifier.

**Headline metric: ensemble ASR.** The attack side is a suite of about ten elementary encoders and renderers, so a behavior counts as broken under a condition if any attack in the suite breaks it (an OR-reduction over per-prompt results, `src/analysis/portfolio.py`). Headline comparisons run `no_defense`, then `guard_baseline`, then `modality_complete` on that number. Per-attack ASR and its mean are diagnostic views only: different attacks break different inputs, so the mean understates the attacker's real power.

**Judge.** `gpt-5-mini` is the main judge, selected by a human-calibration comparison against a more lenient earlier judge. WildGuard is used only as a secondary robustness lens. The `rejudge` mode re-scores stored responses with a different judge without querying the target again.

## Attacks

Text encoders and image renderers share one registry (`src/prompt_transformations/transformation_factory.py`). Steps compose into chains.

| Family | `type_name` |
|---|---|
| Baselines, no LLM | `non_llm_baseline`, `non_llm_homoglyph`, `non_llm_artprompt`, `non_llm_cipher` (base64, caesar), `non_llm_symbol_injection` |
| Semantic encodings (LLM-written) | `llm_set_theory`, `llm_formal_logic`, `llm_quantum_mechanics`, `llm_classical_language`, `llm_semantic_camo` |
| Decomposition and framing | `non_llm_addition_equation_split_reassemble`, `non_llm_conditional_probability`, `deep_inception`, `ecso_evade`, `code_attack`, `code_attack_no_syntax` |
| Image renderers | `ir_plain` (fixed-font, paginated), `ir_figstep`, `ir_fc_typo`, `ir_fc_flowchart`, `ir_blank`, `ir_constant` |
| Established multimodal attacks | `ir_low_contrast`, `ir_occluded` (perceptual blindness), `ir_mm_typo` (MM-SafetyBench), `ir_distraction_grid` (Text-DJ, CS-DJ), `ir_camo` (cross-modal obfuscation, Jiang et al. 2025, reimplemented from the paper) |
| Best-of-N | `non_llm_best_of_n` (Hughes et al., NeurIPS 2025), `llm_paraphrase`, `variance_channel_bon` (one Best-of-N transform, parameterized by where the N draws differ: surface noise, paraphrase, or sampled attack strategy) |
| Adaptive, aimed at this repository's own defense | `llm_decode_evasion`, `cross_modal_split`, `ir_semantic_split` |

## Defenses

All defenses register in `src/defense/defender_factory.py` and implement one interface, `query(prompts, target_service, is_multimodal, source_dir, system_message)`. A defense owns its interaction with the target: it may wrap the input, query once, or query, inspect and query again.

| `type_name` | Kind | Notes |
|---|---|---|
| `no_defense` | none | pass-through baseline, the attack floor |
| `sage` | transform | prompt-level safety guard on the input text |
| `semantic_smooth` | transform | semantic-perturbation smoothing baseline |
| `ecso` | transform | caption-mediated re-verification, active when an image is present |
| `amia_ia` | transform | intention analysis only (AMIA, Zhang et al., Findings of EMNLP 2025) |
| `llm_self_defense` | gate, output side | LLM Self Defense (Phute et al., ICLR 2024 TinyPaper): an LLM screens the target's response |
| `selfdefend` | gate, input side | SelfDefend (Wang et al., USENIX Security 2025): a separate shadow model screens the query |
| `cider` | gate | CIDER (Xu et al., Findings of EMNLP 2024): cross-modal consistency detector, reimplemented from the authors' released code |
| `guard_baseline` | gate | a published guard classifier alone, with no amplifier: the comparison arm |
| `modality_complete` | amplifier + defense | recover and decode every channel, then one safety check. With `guard_model=None` it uses the SAGE transform, with `guard_model=<classifier>` it is a gate |
| `canonicalize` | transform | input normalization against Best-of-N surface noise |
| `canonicalize_guard` | gate | canonicalize, then guard, with an AND/OR compose option |
| `joint_verify` | joint | joint text and image verification (built, left as future work) |

Guard checkpoints available as gates: LlamaGuard-3-8B, LlamaGuard-4-12B, WildGuard, Qwen3Guard-Gen-8B, GuardReasoner-VL-7B, ShieldLM-7B, ThinkGuard.

## Code structure

```
main.py                      # entry point: python main.py <preset>
dispatch.py                  # splits one preset across several SLURM clusters (dry-run by default)

conf/
├── experiment/<namespace>/  # experiment presets, one namespace per paper (see the Papers table)
├── text_encoding/           # encoder configs (set_theory, formal_logic, cipher, classical_language/, ...)
├── imaging/                 # renderer configs (ir_plain, figstep, mm_typo, semantic_split, ...)
├── defense/                 # defense configs (sage, semantic_smooth, canonicalize, amia_ia, modality_complete, ...)
├── evaluation/              # judge config (the evaluator follows from the benchmark)
├── llm/                     # per-model request and serving overrides
├── analysis/                # analysis-tool configs (guard-threshold sweep, ...)
├── clusters/example.yaml    # template SLURM profile
└── cluster_pool.example.yaml

src/
├── experiment/              # orchestrator (experiment.py), task dispatcher (task.py, four modes), judging,
│                            #   rejudge stage, model discovery, multi-cluster split, Pydantic schemas
├── prompt_transformations/  # one registry: text/ encoders and image/ renderers
├── attacks/                 # adaptive pipeline attack
├── defense/                 # all defenses, shared guard utilities, vendored diffusion code for CIDER
├── evaluation/              # HarmBench, JailbreakBench, JailbreakBench-refusal and OR-Bench judges, WildGuard lens
├── analysis/                # ensemble ASR (portfolio.py), paired statistics, severity grading,
│                            #   guard-threshold sweeps, judge disagreement and bias checks
└── utils/                   # logger, MLflow tracker, provenance helpers

data/                        # benchmark prompts (HarmBench, JailbreakBench, OR-Bench, neutral controls)
tests/                       # hermetic tests (no network, no keys)
```

## Experiment structure

**Presets.** An experiment round is one YAML preset under `conf/experiment/<namespace>/`, and `python main.py <namespace>/<preset>` runs it. A preset lists tasks. Each task names its mode, benchmark, models, transformation chain or defense, and the prompt range. Encoder, renderer, defense and judge settings are loaded from the config folders above, so a preset stays short.

**Modes.** `src/experiment/task.py` dispatches on the task mode:

- `prompt_transform` runs a chain of transformation steps. Each step writes its own folder with a cumulative `results.json`. The input is a dataset file or an earlier step, so one encoding is produced once and shared by every condition that uses it.
- `defense+evaluate` applies a defense, queries the target and judges the answer, in one task.
- `analyze` is pure post-processing over many finished runs: ensemble ASR, paired statistics, complementarity between attacks.
- `rejudge` scores stored responses again with a different judge, without querying the target.

**Stages chain by reference.** A later task points at an earlier output folder (`source_transform_subdir`). All conditions of a comparison therefore share the same encoded prompts, which makes the comparisons paired. Presets in this repository keep the output paths of the original runs. To reproduce a result, run the transform stage first, then point the evaluation preset at your own output folder.

**Output layout.**

```
outputs/<namespace>/<mode>/<benchmark>/<short_name>_<timestamp>_<rand>/
├── results.json         # config, metrics, primary metric, git sha, upstream reference
├── prompts.jsonl        # per-prompt records (src/experiment/schemas.py)
├── raw_results.jsonl    # per-prompt response, judge output, judge reasoning, raw judge response
└── images/              # rendered images (image tasks only)
```

Every `results.json` carries an upstream reference (source folder and content hash), so the full provenance of any number can be rebuilt and upstream drift is detected by hash. The complete judge trail is stored per prompt.

**Tracking.** Each task is an MLflow run with parameters, metrics and artifacts, stored locally under `mlruns/` (`mlflow ui` to browse). Token and cost usage per run is recorded in `results.json`.

**Reproducibility notes.**

- *Pairing.* All variants in a (model, defense, encoding) cell share one encoded text, produced once and reused.
- *Empty responses.* The judge counts an empty response as a refusal. This handles providers that block an encoding at the API layer.
- *Schema versioning.* Every `results.json` carries `schema_version`, `git_sha` and `git_dirty`.
- *Judge integrity.* A silently failing judge is the main threat to these numbers. A nonzero `fallback_parse_count` voids a cell, and a wall time far below expectation means the judge failed at once.
- *Ranges.* `prompt_range: [start, end]` is inclusive on both ends and 0-indexed.

## Installation and running

**Code release.** This release is the harness state used for the arXiv versions listed above (snapshot 2026-08-09). From the repository root, use Python 3.12 or 3.13 and run:

```bash
uv sync --locked
uv run python -c "import src; from src.defense import defender_factory; from src.prompt_transformations import transformation_factory"
# Export your chosen provider's API environment variables, then:
uv run python main.py test
```

The `test` preset makes encoder API calls for two prompts (about $0.01). Reproduction presets live under `conf/experiment/`; evaluation presets require the output paths produced by their earlier stages. Cluster runs require your own SLURM wrapper and configuration based on `conf/clusters/example.yaml` and `conf/cluster_pool.example.yaml`. Dataset terms are in [DATA_LICENSES.md](DATA_LICENSES.md).


**Models.** Open-weight targets, guards and judges are served with vLLM on a SLURM cluster. API models (OpenAI, Anthropic, Google, AWS Bedrock, OpenAI-compatible providers, local Ollama) sit behind one call shape. The model registry, provider routing and pricing come from the pinned public package [`llm_utils`](https://github.com/vacantfury/llm_utils). Per-model overrides live in `conf/llm/<model>.yaml`. API keys are read from environment variables, listed in `.env.example`.

## Ongoing work

Work continues on how guardrails behave across the text and image channels and on how their safety is measured.

## Citation

```bibtex
@InProceedings{pmlr-v318-zhang26a,
  title     = {Exposing LLM Safety Gaps Through Mathematical Encoding: New Attacks and Systematic Analysis},
  author    = {Zhang, Haoyu and Zandsalimy, Mohammad and Sushmita, Shanu},
  booktitle = {Proceedings of the The 39th Canadian Conference on Artificial Intelligence},
  pages     = {662--677},
  year      = {2026},
  editor    = {Bouzar-Benlabiod, Lydia and Leung, Carson},
  volume    = {318},
  series    = {Proceedings of Machine Learning Research},
  month     = {25--29 May},
  publisher = {PMLR},
  pdf       = {https://raw.githubusercontent.com/mlresearch/v318/main/assets/zhang26a/zhang26a.pdf},
  url       = {https://proceedings.mlr.press/v318/zhang26a.html},
}
```

## License

The code is released under the MIT license. The evaluation prompts keep the licenses of their original datasets, see [DATA_LICENSES.md](DATA_LICENSES.md).

## Contact

Haoyu Zhang. Please open an issue on this repository.
