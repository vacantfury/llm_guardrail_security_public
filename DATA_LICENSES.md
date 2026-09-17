# Dataset sources and licenses

The MIT license in LICENSE covers this repository's code. Dataset content
retains its upstream terms. Files here are prompt-only JSONL conversions and
subsets, including benchmark splits and a neutral Alpaca sample. Attribution
and source links follow. License declarations checked on 2026-09-17.

| Dataset | Source and license | Local files |
|---|---|---|
| HarmBench (Mazeika et al.; Center for AI Safety) | [Official repository license: MIT](https://github.com/centerforaisafety/HarmBench/blob/main/LICENSE) | `data/harmbench*.jsonl` |
| JailbreakBench / JBB-Behaviors (Chao et al.; JailbreakBench team) | [Official dataset card: MIT](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/blob/main/README.md) | `data/jbb*.jsonl` |
| OR-Bench contributors | [Official dataset card: CC BY 4.0](https://huggingface.co/datasets/bench-llm/or-bench/blob/main/README.md), [license terms](https://creativecommons.org/licenses/by/4.0/) | `data/orbench*.jsonl` |
| Stanford Alpaca (Taori et al.) | [Official data license: CC BY-NC 4.0](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE) | `data/alpaca_neutral_100_prompts.jsonl` |

Alpaca's dataset license restricts commercial use. Its code license is a
different license and does not replace the data license. OR-Bench requires
attribution and an indication of changes. The conversions and subset
selection described above are changes from the upstream distributions.
Upstream dataset declarations were verified; exact upstream revisions of
these historical local subsets were not recorded in this release.

HarmBench's notice is Copyright (c) 2024 centerforaisafety. JailbreakBench's
code repository notice is Copyright (c) 2023 JailbreakBench Team. For their
MIT-covered material, the permission and warranty terms in LICENSE apply
with the respective upstream copyright notice retained here.
`data/smoke_innocuous.jsonl` contains synthetic smoke prompts for this harness.

## Images

`src/prompt_transformations/image/images/rabit.jpeg` in this release is a
placeholder outline drawing made for this repository (MIT, same size as the
original). The decoy drawing used in the papers' experiments is not
redistributed here because its art license could not be verified. The
constant-image conditions use a content-unrelated decoy, so any such image
can take its place; exact pixel-level reproduction of those conditions
needs the original, available from the author on request.
