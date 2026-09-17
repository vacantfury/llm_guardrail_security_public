# LLM Guardrail Security

Research on the security of the **guardrail layer**: the safety mechanisms deployed around large language models and vision-language models, such as guard classifiers, input and output filters, and caption or decode pipelines. The work studies how encoded and image-rendered jailbreak prompts pass these guards, and how black-box defenses can close the gap.

## Papers

| Paper | Where |
|---|---|
| *Exposing LLM Safety Gaps Through Mathematical Encoding: New Attacks and Systematic Analysis* | Canadian AI 2026, PMLR v318, pp. 662–677. [Proceedings](https://proceedings.mlr.press/v318/zhang26a.html) |
| *Attack Ensembles Expose a Safety-Utility Trade-off in Black-Box Guard Defenses Against Encoded VLM Jailbreaks* | [arXiv:2607.26574](https://arxiv.org/abs/2607.26574) |
| *Borrowed Strength: Best-of-N Search over a Code Encoding Breaks Self-Check Jailbreak Defenses* | [arXiv:2607.26639](https://arxiv.org/abs/2607.26639) |
| *Decoy Images Amplify Caption-Mediated Defenses Against Encoded Jailbreaks* | [arXiv:2608.01043](https://arxiv.org/abs/2608.01043) |
| *Whose Refusal Is It? The Unmeasured Contribution of Black-Box Multimodal Guardrails* | [arXiv:2608.08641](https://arxiv.org/abs/2608.08641) |

## The research harness

All papers above share one experiment harness:

- **Attacks.** Text encoders that rewrite a request into another representation (set theory, formal logic, ciphers, code, classical languages) and image renderers that place the text into an image (typography, flowcharts, low-contrast and occluded renders). Steps chain, so any encoding can be rendered into an image.
- **Defenses.** Published black-box defenses and guard classifiers behind one interface, plus a recover, decode, then guard pipeline that lets a guard see an obfuscated payload before it decides.
- **Measurement.** A target model answers, and an external judge scores harm (HarmBench rubric) and over-refusal (OR-Bench rubric). The headline metric is ensemble attack success: a behavior counts as broken if any attack in the suite breaks it.

**Code release.** This release is the harness state used for the arXiv versions listed above (snapshot 2026-08-09). From the repository root, use Python 3.12 or 3.13 and run:

```bash
uv sync --locked
uv run python -c "import src; from src.defense import defender_factory; from src.prompt_transformations import transformation_factory"
# Export your chosen provider's API environment variables, then:
uv run python main.py test
```

The `test` preset makes encoder API calls for two prompts (about $0.01). Reproduction presets live under `conf/experiment/`; evaluation presets require the output paths produced by their earlier stages. Cluster runs require your own SLURM wrapper and configuration based on `conf/clusters/example.yaml` and `conf/cluster_pool.example.yaml`. Dataset terms are in [DATA_LICENSES.md](DATA_LICENSES.md).


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

The code will be released under the MIT license. The evaluation prompts keep the licenses of their original datasets (HarmBench, JailbreakBench, OR-Bench).

## Contact

Haoyu Zhang. Please open an issue on this repository.
