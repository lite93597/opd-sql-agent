# Configuration map

| File / entry | Purpose |
|---|---|
| `endpoint_example.json` | User-provisioned OpenAI-compatible SQL inference endpoint; defaults include repair, unlike the final zero-repair evaluation. |
| `local_smoke.json` | Small local pipeline smoke configuration with a machine-specific model path. It is not an OPD capability result. |
| `baseline_7b.json`, `sft_7b.json`, `opd_7b.json` | Historical Qwen2.5 / 7B reference configurations. The OPD reference uses a different divergence recipe. They must not be used to explain the published 9B / 27B reverse-KL result. |
| `server/deployment.json`, `server/opd_smoke.json` | Original Linux deployment and engineering-verification configuration; review paths and verified model metadata before use. |
| `scripts/server/run_effect_experiment.py` | Generates the actual SFT / OPD experiment stage configurations and freezes the final selection. |

The recorded experiment settings and prerequisites are documented in [reproduction](../docs/reproduction.md). Paths in older examples reflect their original environment and need adapting. An API key is read via the named environment variable, never embedded in configuration.
