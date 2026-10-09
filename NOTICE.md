# Data, models, and third-party attribution

The MIT license applies to this project's code and documentation, subject to any separately identified third-party material. It does not relicense external datasets, models, or installed dependencies.

## BIRD

Dataset and benchmark: Li et al., *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*, [arXiv:2305.03111](https://arxiv.org/abs/2305.03111).

- Official site: https://bird-bench.github.io/
- Fixed upstream license/source reference: https://github.com/AlibabaResearch/DAMO-ConvAI/blob/188835d4f9948563a6b9c8ac50cd0f3ae4021ed6/bird/README.md
- The referenced BIRD dataset license is [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- Local evaluator behavior is documented against the upstream reference; it is not a claim to run the complete official harness.

No raw questions, gold SQL, schemas, database files, or dataset archives are bundled. Download resources from their official source and comply with the applicable dataset terms. Public artifacts contain aggregate statistics and question identifiers, with provenance and transformation notes.

## Qwen models and libraries

The recorded student and teacher were Qwen3.5-9B and Qwen3.8-27B, respectively, obtained as verified ModelScope snapshots. Model weights and adapters are not distributed here. Check the exact upstream model card and license when obtaining or redistributing any model.

PyTorch, Transformers, PEFT, TRL, vLLM, FLA, CUDA/NCCL, and other dependencies retain their own licenses. Installing an optional dependency does not place that dependency under this repository's MIT license.
