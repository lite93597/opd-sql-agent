# Contributing

Start with the CPU demo and the test commands in [examples/README.md](examples/README.md). A normal development install is:

```bash
python -m pip install -e '.[dev]'
```

Useful changes include evaluator edge cases, portable launch configuration, synchronization efficiency backed by correctness checks, and independently repeated experiments.

For a code change, describe the concrete failure or behavior it addresses and run the relevant CPU tests. GPU training is optional and must not be triggered as part of a routine documentation or evaluator check. Numerical loss changes should compare loss and gradients with a dense reference.

For an experiment report, disclose models and revisions, dataset version and split hashes, checkpoint selection policy, seeds, loss direction, generation settings, optimizer/token/compute budgets, gold failures, and paired gained/lost. Keep unsuccessful candidates and do not choose models using final-test feedback.

Do not commit secrets, raw BIRD records, database files, model weights, checkpoints, private host identifiers, or deployment logs. The synthetic demo fixtures are safe substitutes for reproducing evaluator issues.

The recorded result is one controlled run, not a universal guarantee. Claims about faster training, cross-seed effects, or official leaderboard performance need separate evidence.
