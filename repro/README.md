# DQNSelector clean reproduction

This directory is a clean-room implementation guided by the SIGIR 2024 paper rather than by the legacy training script. The legacy code under `../code/` is retained only for comparison.

## Reproduction goals

1. Implement the IC propagation model and the ECM objective in Eqs. (1)--(6).
2. Implement PIIC exactly from Eqs. (10)--(11) / Algorithm 2 and test it on graphs with hand-computable paths.
3. Learn the two directed social-influence embeddings from Monte-Carlo activation probabilities as in Eqs. (7)--(9).
4. Implement the gated dual-embedding fusion in Eqs. (12)--(14).
5. Implement a Rainbow-style selector with Double DQN target selection, dueling value/advantage streams, prioritized replay, n-step returns, distributional C51 learning, and NoisyNet exploration.
6. Keep a strict distinction between (a) paper-defined quantities, (b) assumptions required because the paper omits a detail, and (c) compatibility behavior needed to read the released processed files.

## Quick start

```bash
cd repro
python -m pip install -e '.[dev]'
pytest
python scripts/inspect_legacy.py
python scripts/train_synthetic.py
```

## Important limitation

The paper does not fully specify the raw Gowalla/Brightkite-to-instance transformation. In particular, the exact mapping from check-in counts to each subarea demand `d_i`, the random seeds/splits used to sample users/worker pools/target subareas, and the full set of processed 3000/5000-user instances are not present in the public repository. Therefore exact numerical reproduction of Figure 5 / Tables 2--5 cannot be claimed until those data-generation ambiguities are resolved. The code in this directory makes these gaps explicit instead of silently guessing them.

See `../docs/REPRODUCTION_SPEC.md`, `../docs/LEGACY_GAPS.md`, and `../docs/DATA_RECONSTRUCTION.md`.
