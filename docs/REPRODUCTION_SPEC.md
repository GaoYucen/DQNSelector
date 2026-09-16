# DQNSelector reproduction specification

This document treats the SIGIR 2024 paper as the normative algorithmic specification. The released `code/` directory is used only as historical evidence when a paper detail is missing.

## 1. Problem definition

The social network is a directed weighted graph `G=(V,E,W)`, with edge weight `w(u,v)` interpreted as an IC activation probability. For a seed set `S`, `p_v(S)` denotes the probability that node `v` becomes active under Independent Cascade propagation.

For subarea `a_i`, node `v` has participation probability `p_v^i`, sensing quality `q_v^i`, and the subarea has demand `d_i`. The expected node contribution is

`q_v(a_i,S) = p_v^i * p_v(S) * q_v^i`.

The expected total coverage is

`C(a_i,S) = sum_v q_v(a_i,S)`.

Effective coverage is capped by demand:

`EC(a_i,S) = min(C(a_i,S)/d_i, 1)`

and the objective is the mean over target subareas.

Implementation: `repro/src/dqnselector/ecm.py`.

## 2. IC probability estimation

The paper states that exact effective-coverage computation is #P-hard and uses Monte-Carlo simulation for influence probabilities in the embedding stage. The clean reproduction therefore provides a stochastic IC simulator and MC estimator in `influence.py`.

A separate `layered_product_activation_probabilities` function is retained only to reproduce the behavior of the released script; it is not treated as the exact IC probability on arbitrary cyclic graphs.

## 3. Social influence embedding

For seed node `v` and target node `u`, two directed embeddings are learned:

- `s_v`: influence-as-seed representation;
- `t_u`: susceptibility / being-influenced representation.

The target is the single-seed activation probability `p_u({v})`, estimated by Monte Carlo. The paper minimizes squared error between `s_v^T t_u` and that probability, using balanced positive and negative samples.

Implementation: `embedding.py`.

Paper defaults recorded for full experiments: embedding dimension 128, MC times 100, graph iterations 30.

## 4. PIIC sensing coverage embedding

The paper defines `Delta r_v^0 = q_v` and the recurrence

`Delta r_v^L = sum_{l in N_v^+} w(v,l) Delta r_l^(L-1)`.

The final coverage embedding is the sum of increments from layer 1 through influence range `R`. With weighted adjacency matrix `A`, this is implemented as repeated multiplication `Delta <- A @ Delta` followed by accumulation.

Implementation: `piic.py`.

On a DAG, this is exactly equivalent to explicit weighted path enumeration. The test suite verifies that equivalence. On cyclic graphs the published recurrence counts walk-like repeated contributions; the paper itself calls PIIC an estimate in the presence of cycles. We do not silently invent an unreported cycle-correction rule.

## 5. Gated dual embedding

The paper does not merely concatenate the two embeddings. It defines

`G_s = sigmoid(W1 s_v + W2 r_v + b1)`

`G_r = sigmoid(W3 s_v + W4 r_v + b2)`

`x_v = [G_s * s_v || G_r * r_v]`.

Implementation: `fusion.py`. The gate is trainable jointly with the selector.

## 6. Rainbow selector

The clean implementation contains the Rainbow mechanisms explicitly named in the paper:

- Double DQN action selection;
- dueling value/advantage decomposition;
- prioritized replay;
- n-step returns;
- categorical distributional RL (C51);
- NoisyNet exploration;
- target network synchronization.

Because the action is a worker representation rather than a fixed action ID, the Q network is pair-conditioned on `(state, candidate worker)` and performs dueling centering across the current candidate set. The state is the sum of the currently selected gated worker representations, matching the paper.

Implementation: `rainbow.py` and `selector.py`.

## 7. Reward

The reward is the marginal effective-coverage gain

`EC(S union {v}) - EC(S)`.

The clean training API takes a reward oracle, so the full ECM evaluator can be used directly while synthetic tests can use deterministic objectives.

## 8. What is not yet claimable

A faithful algorithmic implementation is possible from the paper, but exact reproduction of Figure 5 and Tables 2--5 additionally requires the exact generated instances. The public repository exposes only one 3000-user processed sample and does not expose the random seeds, the full 3000/5000 instance collection, or enough information to uniquely reconstruct separate `p_v^i`, `q_v^i`, and `d_i` from the stored node vectors.

Therefore the reproduction should distinguish three milestones:

1. **Algorithm verified:** equations and components pass synthetic/unit tests.
2. **Legacy compatibility verified:** the released processed instance can be loaded and diagnosed.
3. **Paper-number reproduced:** only after the missing data-generation details are reconstructed or recovered.
