# A40 comparison with the original SIGIR baselines

This run answers the user's request for measured DQNSelector quality against all
six conference baselines. It does not claim recovery of the unpublished original
processed instances or equality with the conference's absolute EC numbers.

## Fixed experiment

- Base code: 6a6364640d0cc4a376827ca013de4fb432159e8a (scientific-v3).
- Data: unchanged scientific-v2 reconstruction (rho=1.5), Gowalla/Brightkite,
  structure seed 2024, task seed 2024, 3000/5000 nodes, 300 eligible workers.
- Budgets: 50,60,70,80,90,100, matching SIGIR Fig. 5.
- First results are instance-specific offline training with independent IC worlds
  for training, checkpoint validation and final evaluation. They do not establish
  unseen-task generalization. Results must identify this scope prominently.
- Training seeds: 2024,2025,2026. Baseline results are not duplicated as independent
  observations when aggregating across training seeds.
- Train worlds:100; validation worlds:300; final worlds:1000. All methods share the
  same final evaluator for an instance, which is never used to choose checkpoints.
- The primary metric is the conference EC (cap after expected coverage), not the
  expectation of capped realized coverage. Report both under separate names.
- Quality target: DQNSelector/best original baseline >=.97, both per dataset and
  in the macro average; also report every individual budget and worst-case ratio.
  This is a target, not a rule for omitting unsuccessful results.

## Baseline sources and adapters

1. DegGreedy: out-degree ranking within the common worker pool.
2. CovGreedy: SIGIR Sec.5.1 one-hop sum of w(v,u)*p(u,i)*q(u,i); no demand-aware
   enhancement or inclusion of the candidate's direct term.
3. CELF: optimize canonical EC with 100 fixed IC worlds; final evaluation uses
   independent 1000 worlds. Report oracle construction separately from selection.
4. FastSelector: SIGIR Sec.5.1 rank combination, alpha=.56 Gowalla/.64 Brightkite.
   Average cosine similarity to selected trajectories follows Wang et al. Eq.(11),
   https://arxiv.org/abs/1805.08525. Raw in-region check-ins form 7200-cell mobility
   count vectors (one aggregate temporal period). Unlike released SIGIR code, the
   selected trajectory statistic updates after every choice and favors diversity.
   The empty-set first choice is maximum degree. This is the SIGIR rank adaptation,
   not the original paper's separate participant-budget two-phase algorithm.
5. KTVoting: released SIGIR code/code KTVoting.py recurrence, beta=.9, T=4. Use
   k groups of areas, complete area partition, per-group election and score-ranked
   filling. Repairs: restrict candidates to U, never insert dummy node 0, enforce
   exactly k unique workers, include remainder areas. The released KTVoting2 path
   can exceed k; it is not used as an infeasible comparator.
6. PIANO: explicit PyTorch reimplementation of Li et al. TCSS Eqs.(1)-(3), using
   structure2vec with weighted edges and selected-node flags, a global-sum and
   local-node Q readout, epsilon-greedy n-step DQN and influence-spread reward.
   Sources: https://sunchangsheng.com/assets/2022%20TCSS%20PIANO.pdf and
   https://github.com/lihuixidian/PIANO (00be744). Public code differs from the paper
   (residual-graph all-one features, singleton rather than marginal reward, and
   repeated RNG reseeding); prefer the published equations. Defaults:64 dimensions,
   4 message rounds, gamma=.95, n-step=5, batch=64, learning rate=.001. Use Adam as
   in the public implementation. Use the raw marginal influence-spread reward. Restrict
   selectable workers to U. Report as "PIANO (paper reimplementation)".

   Training adapters must also be disclosed: use the complete fixed 3000-node
   instance instead of sampled training subgraphs; 200 episodes at k=50 (10000
   transitions), with epsilon declining from 1 to .05 over 10000 steps. Decay the
   learning rate by .95 every 1000 optimizer updates; the decay interval is an
   explicit implementation choice because it is unspecified in the paper. The
   initial 5000-step normalized-reward pilot was replaced before PIANO training.
   This is a paper-equation reproduction with stated adapters, not a claim of
   recovering the unavailable original SIGIR implementation.

## DQNSelector training

Retain the v3 gated dual embedding and Rainbow architecture. Set the random seed
before model initialization (the earlier CLI only seeded inside the trainer).
The first complete table uses the paper-horizon setting (100 episodes at k=50).
A separately named multi-budget setting (200 episodes balanced over 50..100) is
available for follow-up if needed. Checkpoints are selected
on independent validation worlds, never on final evaluation. Further variants,
if needed, must have distinct configuration files and labels; do not relabel
imitation or greedy hybrids as the unmodified DQNSelector.

## Timing and provenance

Record source commit, instance and checkpoint SHA256, device/runtime, all seeds,
training/validation/selection/evaluation MC counts, selected node IDs, training
time, model/data preparation and per-budget selection time with CUDA synchronization.
CPU heuristics and CUDA neural methods use the same A40 host; report hardware.
The 4090 original remains intact. All A40 jobs use project DQNSelector and its
Control-v2 local/global leases.
