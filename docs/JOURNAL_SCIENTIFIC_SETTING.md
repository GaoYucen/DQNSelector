# Journal scientific setting v1

This branch intentionally separates two goals:

- `reproduction-v2`: clean implementation of the SIGIR 2024 formulation and explicit reconstruction assumptions.
- `journal-scientific-v1`: a scientifically calibrated setting for the journal extension. It does **not** claim exact numerical reproduction of the conference experiments.

The canonical machine-readable profile is `repro/configs/journal_scientific_v1.json`.

## Why the legacy-style reconstruction saturates

The ECM score clips each subarea at `min(coverage / demand, 1)`. In the first reconstructed benchmark, three choices jointly made the objective too easy:

1. participation and quality were identical (`q = p`);
2. `max_normalized` demand was only around unit scale while coverage sums contributions over many activated users;
3. social-edge probabilities as high as 0.5 caused strong cascades.

The resulting benchmark reached EC=1.0 for very small budgets, so further RL training could not be meaningfully evaluated.

## Scientific setting v1

### Participation

Participation is treated as a spatial willingness / availability proxy:

`p(v,i) = exp(-dist(v,i) / tau)`

where `dist(v,i)` is the minimum historical distance between worker `v` and target subarea `i`. `tau` is a robust quantile of all positive worker-target distances (median by default). This avoids normalizing every probability by one global maximum distance.

### Quality

The public SNAP check-in data do not provide sensing-quality labels. Quality is therefore represented by an explicit worker-level reliability proxy derived from historical activity:

1. take `log(1 + in-region check-in count)` for each sampled worker;
2. robustly normalize between the 10th and 90th percentiles;
3. map the result to `[quality_floor, 1]` (default floor 0.25).

This deliberately decouples quality from participation. Quality is constant across target subareas in v1 because there is no ground-truth target-specific quality signal in the source data.

### Demand and calibrated load regimes

Raw check-in activity determines **relative spatial task intensity**, while an explicit load factor determines the **global demand scale**.

For each target subarea, activity weight is proportional to:

`(checkin_count + 1)^activity_power`

and normalized to mean 1. The global scale is the median direct capacity of the candidate worker pool, where direct capacity is the sum of `p*q` without social diffusion. Demand is then:

`d_i = load_factor * median_candidate_capacity * activity_weight_i`

with a small positive floor. This calibration does not force demand to match each area's local supply, so genuine spatial supply-demand mismatch remains in the problem.

A 3000-user calibration sweep on both Gowalla and Brightkite established the following regimes:

- `rho=0.75`: easy; useful for sensitivity but begins to saturate at larger budgets.
- `rho=1.00`: moderate.
- `rho=1.25`: **main scientific-v1 regime**. It retains strong method separation over the primary budget range while keeping saturation controlled.
- `rho=1.50`: hard-load sensitivity regime.

The primary paper budgets are `k = 5, 10, 20, 30, 40, 50`; `k = 75, 100` are stress tests rather than the default comparison range.

At `rho=1.25, k=50`, the calibration run gave CoverageGreedy EC / saturated-subarea fraction of approximately `0.742 / 0.11` on Gowalla and `0.787 / 0.16` on Brightkite, while the corresponding random EC means were approximately `0.477` and `0.359`. This is deliberately different from the degenerate reconstructed setting where EC was already 1.0 at very small budgets.

### Social influence

The friendship graph does not provide empirical diffusion probabilities. Scientific v1 therefore uses an explicit low-probability trivalency IC protocol with edge probabilities sampled from `{0.01, 0.05, 0.10}`. These probabilities must be treated as a modeling protocol, not measured social influence, and should be sensitivity-tested.

### Objective-aware coverage representation

The legacy PIIC path used `quality` alone as its coverage input. That is retained only as a compatibility ablation. In the scientific setting the direct pre-clipping EC contribution of worker `v` to subarea `i` is

`u(v,i) = p(v,i) * q(v,i) / d_i`.

The main coverage representation therefore propagates `u` through PIIC and **includes the worker's direct term** as well as the propagated term. This makes the representation aware of participation and heterogeneous demand, and prevents isolated but directly useful workers from receiving a zero coverage feature.

`precompute_embeddings.py --coverage-feature auto` selects this objective-aware representation automatically for `journal_scientific_v1`; `legacy-quality` remains available for ablation.

### Fixed-world oracle semantics

The ECM definition is nonlinear because demand clipping is applied after expected coverage is formed. The canonical computation order is:

1. estimate each user's IC activation probability `p_v(S)` across Monte-Carlo/live-edge worlds;
2. compute expected subarea coverage `C_i(S) = sum_v p_v(S) p_v^i q_v^i`;
3. compute `EC_i(S) = min(C_i(S)/d_i, 1)`;
4. average `EC_i(S)` over subareas.

`LiveEdgeECOracle` must follow this order even though it reuses fixed live-edge worlds for deterministic repeated queries. Averaging `min(C_i^world/d_i,1)` over worlds is **not** equivalent and is not the paper-defined ECM objective. A regression test now enforces exact agreement between `LiveEdgeECOracle.score(S)` and an explicit activation-probability-then-clipping computation on the same fixed worlds.

The load-regime calibration generated by `analyze_saturation.py` already used the correct order and remains valid. DQN/CELF runs produced with the earlier per-world-clipping oracle are non-canonical and must not be used in journal results; corrected training/evaluation starts from the oracle semantic fix.

### Baseline policy

The journal benchmark distinguishes compatibility baselines from scientifically aligned comparators:

- `DegreeGreedy`: conference-style structural baseline.
- `CoverageGreedy`: conference-compatible one-hop coverage baseline; its historical behavior does not add the candidate's own direct contribution.
- `ObjCovGreedy`: journal scientific static heuristic. It ranks each candidate using its **direct contribution plus expected one-hop contribution**, normalized by heterogeneous demand and capped per subarea. It is intentionally kept separate rather than silently changing `CoverageGreedy`.
- `CELF`: objective-aware marginal-gain oracle baseline using fixed selection worlds and independent final-evaluation worlds.
- `DQNSelector`: learned selector.
- `Random`: repeated random selection for a distributional reference rather than a single random draw.

This prevents the learned method from being compared only against a compatibility heuristic that is misaligned with the calibrated objective.

## Required diagnostic before RL training

Every generated instance should pass `scripts/analyze_saturation.py` before embeddings or DQN training. Report at least:

- EC versus budget;
- fraction of saturated subareas;
- coverage/demand ratio distribution;
- expected number of activated users;
- random, degree-greedy, and coverage-greedy curves.

A useful instance should retain visible marginal gains across the target budget range. If random or simple greedy methods reach EC near 1 at very small `k`, adjust the load / propagation regime before training RL.

## Reproducibility and evaluation rules

- Instance-generation choices are recorded in each manifest.
- Training and final evaluation use independent live-edge Monte-Carlo worlds.
- Final journal experiments should use multiple instance seeds; the canonical profile records seeds 2024--2028.
- Main results should report both EC and saturation fraction, rather than EC alone.
- Runtime comparisons must separate preprocessing/training time from online selection time.
- Every DQN/CELF result must be generated after the oracle regression gate passes; invalidated pre-fix runs are not mixed into tables.

## Planned setting ablations

The journal study should isolate the source of the conference-style saturation by changing one component at a time:

1. legacy `p=q` vs decoupled quality;
2. legacy demand transforms vs load-calibrated demand;
3. strong uniform diffusion vs low-probability trivalency;
4. legacy quality-only PIIC vs objective-aware `p*q/d` coverage features;
5. combinations of the above.

This lets the paper distinguish an algorithmic effect from an artifact of problem scaling.
